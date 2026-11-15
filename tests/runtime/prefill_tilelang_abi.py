import ast
import os
import pathlib
import types
import unittest
from unittest.mock import patch

import torch
from sglang.test.test_utils import CustomTestCase
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.attention import debug_flash_mla_adapter as A
from hcu_sglang.kernels import mhc as M

GPU = os.environ.get('IFB_PREFILL_GPU_LANE') == '1'
ROOT = pathlib.Path(__file__).resolve().parents[2]


def reference(q, kv, indices, scale, sink, lengths):
    ids = indices.reshape(q.shape[0], -1).long()
    keys = kv[:, 0, :][ids.clamp(0, kv.shape[0]-1)].double()
    valid = (ids >= 0) & (ids < kv.shape[0]) & (torch.arange(ids.shape[-1])[None, :] < lengths[:, None])
    scores = torch.matmul(q.double(), keys.transpose(1, 2)) * scale
    scores = scores.masked_fill(~valid[:, None, :], float('-inf'))
    maximum = scores.amax(-1)
    lse = torch.logsumexp(scores, -1)
    denominator = lse if sink is None else torch.logaddexp(lse, sink.double()[None, :])
    p = torch.exp(scores-denominator.unsqueeze(-1)).masked_fill(~valid[:, None, :], 0)
    out = torch.matmul(p, keys).to(q.dtype)
    return out, maximum.float(), lse.float()


class PrefillCPU(CustomTestCase):
    def case(self, n):
        g=torch.Generator().manual_seed(710+n)
        q=(torch.randn(n,8,512,generator=g)*.25).bfloat16()
        kv=(torch.randn(32,1,512,generator=g)*.25).bfloat16()
        ids=torch.randint(0,32,(n,1,7),generator=g,dtype=torch.int32)
        ids[:,:,-1]=33;ids[:,:,0]=-1
        lengths=torch.full((n,),6,dtype=torch.int32)
        sink=torch.randn(8,generator=g)*.2
        return q,kv,ids,lengths,sink

    def test_cpu_math_and_auxiliary_contract(self):
        for n in (1,6,65):
            q,kv,ids,lengths,sink=self.case(n)
            actual=A.torch_native_flash_mla_sparse_fwd(q,kv,ids,512**-.5,attn_sink=sink,topk_length=lengths)
            expected=reference(q,kv,ids,512**-.5,sink,lengths)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=.02,atol=.004)

    def test_empty_rows_and_invalid_indices(self):
        q,kv,ids,lengths,sink=self.case(2);ids.fill_(-1)
        for value in (float('-inf'),float('inf'),0.):
            sink.fill_(value)
            out,maximum,lse=A.torch_native_flash_mla_sparse_fwd(q,kv,ids,.1,attn_sink=sink,topk_length=lengths)
            self.assertTrue(torch.equal(out,torch.zeros_like(out)))
            self.assertTrue(torch.isneginf(maximum).all());self.assertTrue(torch.isneginf(lse).all())
        empty=A.torch_native_flash_mla_sparse_fwd(q[:0],kv,ids[:0],.1,attn_sink=sink,topk_length=lengths[:0])
        self.assertEqual(tuple(empty[0].shape),(0,8,512))

    def test_actual_caller_hcu_import_branches(self):
        for rel in ('hcu_sglang/models/deepseek/current_v4/backend.py','hcu_sglang/models/deepseek/current_v4/hcu_backend/backend.py'):
            tree=ast.parse((ROOT/rel).read_bytes())
            fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_forward_prefill_sparse')
            branch=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='_is_hcu')
            ns={'_is_hcu':True,'_is_xpu':False};exec(compile(ast.Module(body=[branch],type_ignores=[]),rel,'exec'),ns)
            self.assertIs(ns['flash_mla_sparse_fwd'],A.flash_mla_sparse_fwd_entrypoint)

    def test_native_selector_preserves_exception(self):
        from flash_mla import flash_mla_interface as vendor
        error=RuntimeError('native-original');q=object()
        with patch.dict(os.environ,{'SGLANG_HACK_FLASHMLA_BACKEND':'kernel'}),patch.object(vendor,'flash_mla_sparse_fwd',side_effect=error) as f:
            try:A.flash_mla_sparse_fwd_entrypoint(q)
            except RuntimeError as caught:self.assertIs(caught,error)
            else:self.fail('exception lost')
            self.assertIs(f.call_args.args[0],q)

    def test_lazy_mhc_compiler_flag_is_hcu_scoped(self):
        for hcu in (False,True):
            observed=[]
            class Jit:
                def jit(self,**kwargs):
                    observed.append(kwargs)
                    return lambda f:f
            value=object()
            with patch.object(M,'_load_tilelang',return_value=Jit()),patch.object(M,'_is_hcu',hcu):
                f=M._LazyTilelang().jit(pass_configs={'kept':True})(lambda x:x)
                self.assertIs(f(value),value)
            self.assertEqual(observed[0]['pass_configs'],{'kept':True})
            self.assertEqual(observed[0].get('compile_flags'),['-mcode-object-version=5'] if hcu else None)


@unittest.skipUnless(GPU,'explicit GPU lane')
class PrefillGPU(CustomTestCase):
    def test_sparse_eager_graph_buffer_updates(self):
        g=torch.Generator().manual_seed(903)
        for n in (1,6,65):
            q0=(torch.randn(n,64,512,generator=g)*.25).bfloat16();kv0=(torch.randn(128,1,512,generator=g)*.25).bfloat16()
            ids0=torch.randint(0,128,(n,1,17),generator=g,dtype=torch.int32);ids0[:,:,-1]=129;ids0[:,:,0]=-1
            lengths0=torch.full((n,),15,dtype=torch.int32);sink0=torch.randn(64,generator=g)*.2
            q=q0.cuda();kv=kv0.cuda();ids=ids0.cuda();lengths=lengths0.cuda();sink=sink0.cuda()
            fn=lambda:A.flash_mla_sparse_fwd_entrypoint(q,kv,ids,512**-.5,attn_sink=sink,topk_length=lengths)
            with patch.dict(os.environ,{'SGLANG_HACK_FLASHMLA_BACKEND':'torch'}):
                fn();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=fn()
                for state in range(3):
                    qc=(q0.float()*(1+state*.125)).bfloat16();kc=(kv0.float()*(1-state*.125)).bfloat16();q.copy_(qc);kv.copy_(kc)
                    eager=fn();graph.replay();torch.cuda.synchronize();expected=reference(qc,kc,ids0,512**-.5,sink0,lengths0)
                    for a,b,c in zip(eager,expected,captured):
                        torch.testing.assert_close(a.cpu(),b,rtol=.02,atol=.004);self.assertTrue(torch.equal(a,c))

    def test_mhc_post_codeobject5_eager_graph(self):
        self.assertTrue(M._is_hcu)
        g=torch.Generator().manual_seed(841)
        for n in (1,6,64):
            x0=(torch.randn(n,5120,generator=g)*.25).bfloat16();r0=(torch.randn(n,4,5120,generator=g)*.25).bfloat16()
            mix0=torch.randn(n,4,generator=g)*.1;comb0=torch.randn(n,4,4,generator=g)*.1
            x=x0.cuda();res=r0.cuda();mix=mix0.cuda();comb=comb0.cuda()
            with patch.object(M,'is_dsa_prefill_cp_interleave',return_value=False):
                fn=lambda:M.mhc_post(x,res,mix,comb)
                fn();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=fn()
                for state in range(3):
                    xc=(x0.float()*(1+state*.125)).bfloat16();x.copy_(xc)
                    expected=(mix0.double().unsqueeze(-1)*xc.double().unsqueeze(1)+torch.einsum('nij,nid->njd',comb0.double(),r0.double())).bfloat16()
                    eager=fn();graph.replay();torch.cuda.synchronize();torch.testing.assert_close(eager.cpu(),expected,rtol=.02,atol=.004);self.assertTrue(torch.equal(eager,captured))


if __name__=='__main__':unittest.main()
