import ast,importlib.util,json,os,pathlib,types,unittest
import torch
from sglang.test.test_utils import CustomTestCase

ROOT=pathlib.Path(os.environ.get('IFB_STAGE',str(pathlib.Path(__file__).resolve().parents[2])))
spec=importlib.util.spec_from_file_location('special_int8_under_test',ROOT/'hcu_sglang/layers/quantization/channelwise_int8.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)

def reference(x,w,s,g,n,out=None):
    z=x.float();amax=z.abs().amax(-1,keepdim=True).clamp_min(1e-10)
    v=z*(127/amax)
    q=torch.where(v<0,-torch.floor(-v+0.5),torch.floor(v+0.5)).clamp(-127,127).to(torch.int32)
    y=[]
    for i in range(g):
        raw=q[:,i].cpu() @ w[i*n:(i+1)*n].int().cpu().t()
        y.append(((raw.float().to(x.device)*(amax[:,i]/127))*s[i*n:(i+1)*n].t()).to(out or x.dtype))
    return torch.stack(y,1)

class TestSpecialInt8(CustomTestCase):
    def test_grouped_cpu(self):
        torch.manual_seed(42)
        for dtype in (torch.float32,torch.bfloat16,torch.float16):
            for m in (1,6,64):
                x=torch.randn(m,2,32).to(dtype);w=torch.randint(-127,128,(64,32),dtype=torch.int8);s=torch.rand(64,1)+0.001
                torch.testing.assert_close(Q._grouped(x,w,s,2,32),reference(x,w,s,2,32),rtol=0,atol=0)
        x=torch.tensor([[[127.,0.5,-0.5,0.]]]);w=torch.tensor([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1],[0,0,0,0],[1,1,1,1],[1,0,0,0],[0,1,0,0]],dtype=torch.int8);s=torch.ones(8,1)
        y=Q._grouped(x,w,s,1,8);self.assertEqual(y[0,0,:3].tolist(),[1.,-1.,127.])

    def test_empty_and_bad_layout(self):
        w=torch.ones(32,32,dtype=torch.int8);s=torch.ones(32,1)
        self.assertEqual(Q._grouped(torch.empty(0,1,32),w,s,1,32).shape,(0,1,32))
        torch.testing.assert_close(Q._grouped(torch.zeros(1,1,32),w,s,1,32),torch.zeros(1,1,32))
        for ww,ss in ((w.float(),s),(w,s.double()),(w,s.view(1,32))):
            with self.assertRaises(ValueError):Q._grouped(torch.ones(1,1,32),ww,ss,1,32)

    def test_install_and_tp_loader(self):
        for rank in (0,7):
            layer=torch.nn.Linear(32,16,bias=False)
            def loader(p,v):p.data.copy_(v.narrow(0,rank*16,16))
            layer.weight.output_dim=0;layer.weight.weight_loader=loader
            Q.install(layer)
            w=torch.arange(128*32,dtype=torch.int32).view(128,32).remainder(255).sub(127).to(torch.int8)
            s=torch.arange(1,129,dtype=torch.float32).view(-1,1)/100
            layer.weight.weight_loader(layer.weight,w);layer.weight_scale.weight_loader(layer.weight_scale,s)
            layer.quant_method.process_weights_after_loading(layer)
            torch.testing.assert_close(layer.weight,w[rank*16:(rank+1)*16],rtol=0,atol=0)
            torch.testing.assert_close(layer.weight_scale,s[rank*16:(rank+1)*16],rtol=0,atol=0)
            self.assertEqual(layer.weight.input_dim,1);self.assertEqual(layer.weight_scale.output_dim,0)
            x=torch.randn(3,32)
            torch.testing.assert_close(layer.quant_method.apply(layer,x),Q.linear(x,layer.weight,layer.weight_scale),rtol=0,atol=0)
            with self.assertRaises(ValueError):layer.weight.weight_loader(layer.weight,w.float())
        missing=torch.nn.Linear(32,16,bias=False);Q.install(missing)
        with self.assertRaises(ValueError):missing.quant_method.process_weights_after_loading(missing)

    def test_actual_markov_consumer(self):
        src=(pathlib.Path(importlib.util.find_spec('sglang').submodule_search_locations[0])/'srt/models/deepseek_v4_dspark.py').read_text(encoding='utf-8')
        tree=ast.parse(src);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DSparkV4MarkovHead')
        names={'project_bias','step_bias'}
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
        ns={'torch':torch,'F':torch.nn.functional};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+methods,type_ignores=[])),'actual_markov_consumer','exec'),ns)
        layer=torch.nn.Linear(32,64,bias=False);Q.install(layer)
        layer.weight.weight_loader(layer.weight,torch.randint(-127,128,(64,32),dtype=torch.int8));layer.weight_scale.weight_loader(layer.weight_scale,torch.rand(64,1)+0.001)
        latent=torch.randn(3,32);sentinel=object()
        h=types.SimpleNamespace(markov_w2=layer,_opt_markov_w2_bf16=False,_tp_shard=None,get_prev_embeddings=lambda ids:latent)
        h.project_bias=lambda v:Q.around_project_bias(ns['project_bias'],h,v)
        with self.assertRaises(RuntimeError):ns['project_bias'](h,latent)
        for bf16 in (False,True):
            h._opt_markov_w2_bf16=bf16
            for shard in (None,types.SimpleNamespace(org_vocab_start=16,org_vocab_end=32)):
                h._tp_shard=shard
                value,state=Q.around_step_bias(ns['step_bias'],h,torch.arange(3),hidden_states=None,state=sentinel)
                self.assertIs(state,sentinel)
                lo,hi=(0,64) if shard is None else (16,32)
                dtype=torch.bfloat16 if bf16 else torch.float32
                ref=Q.linear(latent.to(dtype) if shard is None else latent,layer.weight[lo:hi],layer.weight_scale[lo:hi],output_dtype=dtype)
                torch.testing.assert_close(value,ref.float() if shard is None else ref,rtol=0,atol=0)
        with self.assertRaises(ValueError):Q.around_project_bias(ns['project_bias'],h,latent,weight=layer.weight[:16])
        original=lambda *a,**kw:sentinel
        plain=types.SimpleNamespace(markov_w2=types.SimpleNamespace())
        self.assertIs(Q.around_project_bias(original,plain,latent),sentinel)
        self.assertIs(Q.around_step_bias(original,plain,None,hidden_states=None,state=None),sentinel)

    def test_public_logits_consumer(self):
        from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
        l=torch.nn.Linear(32,64,bias=False);Q.install(l)
        self.assertTrue(should_apply_lm_head_quant_method(l,l.quant_method))

    @unittest.skipUnless(os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS')=='1','explicit GPU opt-in')
    def test_gpu_graph_and_fixed_buffer_updates(self):
        torch.manual_seed(20261113)
        shapes=[(1,2,32,256),(6,1,64,256),(32,1,128,256),(1,8,1024,4096),(2,1,16160,256)]
        for m,g,n,k in shapes:
            x=torch.randn(m,g,k,device='cuda',dtype=torch.bfloat16)
            w=torch.randint(-127,128,(g*n,k),dtype=torch.int8,device='cuda');s=torch.rand(g*n,1,device='cuda')*.01+.001
            for _ in range(3):y=Q._grouped(x,w,s,g,n)
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):gy=Q._grouped(x,w,s,g,n)
            for update in range(3):
                if update:x.copy_(torch.randn_like(x));w.copy_(torch.randint_like(w,-127,128));s.copy_(torch.rand_like(s)*.01+.001)
                eager=Q._grouped(x,w,s,g,n);graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(gy,eager,rtol=0,atol=0)
                torch.testing.assert_close(gy,reference(x,w,s,g,n),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
