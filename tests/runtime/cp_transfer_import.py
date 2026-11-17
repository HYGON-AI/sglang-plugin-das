import ast,pathlib,types,unittest
from sglang.test.test_utils import CustomTestCase
from sglang.srt.disaggregation.base.conn import StateType
from hcu_sglang.mem_cache.hcu_v4_pool import DeepSeekV4TokenToKVPool
from hcu_sglang.mem_cache.cp_cache_layer_split.transfer import match_transfer_entries

ROOT=pathlib.Path(__file__).resolve().parents[2]

class TestCPTransferImport(CustomTestCase):
    def pool(self,kv,states):
        p=object.__new__(DeepSeekV4TokenToKVPool)
        p._unified_kv=False;p.requires_descriptor_matched_transfer=True
        p.get_kv_transfer_layout=lambda:list(kv)
        p.get_state_transfer_layout=lambda:list(states)
        p.get_request_state_transfer_layout=lambda:[]
        return p

    def invoke(self,name,pool,draft,args):
        s=(ROOT/('hcu_sglang/disagg/'+name+'.py')).read_text()
        method=next(x for x in ast.walk(ast.parse(s)) if isinstance(x,ast.FunctionDef) and x.name=='_init_kv_manager')
        branch=next(x for x in method.body if isinstance(x,ast.If) and any(isinstance(y,ast.ImportFrom) and y.module and y.module.endswith('cp_cache_layer_split.transfer') for y in ast.walk(x)))
        module=next(y.module for y in ast.walk(branch) if isinstance(y,ast.ImportFrom))
        self.assertEqual(module,'hcu_sglang.mem_cache.cp_cache_layer_split.transfer')
        ns={'self':types.SimpleNamespace(token_to_kv_pool=pool,draft_token_to_kv_pool=draft),'DeepSeekV4TokenToKVPool':DeepSeekV4TokenToKVPool,'draft_kv_pool':draft,'kv_args':args}
        exec(compile(ast.Module(body=[branch],type_ignores=[]),'<actual_'+name+'_init_kv_manager_CP_branch>','exec'),ns)

    def test_both_public_caller_branches(self):
        for name in ('prefill','decode'):
            for with_draft in (False,True):
                p=self.pool([('c4',3),('c128',9)],[('swa',1)])
                draft=self.pool([('c4',0)],[('swa',0)]) if with_draft else None
                args=types.SimpleNamespace(kv_data_ptrs=[10,20]+([30] if draft else []),kv_item_lens=[64,128]+([64] if draft else []),state_types=[StateType.SWA]+([StateType.SWA] if draft else []),state_data_ptrs=[[40]]+([[50]] if draft else []))
                self.invoke(name,p,draft,args)
                self.assertTrue(args.cp_cache_layer_split)
                self.assertEqual(args.v4_transfer_metadata['kv'],[('c4',3),('c128',9)]+([('draft_c4',0)] if draft else []))
                self.assertEqual(args.v4_transfer_metadata['states'],[[('swa',1)]]+([[('draft_swa',0)]] if draft else []))
                self.assertEqual(match_transfer_entries(args.v4_transfer_metadata['kv'],list(reversed(args.v4_transfer_metadata['kv'])),args.kv_item_lens,list(reversed(args.kv_item_lens))),[(i,len(args.kv_item_lens)-1-i) for i in range(len(args.kv_item_lens))])

    def test_invalid_descriptor_rejected(self):
        for name in ('prefill','decode'):
            p=self.pool([('c4',3)],[('swa',1)])
            args=types.SimpleNamespace(kv_data_ptrs=[10,20],kv_item_lens=[64,128],state_types=[StateType.SWA],state_data_ptrs=[[40]])
            with self.assertRaisesRegex(RuntimeError,'KV transfer descriptors'):self.invoke(name,p,None,args)

if __name__=='__main__':unittest.main()
