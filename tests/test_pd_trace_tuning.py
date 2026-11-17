import ast
from collections import deque, namedtuple
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


T = module('hcu_sglang/models/qwen/qsa_tuning.py', 'qsa_tuning_test')
D = module('hcu_sglang/disagg/prefill_drain.py', 'prefill_drain_test')
P = module('hcu_sglang/disagg/prefill_pairing.py', 'prefill_pairing_test')
R = module('hcu_sglang/models/qwen/qsa_tp_rows.py', 'qsa_tp_rows_test')


class TraceTuningTests(unittest.TestCase):
    def test_tp_row_partition_preserves_effective_fallback(self):
        objects=[object()for _ in range(7)]
        calls=[]
        def original(*args):calls.append(args);return objects[-1]
        env=SimpleNamespace(envs=SimpleNamespace(SGLANG_HCU_QSA_PREFILL_TP_ROWS=SimpleNamespace(get=lambda:0)))
        with patch.dict(sys.modules,{'sglang.srt.environ':env}):
            self.assertIs(R.around_select_prefill_tokens(original,*objects),objects[-1])
        self.assertEqual(calls[-1],tuple(objects))
        for cuda,rows,threshold,tp,cp in ((True,8192,0,4,1),(False,8192,4096,4,1),(True,128,4096,4,1),(True,8192,4096,1,1),(True,8192,4096,4,2)):
            q=SimpleNamespace(is_cuda=cuda,shape=(rows,4,128))
            parallel=SimpleNamespace(attn_tp_size=tp,attn_cp_size=cp)
            args=[objects[0],q,*objects[2:]]
            self.assertIs(R.select_rows(original,*args,threshold=threshold,parallel=parallel),objects[-1])
            self.assertEqual(calls[-1],tuple(args))
        with self.assertRaises(ValueError):R.select_rows(original,*objects,threshold=-1,parallel=None)

    def test_disabled_budget_preserves_effective_original(self):
        sentinel = object()
        self.assertIs(T.row_chunk_size(0, 39296, 65536, 4,
                                      lambda *args: sentinel), sentinel)

    def test_budget_alignment_and_empty_inputs(self):
        original = lambda rows, keys, heads: max(rows, 1)
        for mib, expected in ((128,512),(256,1024),(512,2048),(1024,4096)):
            self.assertEqual(T.row_chunk_size(mib,39296,65536,4,original),expected)
        self.assertEqual(T.row_chunk_size(512,17,65536,4,original),17)
        self.assertEqual(T.row_chunk_size(512,0,65536,4,original),1)
        self.assertEqual(T.row_chunk_size(512,100,0,4,original),100)
        with self.assertRaises(ValueError):
            T.row_chunk_size(-1,39296,65536,4,original)

    def test_real_overlap_loop_drains_each_result_once(self):
        tree = ast.parse((ROOT/'hcu_sglang/disagg/prefill.py').read_text())
        owner = next(n for n in tree.body if isinstance(n,ast.ClassDef)
                     and n.name=='SchedulerDisaggregationPrefillMixin')
        function = next(n for n in owner.body if isinstance(n,ast.FunctionDef)
                        and n.name=='event_loop_overlap_disagg_prefill')
        function.decorator_list = []
        source = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
        ns = dict(deque=deque,torch=SimpleNamespace(empty=lambda *a,**k:None,int32='int32'))
        exec(compile(ast.fix_missing_locations(source),'actual-prefill-loop','exec'),ns)
        loop = ns[function.name]
        for mode, ready in [('off',True),('ready',False),('ready',True),('wait',False)]:
            with self.subTest(mode=mode,ready=ready):
                events=[]
                class Batch:
                    def __init__(self,index):self.index=index
                    def copy(self):return self
                class Scheduler:
                    gracefully_exit=False
                    _engine_paused=False
                    _deferred_input_requests=[]
                    enable_staging=False
                    chunked_req=None
                    running_batch=object()
                    last_batch=None
                    iterations=0
                    ngram_embedding_manager=SimpleNamespace(prepare_for_forward=lambda b,**kw:b)
                    def _is_continuous_input_polling_enabled(self):return False
                    def ingest_requests(self,**kwargs):
                        self.iterations+=1
                        if self.iterations==3:self.gracefully_exit=True
                        return False
                    def _admit_prefill_bootstrapped_reqs(self):return []
                    def get_next_disagg_prefill_batch_to_run(self,**kwargs):
                        events.append(('plan',self.iterations))
                        return SimpleNamespace(running_batch=self.running_batch,
                                               batch_to_run=Batch(self.iterations))
                    def run_batch(self,batch):
                        events.append(('run',batch.index))
                        return batch.index
                    def _apply_war_barrier(self):pass
                    def launch_batch_sample_if_needed(self,*args):pass
                    def process_batch_result(self,batch,result):
                        self.assert_result(batch,result)
                        events.append(('process',result))
                    def assert_result(self,batch,result):
                        if batch.index!=result:raise AssertionError('wrong result ownership')
                    def process_disagg_prefill_inflight_queue(self):pass
                    def maybe_supplemental_poll_final_chunks(self):pass
                    def on_idle(self):pass
                    def is_disagg_prefill_batch_result_ready(self,result,status):return ready
                fake=SimpleNamespace(drain_mode=lambda:mode,drain_previous_result=D.drain_previous_result)
                with patch.dict(sys.modules,{'hcu_sglang.disagg.prefill_drain':fake}):
                    scheduler=Scheduler();loop(scheduler)
                self.assertEqual([n for kind,n in events if kind=='process'],[1,2])
                self.assertEqual(len(scheduler.result_queue),0)
                if mode=='wait' or mode=='ready' and ready:
                    self.assertLess(events.index(('plan',2)),events.index(('process',1)))
                    self.assertLess(events.index(('process',1)),events.index(('run',2)))
                else:
                    self.assertLess(events.index(('run',2)),events.index(('process',1)))

    def test_ready_probe_is_collective_owned_by_scheduler(self):
        calls=[]
        scheduler=SimpleNamespace(last_batch=True,result_queue=deque([('batch','result')]),
            is_disagg_prefill_batch_result_ready=lambda r,s:calls.append(('probe',r,s)) or False)
        self.assertFalse(D.drain_previous_result(scheduler,'ready','status'))
        self.assertEqual(calls,[('probe','result','status')])
        self.assertEqual(list(scheduler.result_queue),[('batch','result')])

    def test_split_key_dispatch_preserves_effective_original(self):
        span=0
        calls=[]
        env=SimpleNamespace(envs=SimpleNamespace(
            SGLANG_HCU_QSA_PREFILL_KEY_SPAN=SimpleNamespace(get=lambda:span)))
        q=SimpleNamespace(is_cuda=True,device='gpu',dtype='bf16',ndim=3,shape=(32,4,128))
        k=SimpleNamespace(device='gpu',dtype='bf16',ndim=3,shape=(8192,1,128))
        bounds=SimpleNamespace(device='gpu',ndim=1,numel=lambda:32)
        def original(*args):
            calls.append(('original',args))
            return 'original-result'
        def prefill(*args,**kwargs):
            calls.append(('split',args,kwargs))
            return 'split-result'
        modules={'sglang.srt.environ':env,'torch':SimpleNamespace(bfloat16='bf16'),
                 'hcu_sglang.models.qwen.qsa_mqa_split_key':SimpleNamespace(prefill=prefill)}
        with patch.dict(sys.modules,modules):
            self.assertEqual(T.around_mqa_prefill(original,q,k,bounds,bounds,3),'original-result')
            self.assertIs(calls[-1][1][0],q)
            span=4096
            self.assertEqual(T.around_mqa_prefill(original,q,k,bounds,bounds,3),'split-result')
            self.assertEqual(calls[-1][2],{'key_span':4096})
            for field,value in [('is_cuda',False),('dtype','fp8'),('shape',(32,8,128))]:
                old=getattr(q,field);setattr(q,field,value)
                self.assertEqual(T.around_mqa_prefill(original,q,k,bounds,bounds),'original-result')
                setattr(q,field,old)
            k.shape=(1024,1,128)
            self.assertEqual(T.around_mqa_prefill(original,q,k,bounds,bounds),'original-result')
            span=17
            with self.assertRaises(ValueError):T.around_mqa_prefill(original,q,k,bounds,bounds)

    def test_pairing_release_uses_global_timeout_flag(self):
        output=namedtuple('Output','next_state input_estimation output_allow output_reason wait_forward_passes wait_seconds')
        base=output('state','mixed',False,'delay',0,0.)
        prev=SimpleNamespace(delayed_count=37,start_time=P.time.perf_counter()-.05)
        enabled=False
        env=SimpleNamespace(envs=SimpleNamespace(SGLANG_HCU_PREFILL_MIXED_TIMEOUT=SimpleNamespace(get=lambda:enabled)))
        class Buffer:
            flag=0
            reads=0
            def __getitem__(self,key):
                if key != (slice(None),0,5):raise AssertionError(key)
                self.reads+=1
                return SimpleNamespace(max=lambda:SimpleNamespace(item=lambda:self.flag))
        buffer=Buffer()
        owner=SimpleNamespace(_queue_trigger_enabled=True,_global_info_buffer=buffer)
        calls=[]
        def original(*args,**kwargs):calls.append((args,kwargs));return base
        with patch.dict(sys.modules,{'sglang.srt.environ':env}):
            buffer.flag=1
            self.assertIs(P.around_negotiate(original,owner,prev),base)
            self.assertEqual(buffer.reads,0)
            enabled=True;buffer.flag=0
            self.assertIs(P.around_negotiate(original,owner,prev),base)
            buffer.flag=1
            actual=P.around_negotiate(original,owner,prev)
            self.assertTrue(actual.output_allow)
            self.assertIsNone(actual.next_state)
            self.assertEqual(actual.output_reason,'wait_timeout')
            self.assertEqual(actual.wait_forward_passes,37)
            self.assertGreaterEqual(actual.wait_seconds,.05)
            base=base._replace(input_estimation='all')
            self.assertIs(P.around_negotiate(original,owner,prev),base)
            self.assertEqual(len(calls),4)
            owner._queue_trigger_enabled=False
            with self.assertRaises(ValueError):P.around_negotiate(original,owner,prev)
            self.assertEqual(len(calls),4)


if __name__=='__main__':
    unittest.main()
