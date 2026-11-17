import datetime
import json
import os
import pathlib
import types
from unittest.mock import patch

import unittest

if os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS') != '1':
    raise SystemExit('Set HCU_PLUGIN_RUN_GPU_TESTS=1 for the GPU regression')

import torch
import torch.distributed as dist
from sglang.srt.plugins import load_plugins

load_plugins()

from hcu_sglang.utils.detection import is_hcu
from sglang.srt.distributed import parallel_state as PS
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag as M

from sglang.test.test_utils import CustomTestCase

def run_gather_regression():
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
    root=pathlib.Path(os.environ['HCU_PLUGIN_TEST_OUTPUT_DIR'])
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/('rank-'+str(rank)+'.json')).exists()
    record={'rank':rank,'passed':False,'cases':[],'scope':'actual native gatherer public call through plugin load_plugins and real GroupCoordinator/RCCL/Graph; runtime publication metadata peers; forbidden native symmetric allocator sentinel','is_hcu':is_hcu()}
    def save():
        (root/('rank-'+str(rank)+'.json')).write_bytes((json.dumps(record,ensure_ascii=True,indent=2)+'\n').encode())
    save()
    assert record['is_hcu']
    group=PS.GroupCoordinator([list(range(8))],rank,'nccl',use_pynccl=True,use_mscclpp=False,use_custom_allreduce=False,use_torch_symm_mem_all_reduce=False,use_hpu_communicator=False,use_xpu_communicator=False,use_npu_communicator=False,group_name='hcu-logits-gather-test')
    parallel=types.SimpleNamespace(tp_rank=rank,tp_size=8,tp_group=group,nnodes=1)
    class NativeBuildForbidden(BaseException):pass
    def forbidden(**kwargs):
        raise NativeBuildForbidden('unsupported symmetric allocation must not be reached')
    try:
        with patch.object(PS,'get_parallel',return_value=parallel),patch.object(M,'get_parallel',return_value=parallel),patch.object(M,'create_state',side_effect=forbidden):
            for rows in [1,4]:
                gather=M.MultimemAllGatherer(128,skip_entry_sync=True)
                axis=torch.arange(rows*16160,dtype=torch.float32).reshape(rows,16160)%32
                x=(axis+rank*32).bfloat16().cuda()
                eager=gather(x)
                assert gather._state is None
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured=gather(x)
                snapshots=[]
                for state in range(3):
                    x.copy_((axis+rank*32+state).bfloat16())
                    eager=gather(x)
                    graph.replay()
                    torch.cuda.synchronize()
                    expected=torch.cat([(axis+r*32+state).bfloat16() for r in range(8)],dim=-1)
                    got=eager.cpu()
                    torch.testing.assert_close(got,expected,rtol=0,atol=0)
                    torch.testing.assert_close(captured.cpu(),expected,rtol=0,atol=0)
                    assert gather._state is None
                    snapshots.append(float(got[0,0]))
                    record['cases'].append({'rows':rows,'local_width':16160,'output_width':129280,'state':state,'eager_cpu_exact':True,'graph_cpu_exact':True,'symmetric_state':None})
                    save()
                assert len(set(snapshots))==3
            record['passed']=True
            save()
    finally:
        graph = captured = eager = gather = None
        import gc
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()
        group.destroy()
        dist.destroy_process_group()


class TestLogitsGatherGPU(CustomTestCase):
    def test_gather_and_graph_updates(self):
        run_gather_regression()


if __name__ == '__main__':
    unittest.main()
