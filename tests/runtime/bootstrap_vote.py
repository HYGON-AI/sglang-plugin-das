# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import datetime
import importlib.util
import json
import os
import pathlib
import socket
import sys
import types


ORIGINAL_METHOD = 'def _get_consensus_bootstrap_resource_ready(\n        self, polls: List[KVPoll]\n    ) -> List[bool]:\n        """Agree on bootstrap resource admission across attention TP/CP ranks.\n\n        PD hidden rows are released asynchronously after the local hidden-state\n        transfer completes.  Consequently, attention CP ranks can momentarily\n        observe different free-row counts.  Reserving directly from those local\n        counts lets one rank move a request to the waiting queue while another\n        leaves it in the bootstrap queue, and the next variable-length Gloo\n        collective then fails.  Probe resources without side effects and admit\n        only the common FIFO prefix before performing any allocation.\n        """\n        ready = [True] * len(self.queue)\n        metadata_credits = self.req_to_metadata_buffer_idx_allocator.available_size()\n        pool = getattr(getattr(self, "metadata_buffers", None), "pd_hidden_pool", None)\n        hidden_row_credits = pool.available_size() if pool is not None else 0\n\n        resource_blocked = False\n        for i, (req, poll) in enumerate(zip(self.queue, polls)):\n            if (\n                resource_blocked\n                or poll != KVPoll.WaitingForInput\n                or not req.pending_bootstrap\n                or should_force_retry(req)\n            ):\n                if resource_blocked and poll == KVPoll.WaitingForInput:\n                    ready[i] = False\n                continue\n\n            costs, error = self._probe_bootstrap_ready(\n                req, metadata_credits, hidden_row_credits\n            )\n            # Invalid request metadata is deterministic and must reach\n            # finalize_bootstrap on every rank so the request is failed cleanly.\n            if error is not None:\n                continue\n            if costs is None:\n                ready[i] = False\n                resource_blocked = True\n                continue\n\n            metadata_cost, hidden_cost = costs\n            metadata_credits -= metadata_cost\n            hidden_row_credits -= hidden_cost\n\n        ready_tensor = torch.tensor(ready, dtype=torch.uint8, device="cpu")\n        torch.distributed.all_reduce(\n            ready_tensor,\n            op=torch.distributed.ReduceOp.MIN,\n            group=self.scheduler.attn_tp_cpu_group,\n        )\n        torch.distributed.all_reduce(\n            ready_tensor,\n            op=torch.distributed.ReduceOp.MIN,\n            group=self.scheduler.attn_cp_cpu_group,\n        )\n\n        # Only the rank hosting the bootstrap server hears decode\'s\n        # decode_prefix_len; the other ranks would resolve 0 and derive a\n        # diverging radix cap for the same request. Agree on the value with the\n        # same collectives as the readiness vote above, and backfill the manager\n        # dict so every rank\'s finalize_bootstrap resolves -- and pops -- the\n        # same value. A genuine 0 stays absent and keeps the default path.\n        prefix_tensor = torch.tensor(\n            [\n                int(\n                    self.kv_manager.req_to_decode_prefix_len.get(req.bootstrap_room, 0)\n                    or 0\n                )\n                for req in self.queue\n            ],\n            dtype=torch.int64,\n            device="cpu",\n        )\n        torch.distributed.all_reduce(\n            prefix_tensor,\n            op=torch.distributed.ReduceOp.MAX,\n            group=self.scheduler.attn_tp_cpu_group,\n        )\n        torch.distributed.all_reduce(\n            prefix_tensor,\n            op=torch.distributed.ReduceOp.MAX,\n            group=self.scheduler.attn_cp_cpu_group,\n        )\n        for req, value in zip(self.queue, prefix_tensor.tolist()):\n            if value > 0:\n                self.kv_manager.req_to_decode_prefix_len[req.bootstrap_room] = int(\n                    value\n                )\n\n        return [bool(value) for value in ready_tensor.tolist()]'

def method(source, hcu):
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == '_get_consensus_bootstrap_resource_ready')
    fn.returns = None
    for arg in fn.args.args+fn.args.kwonlyargs:
        arg.annotation = None
    import torch
    ns = dict(torch=torch,KVPoll=types.SimpleNamespace(WaitingForInput=1),should_force_retry=lambda req:req.retry,_is_hcu=hcu)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'bootstrap_vote_method','exec'),ns)
    return ns[fn.name]


def peer(rank,n,case,tp,cp):
    count = [0,1,4,6,3,5,6,4][case]
    queue = [types.SimpleNamespace(bootstrap_room=100+i,pending_bootstrap=(i%3!=1 or case%2==0),retry=(i==2 and case==5),invalid=(i==1 and case==6),hidden_cost=1+i%3,metadata_cost=1) for i in range(count)]
    values = {req.bootstrap_room:3*(i+1) if rank==0 else (-2 if case==7 else 0) for i,req in enumerate(queue)}
    meta = (rank+case)%5
    hidden = (rank*3+case)%13
    self = types.SimpleNamespace(queue=queue,scheduler=types.SimpleNamespace(attn_tp_cpu_group=tp,attn_cp_cpu_group=cp),req_to_metadata_buffer_idx_allocator=types.SimpleNamespace(available_size=lambda:meta),metadata_buffers=types.SimpleNamespace(pd_hidden_pool=types.SimpleNamespace(available_size=lambda:hidden)),kv_manager=types.SimpleNamespace(req_to_decode_prefix_len=values))
    def probe(req,metadata_credits,hidden_credits):
        if req.invalid:
            return None,'invalid metadata'
        if metadata_credits<req.metadata_cost or hidden_credits<req.hidden_cost:
            return None,None
        return (req.metadata_cost,req.hidden_cost),None
    self._probe_bootstrap_ready = probe
    polls = [0 if i%4==3 else 1 for i in range(count)]
    return self,polls


def child(rank,port,out,repo):
    import torch
    import torch.distributed as dist
    root = pathlib.Path(out)
    repo = pathlib.Path(repo)
    spec = importlib.util.spec_from_file_location('hcu_sglang.disagg.bootstrap_vote',repo/'hcu_sglang/disagg/bootstrap_vote.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for name in ['hcu_sglang','hcu_sglang.disagg']:
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    sys.modules['hcu_sglang.disagg.bootstrap_vote'] = helper
    dist.init_process_group('gloo',init_method='tcp://127.0.0.1:'+str(port),rank=rank,world_size=4,timeout=datetime.timedelta(seconds=60))
    singleton = None
    for index in range(4):
        group = dist.new_group([index],backend='gloo')
        if rank==index:
            singleton=group
    tp = cp = None
    for indices in [[0,1],[2,3],[0,2],[1,3]]:
        group = dist.new_group(indices,backend='gloo')
        if rank in indices:
            if indices in [[0,1],[2,3]]:
                tp=group
            else:
                cp=group
    original_reduce = dist.all_reduce
    calls = []
    def observe(tensor,op=dist.ReduceOp.SUM,group=None,**kw):
        calls.append(dict(size=dist.get_world_size(group),dtype=str(tensor.dtype),op=str(op)))
        return original_reduce(tensor,op=op,group=group,**kw)
    dist.all_reduce = observe
    rows = []
    try:
        for label,groups,hcu,expected in [('TP1_CP4',(singleton,dist.group.WORLD),True,2),('TP2_CP2',(tp,cp),True,4),('TP1_CP1',(singleton,singleton),True,4),('nonHCU',(singleton,dist.group.WORLD),False,4)]:
            before = method(ORIGINAL_METHOD,hcu)
            after = method((repo/'hcu_sglang/disagg/prefill.py').read_bytes(),hcu)
            for case in range(8):
                left,polls = peer(rank,4,case,*groups)
                right,_ = peer(rank,4,case,*groups)
                calls.clear()
                wanted = before(left,polls)
                old_calls = copy.deepcopy(calls)
                calls.clear()
                actual = after(right,polls)
                new_calls = copy.deepcopy(calls)
                assert actual == wanted
                assert right.kv_manager.req_to_decode_prefix_len == left.kv_manager.req_to_decode_prefix_len
                assert len(old_calls)==4 and len(new_calls)==expected,(label,case,old_calls,new_calls)
                rows.append(dict(case=label+'_'+str(case),original_calls=old_calls,candidate_calls=new_calls,ready=actual,prefix=right.kv_manager.req_to_decode_prefix_len))
        dist.all_reduce = original_reduce
        dist.barrier()
        assert not torch.cuda.is_initialized()
        (root/('rank-'+str(rank)+'.json')).write_text(json.dumps(dict(passed=True,pid=os.getpid(),rank=rank,cases=rows,cuda_initialized=False,backend='gloo'),indent=2))
    finally:
        dist.all_reduce = original_reduce
        dist.destroy_process_group()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    root = pathlib.Path(sys.argv[1]).resolve()
    root.mkdir()
    repo = pathlib.Path(__file__).resolve().parents[2]
    with socket.socket() as server:
        server.bind(('127.0.0.1',0))
        port = server.getsockname()[1]
    mp.spawn(child,args=(port,str(root),str(repo)),nprocs=4,join=True)
    ranks = [json.loads((root/('rank-'+str(i)+'.json')).read_bytes()) for i in range(4)]
    result = dict(passed=True,cases_per_rank=32,ranks=ranks,scope='Actual before/after bootstrap resource method AST with real four-process CPU Gloo; framework queue/probe/KVPoll and HCU selection are explicit CPU peers. MIN/MAX order and prefix backfill preserved; multi-rank reductions never skipped. No device API/model qualification.')
    (root/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=True,cases_per_rank=32,ranks=4)))
