import ast
import dataclasses
import hashlib
import json
import pathlib
import sys
import types


def load_functions(path, namespace):
    tree=ast.parse(path.read_bytes())
    names={'_build_sparse_prefill_chunk_cache','_as_int_list','_pad_tensor_to_size','interleave_rows_per_request'}
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)


def equal(a,b,torch):
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape and a.stride()==b.stride()
        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
    elif dataclasses.is_dataclass(a) and hasattr(a,'c0_combined_indices'):
        torch.testing.assert_close(a.c0_combined_lens,b.c0_combined_lens,rtol=0,atol=0)
        valid=torch.arange(a.c0_combined_indices.shape[1],device=a.c0_combined_indices.device)[None,:]<a.c0_combined_lens[:,None]
        torch.testing.assert_close(a.c0_combined_indices[valid],b.c0_combined_indices[valid],rtol=0,atol=0)
        for k,value in vars(a).items():
            if k!='c0_combined_indices':equal(value,getattr(b,k),torch)
    elif isinstance(a,(dict,types.SimpleNamespace)) or dataclasses.is_dataclass(a):
        a=a if isinstance(a,dict) else vars(a);b=b if isinstance(b,dict) else vars(b)
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k],torch)
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y,torch)
    else:assert a==b,(a,b)


def run(root,mode):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    gpu=mode=='hcu'
    if gpu:
        from hcu_sglang.utils.detection import is_hcu
        assert is_hcu() and torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx936'
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.sparse_prefill_utils import build_cp_sparse_query_metadata,SparsePrefillChunkCache
        source=root/'inputs/hcu_sglang/models/deepseek/current_v4/hcu_backend/sparse_prefill_utils.py'
        assert pathlib.Path(sys.modules[build_cp_sparse_query_metadata.__module__].__file__).read_bytes()==source.read_bytes()
        producer=build_cp_sparse_query_metadata
        consumer=SparsePrefillChunkCache
    else:
        producer=None;consumer=types.SimpleNamespace(build=lambda **kw:types.SimpleNamespace(**kw))
    parallel=types.SimpleNamespace(attn_cp_size=2,attn_cp_rank=0)
    device='cuda' if gpu else 'cpu'
    namespaces=[]
    for filename in ['before.py','candidate.py']:
        ns=dict(torch=torch,_is_hcu=True,get_parallel=lambda:parallel,is_cp_active=lambda f:f.cp,dsa_use_prefill_cp=lambda f:f.cp,SWA_WINDOW=128,SparsePrefillChunkCache=consumer)
        load_functions(root/'inputs/hcu_sglang/models/deepseek/current_v4/hcu_backend/interleave.py',ns)
        load_functions(root/filename,ns)
        namespaces.append(ns)
    def cpu_producer(extend_seq_lens,extend_start_loc,query_positions,cp_size,cp_rank):
        lengths=namespaces[0]['interleave_rows_per_request'](extend_seq_lens.tolist(),cp_rank,cp_size)
        physical=list(lengths);physical[-1]+=query_positions.numel()-sum(lengths)
        pos=query_positions.clone();pos[sum(lengths):]=-1
        return torch.tensor(physical,dtype=torch.int32),torch.tensor(lengths,dtype=torch.int32),pos
    for ns in namespaces:ns['build_cp_sparse_query_metadata']=producer if gpu else cpu_producer
    class Observe(TorchDispatchMode):
        def __init__(self):super().__init__();self.scalar=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if func==torch.ops.aten._local_scalar_dense.default:self.scalar+=1
            return func(*args,**(kwargs or {}))
    def fixture(lengths,size,rank,padding):
        counts=namespaces[0]['interleave_rows_per_request'](lengths,rank,size)
        capacity=max(sum(namespaces[0]['interleave_rows_per_request'](lengths,r,size)) for r in range(size))+padding
        capacity=max(capacity,1)
        prefixes=[11+i for i in range(len(lengths))]
        seq=[p+n for p,n in zip(prefixes,lengths)]
        positions=[];start=0
        for prefix,n in zip(prefixes,lengths):
            positions += [prefix+j for j in range(n) if (start+j)%size==rank]
            start+=n
        positions += [0]*(capacity-len(positions))
        lens=torch.tensor(lengths,dtype=torch.int32,device=device)
        starts=torch.cumsum(lens,0,dtype=torch.int32)-lens
        pos=torch.tensor(positions,dtype=torch.int32,device=device)
        width=((max(seq)+255)//256)*256
        req=torch.arange(len(seq)*width,dtype=torch.int32,device=device).reshape(len(seq),width)
        mapping=torch.arange(req.numel()+256,dtype=torch.int64,device=device)
        fb=types.SimpleNamespace(cp=True,seq_lens_cpu=torch.tensor(seq,dtype=torch.int32),extend_seq_lens=lens,extend_seq_lens_cpu=list(lengths),extend_start_loc=starts,seq_lens=torch.tensor(seq,dtype=torch.int32,device=device),req_pool_indices=torch.arange(len(seq),dtype=torch.int32,device=device))
        obj=types.SimpleNamespace(forward_metadata=types.SimpleNamespace(late_layer_tail=None),req_to_token=req,token_to_kv_pool=types.SimpleNamespace(full_to_swa_index_mapping=mapping,get_swa_key_page_size=lambda:256))
        core=types.SimpleNamespace(request_window_layout=None,seq_lens_casual=pos+1)
        return obj,fb,core,capacity,sum(counts)
    cases=[]
    shapes=[[1],[0,1,0,5],[7,0,3],[128,129,17],[32768]]
    for size in [2,4,8]:
        for rank in range(size):
            for lengths in shapes:
                for padding in ([0,3] if not gpu else [3]):
                    parallel.attn_cp_size=size;parallel.attn_cp_rank=rank
                    obj,fb,core,capacity,real=fixture(lengths,size,rank,padding)
                    outputs=[];reads=[]
                    for ns in namespaces:
                        with Observe() as observed:outputs.append(ns['_build_sparse_prefill_chunk_cache'](obj,fb,core,num_qo_tokens=capacity))
                        reads.append(observed.scalar)
                    assert reads==[1,0],reads
                    equal(*outputs,torch)
                    cases.append(dict(size=size,rank=rank,lengths=lengths,padding=capacity-real,scalar_reads=reads))
    fallback=[]
    for hcu,cp in ([(True,False)] if gpu else [(False,True),(False,False),(True,False)]):
        parallel.attn_cp_size=2;parallel.attn_cp_rank=0
        obj,fb,core,capacity,_=fixture([7,5],2,0,3);fb.cp=cp
        if not cp:
            capacity=sum(fb.extend_seq_lens_cpu);core.seq_lens_casual=torch.arange(capacity,dtype=torch.int32,device=device)+1
        outputs=[];reads=[]
        for ns in namespaces:
            ns['_is_hcu']=hcu
            with Observe() as observed:outputs.append(ns['_build_sparse_prefill_chunk_cache'](obj,fb,core,num_qo_tokens=capacity))
            reads.append(observed.scalar)
        assert reads[0]==reads[1]
        equal(*outputs,torch);fallback.append(dict(hcu=hcu,cp=cp,scalar_reads=reads))
    for ns in namespaces:ns['_is_hcu']=True
    parallel.attn_cp_size=8;parallel.attn_cp_rank=0
    obj,fb,core,_,_=fixture([32768],8,0,3)
    called=[];ns=namespaces[1];original_producer=ns['build_cp_sparse_query_metadata']
    ns['build_cp_sparse_query_metadata']=lambda **kw:called.append(True)
    try:
        ns['_build_sparse_prefill_chunk_cache'](obj,fb,core,num_qo_tokens=2)
        raise RuntimeError('undersized geometry accepted')
    except AssertionError:assert not called
    finally:ns['build_cp_sparse_query_metadata']=original_producer
    graphs=[]
    if gpu:
        obj,fb,core,capacity,_=fixture([128,129,17],8,0,3)
        fn=namespaces[1]['_build_sparse_prefill_chunk_cache']
        for _ in range(3):fn(obj,fb,core,num_qo_tokens=capacity)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=fn(obj,fb,core,num_qo_tokens=capacity)
        for i in range(3):
            begin=0
            for n in namespaces[0]['interleave_rows_per_request'](fb.extend_seq_lens_cpu,0,8):
                core.seq_lens_casual[begin:begin+n].copy_(core.seq_lens_casual[begin:begin+n].flip(0))
                begin+=n
            graph.replay()
            reference=namespaces[0]['_build_sparse_prefill_chunk_cache'](obj,fb,core,num_qo_tokens=capacity)
            equal(reference,captured,torch);graphs.append(dict(replay=i+1,passed=True))
        torch.cuda.synchronize()
    else:assert not torch.cuda.is_initialized()
    return dict(passed=True,mode=mode,cases=cases,fallback=fallback,undersized_rejected_before_native=True,graph=graphs,source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['before.py','candidate.py']},scope='Complete original/candidate cache-builder methods. CPU uses explicit topology/HCU-selector/native-producer and consumer peers; HCU uses actual producer and complete native cache consumer, zero-tolerance defined indices/lengths and all other fields, ignored index capacity excluded by actual topk_length consumer, scalar dispatch observation and Graph updates. No model/collective grant.')


if __name__=='__main__':
    root=pathlib.Path(sys.argv[1]);mode=sys.argv[2];out=pathlib.Path(sys.argv[3]);assert not out.exists()
    result=run(root,mode);out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=True,mode=mode,cases=len(result['cases']),fallback=len(result['fallback']),graphs=len(result['graph']))))
