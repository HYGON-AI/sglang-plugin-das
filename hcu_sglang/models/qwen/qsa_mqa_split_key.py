import math
import logging
import torch
import tilelang
from tilelang import language as T

_route_logged = False


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def split_key_kernel(key_span: int=4096, block_q: int=32, block_n: int=64):
    rows=T.dynamic('rows');keys=T.dynamic('keys')

    @T.prim_func
    def qsa_mqa_prefill_split_key(
        Q:T.Tensor([rows*4,128],T.bfloat16),
        K:T.Tensor([keys,128],T.bfloat16),
        Logits:T.Tensor([rows,keys],T.float32),
        Starts:T.Tensor([rows],T.int32),
        Ends:T.Tensor([rows],T.int32),
        InvScale:T.float32,
    ):
        with T.Kernel(T.ceildiv(rows,block_q),T.ceildiv(keys,key_span),threads=256)as(bx,bk):
            q_shared=T.alloc_shared([block_q*4,128],T.bfloat16)
            k_shared=T.alloc_shared([block_n,128],T.bfloat16)
            scores=T.alloc_fragment([block_n,block_q*4],T.float32)
            scores_3d=T.reshape(scores,(block_n,block_q,4))
            reduced=T.alloc_fragment([block_n,block_q],T.float32)
            starts_f=T.alloc_fragment([block_q],T.int32)
            ends_f=T.alloc_fragment([block_q],T.int32)
            start_min_f=T.alloc_fragment([1],T.int32)
            end_max_f=T.alloc_fragment([1],T.int32)
            row_base=bx*block_q
            for qi in T.Parallel(block_q):
                row=T.min(row_base+qi,rows-1)
                starts_f[qi]=Starts[row]
                ends_f[qi]=Ends[row]
            T.reduce_min(starts_f,start_min_f,dim=0,clear=True)
            T.reduce_max(ends_f,end_max_f,dim=0,clear=True)
            start_min=T.min(start_min_f[0],keys)
            end_max=T.min(end_max_f[0],keys)
            part_start=T.min(start_min+bk*key_span,end_max)
            part_end=T.min(part_start+key_span,end_max)
            T.copy(Q[row_base*4,0],q_shared)
            for ni in T.Pipelined(T.ceildiv(part_end-part_start,block_n),num_stages=1):
                T.copy(K[part_start+ni*block_n,0],k_shared)
                T.gemm(k_shared,q_shared,scores,transpose_B=True,clear_accum=True,policy=T.GemmWarpPolicy.FullCol)
                for n,qi,head in T.Parallel(block_n,block_q,4):
                    scores_3d[n,qi,head]=T.max(scores_3d[n,qi,head],0.0)
                T.reduce_sum(scores_3d,reduced,dim=-1,clear=True)
                for qi,n in T.Parallel(block_q,block_n):
                    Logits[row_base+qi,part_start+ni*block_n+n]=reduced[n,qi]*InvScale
    return qsa_mqa_prefill_split_key


def prefill(q,k,starts,ends,score_scale=None,*,key_span=4096):
    global _route_logged
    from boltops.mqa_logits.tilelang.kernels.common.mqa import _tilelang_qsa_mqa_mask_kernel
    assert q.dtype==k.dtype==torch.bfloat16 and q.shape[1:]==(4,128) and k.shape[1:]==(1,128)
    assert key_span>0 and key_span%64==0
    logits=torch.empty((q.shape[0],k.shape[0]),device=q.device,dtype=torch.float32)
    if not q.shape[0]or not k.shape[0]:
        return logits.fill_(-float('inf'))
    starts=starts.to(torch.int32).contiguous()
    ends=ends.to(torch.int32).contiguous()
    block_q=16 if k.shape[0]>=32768 and q.shape[0]>=128 else 32
    split_key_kernel(key_span,block_q)(q.contiguous().view(-1,128),k[:,0].contiguous(),logits,
                               starts,ends,
                               1./(score_scale or math.sqrt(128)))
    _tilelang_qsa_mqa_mask_kernel()(logits,starts,ends)
    if not _route_logged:
        logging.getLogger(__name__).info(
            'Qwen4 QSA key-parallel MQA prefill active: Q=%s, K=%s, span=%s',
            tuple(q.shape),tuple(k.shape),key_span)
        _route_logged=True
    return logits
