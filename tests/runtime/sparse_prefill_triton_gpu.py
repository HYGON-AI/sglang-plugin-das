import importlib.util
import json
import pathlib
import os
if os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS') != '1':
    raise SystemExit('Set HCU_PLUGIN_RUN_GPU_TESTS=1 for the GPU regression')
import torch

root=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ifb_sparse_triton',root.parents[1]/'hcu_sglang/layers/attention/dsv41_sparse_prefill_triton.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.cuda.set_device(0);torch.manual_seed(20261115)
cases=[]
def reference(q,kv,indices,scale,sink,lengths):
    ids=indices[:,0].long();mask=(ids>=0)&(ids<kv.shape[0])
    if lengths is not None:mask &= torch.arange(ids.shape[1],device=q.device)[None,:]<lengths[:,None]
    keys=kv[:,0][ids.clamp(0,kv.shape[0]-1)].float().masked_fill(~mask[:,:,None],0)
    logits=torch.einsum('qhd,qkd->qhk',q.float(),keys)*scale
    logits=logits.masked_fill(~mask[:,None,:],float('-inf'))
    if sink is not None:
        logits=torch.cat([logits,sink[None,:,None].expand(q.shape[0],-1,-1)],dim=-1)
        values=torch.cat([keys,torch.zeros(q.shape[0],1,512,device=q.device)],dim=1)
    else:values=keys
    probabilities=torch.softmax(logits,dim=-1)
    probabilities=torch.where(torch.isneginf(logits).all(-1,keepdim=True),0,probabilities)
    return torch.einsum('qhk,qkd->qhd',probabilities,values).to(q.dtype)
for heads in [8,64]:
    for name in ['finite','ragged','masked_nan','valid_nan','all_invalid','no_sink']:
        q=torch.randn(17,heads,512,device='cuda',dtype=torch.bfloat16)*0.25
        kv=torch.randn(256,1,512,device='cuda',dtype=torch.bfloat16)*0.25
        ids=torch.randint(1,256,(17,1,65),device='cuda',dtype=torch.int32)
        lengths=torch.arange(17,device='cuda',dtype=torch.int32)+32 if name=='ragged' else torch.full((17,),65,device='cuda',dtype=torch.int32)
        sink=None if name=='no_sink' else torch.randn(heads,device='cuda')*0.1
        if name=='masked_nan':
            kv[0]=float('nan');ids[:,:,0]=-1
        if name=='valid_nan':
            kv[0]=float('nan');ids[:,:,0]=0
        if name=='all_invalid':ids.fill_(-1)
        got=module.sparse_prefill(q,kv,ids,512**-0.5,512,sink,lengths)[0]
        expected=reference(q,kv,ids,512**-0.5,sink,lengths)
        torch.testing.assert_close(got,expected,rtol=0.02,atol=0.004,equal_nan=True)
        torch.cuda.synchronize()
        cases.append({'case':name,'heads':heads,'passed':True})
q=torch.randn(32,64,512,device='cuda',dtype=torch.bfloat16)*0.25
kv=torch.randn(512,1,512,device='cuda',dtype=torch.bfloat16)*0.25
ids=torch.randint(0,512,(32,1,128),device='cuda',dtype=torch.int32)
lengths=torch.full((32,),128,device='cuda',dtype=torch.int32);sink=torch.zeros(64,device='cuda')
module.sparse_prefill(q,kv,ids,512**-0.5,512,sink,lengths);torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):captured=module.sparse_prefill(q,kv,ids,512**-0.5,512,sink,lengths)[0]
for state in range(3):
    kv.mul_(1.25)
    expected=reference(q,kv,ids,512**-0.5,sink,lengths)
    eager=module.sparse_prefill(q,kv,ids,512**-0.5,512,sink,lengths)[0]
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(captured,expected,rtol=0.02,atol=0.004)
    torch.testing.assert_close(captured,eager,rtol=0,atol=0)
    cases.append({'case':'graph_kv_update','state':state,'passed':True})
result={'passed':True,'count':len(cases),'cases':cases,'scope':'isolated sparse-prefill BF16 kernel, poison/mask and graph KV updates; no full-model accuracy claim'}
print(json.dumps(result),flush=True)
