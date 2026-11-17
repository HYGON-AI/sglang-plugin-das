import ast
import json
import pathlib
import sys
import types

import torch

root=pathlib.Path(__file__).resolve().parent
source=(root.parents[1]/'hcu_sglang/layers/moe/mega_moe.py').read_bytes()
names={'run_mega_routed_experts','build_hcu_w4a8_mega_moe_experts_weights','_run_deep_gemm_hcu_w4a8_mega_moe'}
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes}==names
events=[]
deep=types.ModuleType('deepgemm');deep.__path__=[]
mega=types.ModuleType('deepgemm.mega');deep.mega=mega
deep.pack_w4a8_moe_hipc_weight=lambda value:value.clone()
def tile(value):
    e,n=value.shape[:2];tail=value.shape[2:]
    return torch.stack([value[:,:n//2].reshape(e,n//512,256,*tail),value[:,n//2:].reshape(e,n//512,256,*tail)],dim=2).reshape_as(value).contiguous()
mega.tile256_gate_up=tile
mega.mega_moe_preprocess=lambda *args,**kwargs:events.append(('preprocess',args,kwargs))
def native(output,*args,**kwargs):
    events.append(('native',args,kwargs));output.fill_(3)
mega.fp8_mega_moe=native
sys.modules['deepgemm']=deep;sys.modules['deepgemm.mega']=mega
expert=types.ModuleType('sglang.srt.eplb.expert_location');expert.get_global_expert_location_metadata=lambda:None
graph=types.ModuleType('sglang.srt.model_executor.cuda_graph_config');graph.cuda_graph_fully_disabled=lambda:True
runtime=types.ModuleType('sglang.srt.runtime_context');runtime.get_parallel=lambda:types.SimpleNamespace(moe_ep_group=types.SimpleNamespace(device_group='CPU_peer_EP8'))
for name in ['sglang','sglang.srt','sglang.srt.eplb','sglang.srt.model_executor']:
    module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
sys.modules[expert.__name__]=expert;sys.modules[graph.__name__]=graph;sys.modules[runtime.__name__]=runtime
active={'runtime':'deep_gemm'}
env=types.SimpleNamespace(SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=types.SimpleNamespace(get=lambda:4096))
ns={'torch':torch,'_is_hcu':True,'_HCU_MEGA_MOE_RUNTIME_DEEP_GEMM':'deep_gemm','_HCU_MEGA_MOE_RUNTIME_MEGAMOE':'megamoe','envs':env,'get_hcu_mega_moe_runtime':lambda:active['runtime'],'get_exec':lambda:types.SimpleNamespace(moe=types.SimpleNamespace(enable_eplb=False)),'get_dp_global_num_tokens':lambda:None,'is_dsa_enable_prefill_cp':lambda:True,'get_is_capture_mode':lambda:False,'_mega_moe_mma_type':lambda _: 'w4a8','_get_mega_moe_symm_buffer':lambda *args,**kwargs:types.SimpleNamespace(runtime=kwargs['runtime'],quant_mode=kwargs['quant_mode'])}
ns['_run_standalone_hcu_w4a8_mega_moe']=lambda x,*args,**kwargs:torch.full_like(x,5)
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
exec(compile(ast.fix_missing_locations(module),str(root/'mega_moe.py'),'exec'),ns)
cases=[]
for choice,expected in [('deep_gemm',6),('megamoe',10)]:
    active['runtime']=choice
    obj=types.SimpleNamespace(num_experts=384,_mega_moe_hcu_w4a8_weights=True,mega_l1_weights=('w1','s1'),mega_l2_weights=('w2','s2'))
    x=torch.zeros(4,5120,dtype=torch.bfloat16)
    y=ns['run_mega_routed_experts'](obj,x,torch.zeros(4,6,dtype=torch.int64),torch.zeros(4,6),hidden_size=5120,intermediate_size=2304,top_k=6,num_tokens=4,routed_scaling_factor=2)
    assert torch.equal(y,torch.full_like(x,expected))
    cases.append(choice+'_public_dispatch_and_scale')
active['runtime']='deep_gemm'
for count in [0,2]:
    events.clear();x=torch.zeros(count,5120,dtype=torch.bfloat16)
    ns['_run_deep_gemm_hcu_w4a8_mega_moe'](x,torch.zeros(count,6,dtype=torch.int64),torch.zeros(count,6),obj,object())
    assert [e[0] for e in events]==(['preprocess','native'] if count else ['native'])
    assert events[-1][2]['mode']=='w4a8'
    cases.append('native_consumer_empty_'+str(count))
for choice in ['deep_gemm','megamoe']:
    active['runtime']=choice
    w13=torch.arange(4608,dtype=torch.float32)[None,:,None].expand(1,4608,2560).to(torch.int8).contiguous()
    w2=torch.zeros((1,5120,1152),dtype=torch.int8)
    s13=torch.arange(4608,dtype=torch.float32)[None,:];s2=torch.ones((1,5120))
    obj=types.SimpleNamespace(w13_weight=torch.nn.Parameter(w13.clone(),False),w2_weight=torch.nn.Parameter(w2,False),w13_weight_scale=torch.nn.Parameter(s13.clone(),False),w2_weight_scale=torch.nn.Parameter(s2,False))
    ns['build_hcu_w4a8_mega_moe_experts_weights'](obj)
    expected_w=tile(w13) if choice=='deep_gemm' else w13
    expected_s=tile(s13) if choice=='deep_gemm' else s13
    assert torch.equal(obj.w13_weight,expected_w) and torch.equal(obj.w13_weight_scale,expected_s*16)
    assert obj._mega_moe_hcu_runtime==choice and obj._mega_moe_weights_built
    cases.append(choice+'_postload_weight_and_scale_layout')
active['runtime']='deep_gemm'
for shape in [(1,4096,2560),(65,4608,2560)]:
    obj=types.SimpleNamespace(w13_weight=torch.empty(shape,device='meta',dtype=torch.int8),w2_weight=torch.empty((shape[0],5120,1152),device='meta',dtype=torch.int8))
    try:ns['build_hcu_w4a8_mega_moe_experts_weights'](obj)
    except ValueError:cases.append('unsupported_shape_'+str(shape))
    else:raise AssertionError('unsupported shape accepted')
result={'passed':True,'cases':cases,'count':len(cases),'scope':'actual plugin constructor/native-caller/public-dispatch code with CPU Torch and explicit platform/collective/native-pack peers; not hardware numerics'}
print(json.dumps(result))
