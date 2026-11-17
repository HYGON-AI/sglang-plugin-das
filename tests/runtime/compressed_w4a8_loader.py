import ast,functools,importlib.util,json,pathlib,types
import torch
torch.set_num_threads(1)
repo=pathlib.Path(__file__).resolve().parents[2]
source=(repo/'hcu_sglang/layers/quantization/slimquant_w4a8_marlin.py').read_text()
tree=ast.parse(source)
method=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HCUW4A8Int8DeepEPMoEMethod')
factory=next(n for n in method.body if isinstance(n,ast.FunctionDef) and n.name=='create_weights')
native=(pathlib.Path(next(iter(importlib.util.find_spec('sglang').submodule_search_locations)))/'srt/layers/moe/fused_moe_triton/layer.py').read_text()
ntree=ast.parse(native)
names={'_weight_loader_impl','_load_w13','_load_w2','_load_per_channel_weight_scale','_load_model_weight_or_group_weight_scale','_load_single_value'}
functions=[n for n in ast.walk(ntree) if isinstance(n,ast.FunctionDef) and n.name in names]
class Dummy:pass
class Scale:
 CHANNEL=types.SimpleNamespace(value='channel');GROUP=types.SimpleNamespace(value='group');BLOCK=types.SimpleNamespace(value='block');TENSOR=types.SimpleNamespace(value='tensor')
class ScaleEnum:
 def __iter__(self):return iter([Scale.CHANNEL,Scale.GROUP,Scale.BLOCK,Scale.TENSOR])
g={'torch':torch,'functools':functools,'set_weight_attrs':lambda p,a:[setattr(p,k,v) for k,v in a.items()],'FusedMoeWeightScaleSupported':ScaleEnum(),'UnquantizedFusedMoEMethod':Dummy,'Fp8MoEMethod':Dummy,'ModelOptNvFp4FusedMoEMethod':Dummy,'CompressedTensorsMxInt4MoE':Dummy,'_is_hip':False,'_is_cpu':True,'get_bool_env_var':lambda *a:False,'_maybe_copy_weight_view_before_h2d':lambda w:w}
for key in ['CHANNEL','GROUP','BLOCK','TENSOR']:setattr(g['FusedMoeWeightScaleSupported'],key,getattr(Scale,key))
body=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='NativeLoader',bases=[],keywords=[],body=functions,decorator_list=[])],type_ignores=[])
exec(compile(ast.fix_missing_locations(body),'<actual-native-loader-ast>','exec'),g)
Loader=g['NativeLoader']
def get_method(fixed):
 f=ast.parse(ast.unparse(factory)).body[0]
 wrapper=next(n for n in method.body if isinstance(n,ast.FunctionDef) and n.name=='_load_compressed_weight')
 if not fixed:
  assignments=[n for n in f.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].slice,ast.Constant) and n.targets[0].slice.value=='weight_loader']
  assert len(assignments)==1
  f.body.remove(assignments[0])
 funcs=[wrapper,f]
 c=ast.ClassDef(name='Method',bases=[],keywords=[],body=funcs,decorator_list=[])
 exec(compile(ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[])),'<ct-factory>', 'exec'),g)
 return g['Method']()
def run(fixed,tp,rank,inter,hidden):
 m=get_method(fixed);owner=Loader();owner.quant_method=m;owner.scheme=None;owner.use_flashinfer_trtllm_moe=False;owner.use_triton_kernels=False;owner.use_padded_loading=False;owner.use_presharded_weights=False;owner.moe_tp_size=tp;owner.moe_tp_rank=rank;owner.moe_runner_config=types.SimpleNamespace(is_gated=True);owner.quant_config=types.SimpleNamespace(get_name=lambda:'compressed-tensors')
 owner._load_gguf_weight=lambda *a:False;owner._maybe_load_fp8_shared_expert_as_fp4=lambda **k:False
 layer=torch.nn.Module();m.create_weights(layer,1,hidden,inter//tp,torch.float32,weight_loader=owner._weight_loader_impl)
 records=[]
 for shard in ['w1','w3','w2']:
  for suffix in ['weight_scale','weight_packed']:
   name=('w2_' if shard=='w2' else 'w13_')+suffix;p=getattr(layer,name)
   shape=(hidden,1) if shard=='w2' else (inter,1)
   if suffix=='weight_packed':shape=(hidden,inter//8) if shard=='w2' else (inter,hidden//8)
   loaded=torch.arange(shape[0]*shape[1],dtype=p.dtype).reshape(shape)+(1000 if shard=='w3' else 0)
   p.weight_loader(p,loaded,name,shard,0)
   records.append((shard,suffix,loaded.clone()))
 for shard,suffix,loaded in records:
  actual=getattr(layer,('w2_' if shard=='w2' else 'w13_')+suffix).data[0]
  expected=loaded.t().contiguous()
  if shard!='w2':
   width=inter//tp;expected=expected[:,rank*width:(rank+1)*width];actual=actual[:,(width if shard=='w3' else 0):(2*width if shard=='w3' else width)]
  elif suffix=='weight_packed':expected=expected[rank*(inter//8//tp):(rank+1)*(inter//8//tp),:]
  torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 return len(records)
cases=[]
try:run(False,1,0,2304,32)
except RuntimeError as e:
 assert "broadcast shape [2304, 1]" in str(e);cases.append({'name':'actual_original_scale_layout_failure','passed':True,'error':str(e)})
else:raise AssertionError('original failed to reproduce')
for tp,rank in [(1,0),(8,0),(8,7)]:
 count=run(True,tp,rank,2304,32);cases.append({'name':f'corrected_native_loader_tp{tp}_rank{rank}_all_three_projections','passed':True,'records':count})
print(json.dumps({'cases':cases,'actual_cpu_tensors':True,'native_loader_AST':True,'framework_placement_and_unrelated_quant_methods':'explicit CPU peers','GPU_or_model':False},indent=2))
