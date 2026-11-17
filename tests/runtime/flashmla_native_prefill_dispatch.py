import ast,json,pathlib,sys,types
source=pathlib.Path('/home/sglang/sglang-plugin-das/hcu_sglang/layers/attention/debug_flash_mla_adapter.py')
node=next(n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='flash_mla_sparse_fwd_entrypoint')
selection={'global':'torch'};peers={};events=[]
for name in ['sglang','sglang.srt','sglang.srt.environ','flash_mla','flash_mla.flash_mla_interface','hcu_sglang','hcu_sglang.utils','hcu_sglang.utils.detection','hcu_sglang.layers','hcu_sglang.layers.attention','hcu_sglang.layers.attention.dsv41_sparse_prefill_triton']:
 peer=types.ModuleType(name);peer.__path__=[];peers[name]=peer;sys.modules[name]=peer
peers['sglang.srt.environ'].envs=types.SimpleNamespace(SGLANG_HACK_FLASHMLA_BACKEND=types.SimpleNamespace(get=lambda:selection['global']))
peers['hcu_sglang.utils.detection'].is_hcu=lambda:True
def native(*args,**kwargs):events.append('kernel');return 'kernel'
peers['flash_mla.flash_mla_interface'].flash_mla_sparse_fwd=native
peers['hcu_sglang.layers.attention.dsv41_sparse_prefill_triton'].sparse_prefill=lambda *a,**k:'triton'
environment={};ns={'os':types.SimpleNamespace(environ=environment),'torch_native_flash_mla_sparse_fwd':lambda *a,**k:'torch'}
ns['kernel_flash_mla_sparse_fwd']=lambda *a,**k:peers['flash_mla.flash_mla_interface'].flash_mla_sparse_fwd(*a,**k)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),ns);function=ns[node.name];cases=[]
for explicit,global_value,expected in [('', 'torch','torch'),('kernel','torch','kernel'),('triton','torch','triton'),('torch','kernel','torch'),('','kernel','kernel')]:
 environment['SGLANG_HCU_SPARSE_PREFILL_BACKEND']=explicit;selection['global']=global_value;assert function('sentinel')==expected;cases.append({'explicit':explicit,'global':global_value,'result':expected})
environment['SGLANG_HCU_SPARSE_PREFILL_BACKEND']='bad'
try:function()
except ValueError:cases.append({'case':'invalid_value','passed':True})
else:raise AssertionError('invalid backend accepted')
error=RuntimeError('native_error_identity')
def failed(*args,**kwargs):raise error
peers['flash_mla.flash_mla_interface'].flash_mla_sparse_fwd=failed;environment['SGLANG_HCU_SPARSE_PREFILL_BACKEND']='kernel';selection['global']='torch'
try:function()
except RuntimeError as actual:assert actual is error;cases.append({'case':'native_error_not_swallowed','passed':True})
else:raise AssertionError('native error swallowed')
assert len(events)==2
print(json.dumps({'passed':True,'count':len(cases),'cases':cases,'scope':'actual dispatch function with explicit CPU native/Torch/framework peers; no GPU numeric qualification'}))
