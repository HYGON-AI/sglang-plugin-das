import ast
import pathlib
import sys
import types


def run(backend):
    tree = ast.parse(pathlib.Path(backend).read_bytes())
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == '_hcu_dense_fp4_mqa_logits')
    fn.returns = None
    for arg in fn.args.args+fn.args.kwonlyargs:
        arg.annotation = None
    calls = []
    original_modules = {}
    for name in ['lightop.attention','hcu_sglang.models.deepseek.current_v4.hcu_backend.dense_mqa_out']:
        original_modules[name] = sys.modules.get(name)
    attention = types.ModuleType('lightop.attention')
    attention.fp8_fp4_mqa_logits = lambda *a,**k: calls.append(('baseline',a,k)) or 'baseline'
    output_module = types.ModuleType('hcu_sglang.models.deepseek.current_v4.hcu_backend.dense_mqa_out')
    output_module.make_dense_mqa_output_workspace = lambda *a: calls.append(('allocate',a)) or 'workspace'
    sys.modules['lightop.attention'] = attention
    sys.modules[output_module.__name__] = output_module
    runtime_name = 'sglang.srt.runtime_context'
    runtime_original = sys.modules.get(runtime_name)
    runtime = types.ModuleType(runtime_name)
    runtime.get_schedule = lambda:types.SimpleNamespace(mem_fraction_static=.85)
    sys.modules[runtime_name] = runtime
    cases = []
    try:
        ns = dict(_log_lightop_candidate_fp4_indexer_once=lambda:None,_log_lightop_dense_fp4_indexer_once=lambda:None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(backend),'exec'),ns)
        dispatch = ns[fn.name]
        args = ('q','kv','w','ks','ke',128)
        assert dispatch(*args) == 'baseline' and calls[-1] == ('baseline',args[:5]+(False,128),{})
        cases.append('original_dense_fallback')
        assert dispatch(*args,candidate_blocks=['blocks'],candidate_query_lens=[3]) == 'baseline'
        assert calls[-1][2] == dict(candidate_blocks=['blocks'],candidate_query_lens=[3])
        cases.append('original_candidate_fallback')
        operation = lambda *a: calls.append(('out',a)) or 'output'
        assert dispatch(*args,output='storage',output_operation=operation) == 'output'
        assert calls[-1] == ('out',args[:5]+(False,128,None,None,'storage'))
        cases.append('owned_dense_output')
        assert dispatch(*args,output='storage',output_operation=operation,candidate_blocks=['blocks'],candidate_query_lens=[3]) == 'output'
        assert calls[-1] == ('out',args[:5]+(False,128,['blocks'],[3],'storage'))
        cases.append('owned_candidate_output')
        error = RuntimeError('native failure')
        def fail(*a):
            raise error
        try:
            dispatch(*args,output='storage',output_operation=fail)
        except RuntimeError as actual:
            assert actual is error
        else:
            raise AssertionError('native error swallowed')
        cases.append('native_error_identity')

        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'DeepseekV4AttnBackend')
        init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '__init__')
        assert isinstance(init.body[-2],ast.Assign) and isinstance(init.body[-1],ast.If)
        assert ast.unparse(init.body[-2].targets[0]) == 'self._dense_mqa_workspace'
        tail = ast.Module(body=init.body[-2:],type_ignores=[])
        for change in [None,'hcu','model','ratios','draft','bounded','cp','decode']:
            self = types.SimpleNamespace(is_dsv41=change!='model',low_ratios=() if change=='ratios' else (1,2),is_draft_runner=change=='draft',enable_decoder_swa_bounded_replay=change!='bounded',device='cuda:0')
            ns = dict(self=self,_is_hcu=change!='hcu',get_parallel=lambda:types.SimpleNamespace(attn_cp_size=1 if change=='cp' else 8),get_disagg=lambda:types.SimpleNamespace(disaggregation_mode='decode' if change=='decode' else 'prefill'),logger=types.SimpleNamespace(info=lambda *a:None))
            output_module.make_dense_mqa_output_workspace = lambda *a:types.SimpleNamespace(capacity_bytes=32)
            exec(compile(ast.fix_missing_locations(tail),str(backend),'exec'),ns)
            assert (self._dense_mqa_workspace is not None) == (change is None)
            cases.append('startup_admission_'+str(change))
        dense = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '_low_ratio_index_topk_dense')
        chunked = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '_low_ratio_index_topk_chunked')
        for fn in [dense,chunked]:
            matched = [n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_hcu_dense_fp4_mqa_logits']
            assert len(matched) == 1
            keywords = {n.arg:n.value for n in matched[0].keywords}
            assert isinstance(keywords['output'],ast.IfExp) and isinstance(keywords['output_operation'],ast.IfExp)
            assert ast.unparse(keywords['output'].test) == ast.unparse(keywords['output_operation'].test) == 'self._dense_mqa_workspace is not None'
            cases.append('real_callsite_'+fn.name)
    finally:
        for name,value in list(original_modules.items())+[(runtime_name,runtime_original)]:
            if value is None:
                sys.modules.pop(name,None)
            else:
                sys.modules[name] = value
    return dict(passed=True,cases=cases,scope='Actual dispatch function and constructor admission AST under explicit native/runtime/source peers; native arithmetic independently qualified, no model/HCU claim from this CPU test.')


if __name__ == '__main__':
    import json
    result = run(sys.argv[1])
    pathlib.Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
