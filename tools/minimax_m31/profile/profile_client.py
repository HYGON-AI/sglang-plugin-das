import argparse
import contextlib
import importlib.util
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['prefill', 'decode'], required=True)
p.add_argument('--output', required=True)
p.add_argument('--trace-dir', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('profile_bench', root/'bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
original_adapter = bench.b.native_adapter
traces = Path(a.trace_dir)
traces.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def adapter(*args, **kwargs):
    session = args[3]
    base = 'http://127.0.0.1:11010'
    request = dict(output_dir=str(traces), profile_by_stage=True,
                   profile_stages=[a.mode], activities=['CPU', 'GPU'],
                   num_steps=10 if a.mode == 'decode' else 100000,
                   with_stack=True, record_shapes=True, detailed_annotations=True,
                   merge_profiles=False, profile_prefix=a.mode)
    def api(name, data):
        response = session.post(base+'/'+name, json=data, timeout=1800)
        response.raise_for_status()
        try:
            reply = response.json()
        except ValueError:
            return response.text
        if isinstance(reply, dict) and reply.get('success') is False:
            raise RuntimeError(f'{name}: {reply}')
        return reply
    reply = api('start_profile', request)
    (traces/'profile-start.json').write_text(json.dumps(dict(request=request,response=reply), indent=2))
    try:
        with original_adapter(*args, **kwargs):
            yield
    finally:
        if a.mode == 'prefill':
            reply = api('stop_profile', {})
            (traces/'profile-stop.json').write_text(json.dumps(reply, indent=2))


bench.b.native_adapter = adapter
bench.main(['--mode', a.mode, '--bs', '4' if a.mode == 'prefill' else '64',
            '--output', a.output] + (['--shared-prefix'] if a.mode == 'prefill' else []))
