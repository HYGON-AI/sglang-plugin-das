import gzip
import json
import re
import sys
from pathlib import Path

class MissingCPUStack(ValueError):
    """A trace recorded CPU events without the Python call stack."""


def verify_trace(path, stage, graph_enabled):
    steps = set()
    cpu_ops = gpu_runtime_calls = kernels = graph_launches = 0
    stack_frames = parent_links = 0
    label = re.compile(r'M31_PROFILE_(?:DECODE|PREFILL)_STEP_(\d+)_LOCAL_BS_(\d+)_FORWARD_\d+')
    kernel = re.compile(r'"cat"\s*:\s*"kernel"')
    cpu_op = re.compile(r'"cat"\s*:\s*"cpu_op"')
    gpu_runtime = re.compile(r'"cat"\s*:\s*"(?:cuda|hip)_(?:runtime|driver)"')
    python_frame = re.compile(r'"cat"\s*:\s*"python_function"')
    python_parent = re.compile(r'"Python parent id"\s*:\s*[0-9]+')
    launch = re.compile(r'"name"\s*:\s*"(?:hip|cuda|cu)GraphLaunch[^"]*"', re.IGNORECASE)
    last_line = ''
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                last_line = line.strip()
            match = label.search(line)
            if match:
                steps.add(tuple(map(int, match.groups())))
            stack_frames += bool(python_frame.search(line))
            parent_links += bool(python_parent.search(line))
            cpu_ops += bool(cpu_op.search(line))
            gpu_runtime_calls += bool(gpu_runtime.search(line))
            kernels += bool(kernel.search(line))
            graph_launches += bool(launch.search(line))
    expected = {(1, 4)} if stage == 'Prefill' else {(i, 8) for i in range(11, 21)}
    if last_line != '}' or steps != expected or not kernels:
        raise ValueError(f'{path.name}: invalid trace, steps={steps}, kernels={kernels}')
    if not cpu_ops or not gpu_runtime_calls:
        raise ValueError(f'{path.name}: missing CPU or GPU runtime calls '
                         f'(cpu_ops={cpu_ops}, gpu_runtime_calls={gpu_runtime_calls})')
    if not stack_frames or not parent_links:
        raise MissingCPUStack(f'{path.name}: CPU call stack is missing')
    if bool(graph_launches) != graph_enabled:
        raise ValueError(f'{path.name}: graph replay count {graph_launches} conflicts with config')
    return dict(name=path.name, bytes=path.stat().st_size, steps=sorted(steps),
                cpu_ops=cpu_ops, gpu_runtime_calls=gpu_runtime_calls,
                cpu_stack_frames=stack_frames, cpu_stack_parent_links=parent_links,
                kernels=kernels, graph_launches=graph_launches)


def main():
    run = Path(sys.argv[1])
    stage = sys.argv[2].capitalize()
    graph = sys.argv[3] == "full"
    paths = sorted((run / "traces").glob("*.trace.json.gz"))
    ranks = sorted(int(re.search(r"TP-(\d+)", path.name).group(1)) for path in paths)
    if ranks != list(range(8)):
        raise ValueError(f"Expected trace files for TP ranks 0–7, found {ranks}")
    traces = [verify_trace(path, stage, graph) for path in paths]
    benchmark = run / "measurements" / "PASS.json"
    result = json.loads(benchmark.read_text()) if benchmark.is_file() else None
    bs = 4 if stage == "Prefill" else 64
    complete = bool(result and result["completed"] == bs and result["cache_min"] == 131072)
    index = dict(stage=stage, runtime="cuda_graph" if graph else "eager",
                 traces=traces, input_tokens=147456, cached_tokens=131072, batch_size=bs,
                 output_tokens=1 if stage == "Prefill" else 1024,
                 benchmark_completed=complete, benchmark=result)
    target = run / ("PASS.json" if complete else "CAPTURE.json")
    target.write_text(json.dumps(index, indent=2) + "\n")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
