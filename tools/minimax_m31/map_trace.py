#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

FAMILIES = re.compile(
    r"minimax|dspark|slimquant|compressed_tensors|deep_ep|deepgemm|"
    r"deep_gemm|flash_mla|flash_attn|lightop|all_gather|rmsnorm",
    re.IGNORECASE,
)


def collect(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    python = {}
    stacks = Counter()
    correlations = {}
    launches = {}
    cpu_operations = {}
    kernels = Counter()
    linked = Counter()
    for event in events:
        args = event.get("args", {})
        cat = event.get("cat")
        name = event.get("name", "")
        if cat == "python_function" and "Python id" in args:
            key = (event["pid"], args["Python id"])
            python[key] = (name, args.get("Python parent id"))
            if FAMILIES.search(name):
                stacks[key] += event.get("dur", 0)
        elif cat == "cpu_op" and "External id" in args:
            cpu_operations[args["External id"]] = name
        elif cat in ("cuda_runtime", "cuda_driver", "hip_runtime"):
            if "correlation" in args:
                launches[args["correlation"]] = name
                correlations[args["correlation"]] = args.get("External id")
    for event in events:
        if event.get("cat") != "kernel":
            continue
        correlation = event.get("args", {}).get("correlation")
        name = event["name"]
        kernels[name] += event.get("dur", 0)
        runtime = launches.get(correlation)
        cpu_op = cpu_operations.get(correlations.get(correlation))
        if runtime and cpu_op:
            linked[(cpu_op, runtime, name)] += 1

    def parents(key):
        names = []
        for _ in range(12):
            frame = python.get(key)
            if frame is None:
                break
            name, parent = frame
            names.append(name)
            key = (key[0], parent)
        return names

    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        sha = digest.hexdigest()
    return {
        "file": path.name,
        "sha256": sha,
        "python_frames": len(python),
        "stacks": [{"frames": parents(key), "inclusive_us": us} for key, us in stacks.most_common(120)],
        "kernel_us": kernels.most_common(80),
        "linked_launches": [
            {"cpu_op": op, "runtime": runtime, "kernel": kernel, "count": count}
            for (op, runtime, kernel), count in linked.most_common(80)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    index = json.loads((args.trace_root / "TRACE_INDEX.json").read_text())
    for case, entry in index["cases"].items():
        if not case.startswith("no_cudagrpah/") or not entry["benchmark_completed"]:
            continue
        traces = sorted((args.trace_root / case / entry["run"]).rglob("*.trace.json.gz"))
        if len(traces) != entry["trace_count"]:
            raise ValueError(f"{case}: expected {entry['trace_count']} traces, found {len(traces)}")
        output = args.output / case
        output.mkdir(parents=True, exist_ok=True)
        for rank, path in enumerate(traces):
            print(case, path.name, flush=True)
            result = collect(path)
            dest = output / f"trace-{rank}.json"
            temp = dest.with_suffix(".tmp")
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            temp.replace(dest)


if __name__ == "__main__":
    main()
