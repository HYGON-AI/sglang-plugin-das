# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch

def _case(t, h, length, width):
    rng = torch.Generator().manual_seed(20260924)
    q = torch.randn(t, h, 128, generator=rng).bfloat16()
    payload = torch.randint(256, (length, 64), generator=rng,
                            dtype=torch.uint8).view(torch.int8)
    scales = torch.randint(118, 137, (length, 4), generator=rng,
                           dtype=torch.uint8).view(torch.int32).reshape(length)
    weights = torch.randn(t, h, generator=rng)
    starts = [(row * 37) % (length + 1) for row in range(t)]
    ends = [start + min(length - start, width, (row * 29) % (width + 1))
            for row, start in enumerate(starts)]
    if t:
        ends[0] = min(length, width)
    if t > 1:
        ends[1] = starts[1]  # An empty window with nonempty K.
    return (q, payload, scales, weights,
            torch.tensor(starts, dtype=torch.int32),
            torch.tensor(ends, dtype=torch.int32))

def _reference(q, payload, scales, weights, starts, ends, width):
    # Independent E2M1 decode; do not call any Lightop decoder/kernel.
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    raw = payload.view(torch.uint8).long()
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(-1, 128)
    exponents = scales.view(torch.uint8).reshape(-1, 4).float() - 127
    k = (table[codes] * torch.exp2(exponents).repeat_interleave(32, 1))
    k = k.bfloat16().float()
    result = torch.full((q.shape[0], width), -torch.inf)
    for row in range(q.shape[0]):
        start, end = starts[row].item(), ends[row].item()
        dots = q[row].float() @ k[start:end].T
        result[row, :end - start] = (dots.relu() * weights[row, :, None]).sum(0)
    return result


def run(repo):
    import importlib.util
    module_path = repo/'hcu_sglang/models/deepseek/current_v4/hcu_backend/dense_mqa_out.py'
    spec = importlib.util.spec_from_file_location('dense_mqa_output_runtime_test',module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from lightop.gemmopt import fp8_fp4_mqa_logits as baseline
    workspace = module.DenseMqaOutputWorkspace('cuda',32000000)
    cases = []
    for shape in [(1,1,1,4),(33,3,257,260),(129,32,4097,512)]:
        cpu = _case(*shape)
        q,k,s,w,ks,ke = [x.cuda() for x in cpu]
        output = workspace.view(shape[0],shape[3])
        actual = workspace.operation((q,None),(k,s),w,ks,ke,False,shape[3],None,None,output)
        expected = baseline((q,None),(k,s),w,ks,ke,False,shape[3])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
        torch.testing.assert_close(actual.cpu(),_reference(*cpu,shape[3]),rtol=3e-3,atol=3e-2)
        cases.append('installed_baseline_and_independent_reference_'+str(shape))
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            workspace.operation((q,None),(k,s),w,ks,ke,False,shape[3],None,None,output)
    stream.synchronize()
    with torch.cuda.graph(graph,stream=stream):
        actual = workspace.operation((q,None),(k,s),w,ks,ke,False,shape[3],None,None,output)
    for _ in range(3):
        w.mul_(.5)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual,baseline((q,None),(k,s),w,ks,ke,False,shape[3]),rtol=0,atol=0,equal_nan=True)
    cases.append('actual_packaged_module_graph_three_replays')
    assert workspace.view(32000000,1) is None
    with torch.cuda.stream(stream):
        assert workspace.view(1,4) is None
    cases.append('no_growth_and_other_stream_fallback')
    torch.cuda.synchronize()
    return dict(passed=True,cases=cases,scope='Actual packaged plugin-native output source and installed baseline, independent vendor CPU reference tolerance unchanged; operator qualification only.')

if __name__ == '__main__':
    import json,pathlib,sys
    result = run(pathlib.Path(sys.argv[1]))
    pathlib.Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
