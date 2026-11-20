import os
import ast
import __future__
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hcu_sglang.models.qwen import ple_activation as A

if os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1":
    from sglang.srt.models import qwen4_exp as N
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
else:
    source = Path(importlib.util.find_spec("sglang").origin).parent / "srt/models/qwen4_exp.py"
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpPLELayer")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_short_conv")
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    N = SimpleNamespace(Qwen4ExpPLELayer=SimpleNamespace(_short_conv=namespace["_short_conv"]))
    ForwardMode = SimpleNamespace(EXTEND=SimpleNamespace(is_target_verify=lambda: False))


def make_case(monkeypatch, device, lengths=(7, 3), channels=16, dtype=torch.float32):
    state = torch.randn(4, channels, 4, device=device, dtype=dtype)
    weight = torch.randn(channels, 1, 3, device=device, dtype=dtype)
    length = torch.tensor(lengths, device=device)
    req = torch.repeat_interleave(torch.arange(len(lengths), device=device), length)
    offsets = torch.cat([torch.arange(n, device=device) for n in lengths])
    indices = torch.arange(len(lengths), device=device) + 1
    x = torch.randn(sum(lengths), channels, device=device, dtype=dtype)
    pool = SimpleNamespace(short_conv_layer_cache=lambda _: state)
    monkeypatch.setitem(N.Qwen4ExpPLELayer._short_conv.__globals__, "get_req_to_token_pool", lambda: pool)
    monkeypatch.setitem(N.Qwen4ExpPLELayer._short_conv.__globals__, "_ple_track_targets", lambda *args: None)
    layer = SimpleNamespace(layer_id=0, short_conv_state_len=4,
                            conv_channels=channels, short_conv_dilation=2,
                            conv1d=SimpleNamespace(weight=weight))
    batch = SimpleNamespace(mode=ForwardMode.EXTEND, use_decode_fast_path=False,
                            lengths=length, row_width=max(lengths), req_indices=req,
                            token_offsets=offsets, state_indices=indices)
    forward = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    A._build.cache_clear()
    return layer, x, forward, batch, state


def independent_reference(layer, x, batch, initial):
    outputs = []
    next_state = initial.clone()
    cursor = 0
    for rank, n in enumerate(batch.lengths.tolist()):
        seq = x[cursor:cursor+n]
        joined = torch.cat([initial[rank+1], seq.T], dim=1).float()
        y = sum(joined[:, j*2:j*2+n].T * layer.conv1d.weight[:, 0, j].float()
                for j in range(3))
        outputs.append(y * torch.sigmoid(y))
        next_state[rank+1].copy_(joined[:, n:n+4])
        cursor += n
    return torch.cat(outputs), next_state


@pytest.mark.parametrize("lengths", [(7, 3), (0, 5), (2, 1)])
def test_community_method_math_state_and_no_input_alias(monkeypatch, lengths):
    layer, x, forward, batch, state = make_case(monkeypatch, "cpu", lengths)
    initial, original_x = state.clone(), x.clone()
    expected, expected_state = independent_reference(layer, x, batch, initial)
    original = N.Qwen4ExpPLELayer._short_conv
    original_f = original.__globals__["F"]
    with torch.no_grad():
        baseline = original(layer, x, forward, batch)
        state.copy_(initial)
        result = A._build(original)(layer, x, forward, batch)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result, baseline, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state)
    assert torch.equal(x, original_x)
    assert result.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    assert original.__globals__["F"] is original_f


def test_unknown_upstream_method_is_rejected():
    def changed(*args):
        return args
    with pytest.raises(RuntimeError, match="community method changed"):
        A._build(changed)


@pytest.mark.skipif(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1", reason="explicit GPU lane")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_registered_hook_peak_memory_math_and_changed_graph(monkeypatch, dtype):
    from hcu_sglang.models.qwen import qwen4_exp as Q
    from sglang.srt.plugins.hook_registry import HookRegistry
    monkeypatch.setenv("SGLANG_QWEN4_PLE_PREFILL_INPLACE_SILU", "1")
    monkeypatch.setattr(Q, "is_hcu", lambda: True)
    from hcu_sglang.environ import register_envs
    register_envs()
    records = []
    monkeypatch.setattr(HookRegistry, "register", lambda *args: records.append(args))
    Q.register_hooks()
    target = "sglang.srt.models.qwen4_exp.Qwen4ExpPLELayer._short_conv"
    selected = [(kind, fn, None) for name, fn, kind in records if name == target]
    assert len(selected) == 1
    original = N.Qwen4ExpPLELayer._short_conv
    import sglang.srt.runtime_context as runtime
    monkeypatch.setattr(runtime, "get_parallel", lambda: SimpleNamespace(attn_cp_size=1))
    HookRegistry._apply_target(target, selected)
    patched = N.Qwen4ExpPLELayer._short_conv
    try:
        N.Qwen4ExpPLELayer._short_conv = original
        layer, x, forward, batch, state = make_case(monkeypatch, "cuda", (1024, 512), 10240, dtype)
        N.Qwen4ExpPLELayer._short_conv = patched
        initial, original_x = state.clone(), x.clone()
        with torch.no_grad():
            original(layer, x, forward, batch)
            patched(layer, x, forward, batch)
            torch.cuda.synchronize()
            state.copy_(initial)
            torch.cuda.reset_peak_memory_stats()
            base_start = torch.cuda.memory_allocated()
            baseline = original(layer, x, forward, batch)
            baseline_peak = torch.cuda.max_memory_allocated() - base_start
            expected = baseline.clone()
            expected_state = state.clone()
            del baseline
            torch.cuda.synchronize()
            state.copy_(initial)
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.memory_allocated()
            result = patched(layer, x, forward, batch)
            peak = torch.cuda.max_memory_allocated() - start
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
            assert torch.equal(x, original_x)
            print(dict(dtype=str(dtype), baseline_peak=baseline_peak, inplace_peak=peak,
                       saved_bytes=baseline_peak-peak, output_bytes=x.numel()*x.element_size()))
            assert baseline_peak - peak >= x.numel() * x.element_size()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = patched(layer, x, forward, batch)
            for seed in (41, 93):
                torch.manual_seed(seed)
                x.normal_()
                initial.normal_()
                state.copy_(initial)
                graph.replay()
                actual = captured.clone()
                actual_state = state.clone()
                state.copy_(initial)
                reference = original(layer, x, forward, batch)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                torch.testing.assert_close(actual_state, state, rtol=0, atol=0)
    finally:
        N.Qwen4ExpPLELayer._short_conv = original
