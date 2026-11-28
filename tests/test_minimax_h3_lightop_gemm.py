# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hcu_sglang.multimodal_gen import minimax_h3_lightop_gemm as lightop_gemm
from hcu_sglang.multimodal_gen import minimax_h3_quant_env as envs

FLAG = "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_LIGHTOP_GEMM"
WHOLE_FLAG = "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_LIGHTOP_GEMM_WHOLE"
requires_gfx936 = pytest.mark.skipif(
    os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1"
    or torch.version.hip is None
    or not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936",
    reason="Requires an isolated gfx936 GPU lane with HCU_PLUGIN_RUN_GPU_TESTS=1",
)


@pytest.fixture(autouse=True)
def reset_capability_cache():
    lightop_gemm._ordered_gemm.cache_clear()
    yield
    lightop_gemm._ordered_gemm.cache_clear()


def test_lightop_is_opt_in(monkeypatch):
    for flag in (FLAG, WHOLE_FLAG):
        monkeypatch.delenv(flag, raising=False)
        assert not getattr(envs, flag)
        monkeypatch.setenv(flag, "1")
        assert getattr(envs, flag)


def test_missing_and_legacy_lightop():
    with patch.object(lightop_gemm.importlib, "import_module", side_effect=ImportError):
        assert lightop_gemm._ordered_gemm() is None
    lightop_gemm._ordered_gemm.cache_clear()

    def gemm(a, b, sa, sb, bias, dtype):
        return False, None

    with patch.object(
        lightop_gemm.importlib,
        "import_module",
        return_value=SimpleNamespace(gemm_w8a8_smooth=gemm),
    ):
        assert lightop_gemm._ordered_gemm() is None


def _inputs(m=65, n=256, k=128):
    generator = torch.Generator(device="cuda").manual_seed(47)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    w = torch.randint(-127, 128, (n, k), device=x.device, dtype=torch.int8, generator=generator)
    sw = torch.rand((n, 1), device=x.device, generator=generator) * 0.01
    return x, w, sw


@requires_gfx936
@pytest.mark.parametrize("m", [1, 65])
def test_ordered_lightop_matches_linear(monkeypatch, m):
    from hcu_sglang.multimodal_gen.minimax_h3_quant import linear

    x, w, sw = _inputs(m)
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(WHOLE_FLAG, raising=False)
    with torch.inference_mode():
        expected = linear(x, w, sw, chunk=32)
    monkeypatch.setenv(FLAG, "1")
    hits = []
    native_gemm = lightop_gemm._ordered_gemm()
    assert native_gemm is not None

    def gemm(*args, **kwargs):
        assert kwargs == {"input_scale_first": True}
        supported, result = native_gemm(*args, **kwargs)
        assert supported
        hits.append(args[0].shape[0])
        return supported, result

    with torch.inference_mode(), patch.object(lightop_gemm, "_ordered_gemm", return_value=gemm):
        actual = linear(x, w, sw, chunk=32)

    assert hits == [32] * ((m + 31) // 32)
    assert actual.shape == (m, w.shape[0])
    assert torch.equal(actual, expected)


@requires_gfx936
def test_ordered_lightop_rounds_before_weight_scale(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    q = torch.ones((32, 128), device="cuda", dtype=torch.int8)
    w = torch.zeros((256, 128), device=q.device, dtype=q.dtype)
    w[:, 0], w[:, 1] = 127, 10
    sx = torch.full((len(q), 1), 0.06446823477745056, device=q.device)
    sw = torch.full((len(w), 1), 0.07585038244724274, device=q.device)
    acc = torch.full((len(q), len(w)), 137, device=q.device, dtype=torch.float32)
    expected = ((acc * sx) * sw.t()).bfloat16()
    folded = (acc * (sx * sw.t())).bfloat16()
    assert torch.all(expected != folded)

    with torch.inference_mode():
        actual = lightop_gemm.try_lightop_gemm(q, w, sx, sw)
    assert actual is not None
    assert torch.equal(actual, expected)


@requires_gfx936
@pytest.mark.parametrize("case", ["bias", "reference", "unaligned", "declined"])
def test_lightop_fallback_preserves_linear(monkeypatch, case):
    from hcu_sglang.multimodal_gen.minimax_h3_quant import linear

    x, w, sw = _inputs(n=80, k=112) if case == "unaligned" else _inputs()
    bias = torch.full((len(w),), 0.5, device=x.device) if case == "bias" else None
    mode = "reference" if case == "reference" else "both"
    monkeypatch.delenv(FLAG, raising=False)
    with torch.inference_mode():
        expected = linear(x, w, sw, bias, mode=mode, chunk=32)

    monkeypatch.setenv(FLAG, "1")
    with torch.inference_mode():
        if case == "unaligned":
            actual = linear(x, w, sw, bias, mode=mode, chunk=32)
        else:
            with patch.object(lightop_gemm, "_ordered_gemm") as capability:
                gemm = capability.return_value
                gemm.return_value = (False, None)
                actual = linear(x, w, sw, bias, mode=mode, chunk=32)
                if case in ("bias", "reference"):
                    capability.assert_not_called()
                else:
                    assert gemm.call_count == 3

    assert torch.equal(actual, expected)


@requires_gfx936
def test_lightop_kernel_failure_propagates(monkeypatch):
    from hcu_sglang.multimodal_gen.minimax_h3_quant import linear

    monkeypatch.setenv(FLAG, "1")
    x, w, sw = _inputs()
    with (
        torch.inference_mode(),
        patch.object(lightop_gemm, "_ordered_gemm") as capability,
        pytest.raises(RuntimeError, match="kernel failure"),
    ):
        capability.return_value.side_effect = RuntimeError("kernel failure")
        linear(x, w, sw)


@requires_gfx936
@pytest.mark.parametrize("m", [1, 65])
def test_whole_projection_returns_lightop_storage(monkeypatch, m):
    from hcu_sglang.multimodal_gen.minimax_h3_quant import MiniMaxH3Int8LinearMethod

    x, w, sw = _inputs(m)
    layer = SimpleNamespace(weight=w, weight_scale_inv=sw)
    method = MiniMaxH3Int8LinearMethod(fused=True)
    monkeypatch.delenv(FLAG, raising=False)
    with torch.inference_mode():
        expected = method.apply(layer, x)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(WHOLE_FLAG, "1")

    native_gemm = lightop_gemm._ordered_gemm()
    assert native_gemm is not None
    outputs = []

    def gemm(*args, **kwargs):
        supported, result = native_gemm(*args, **kwargs)
        assert supported
        outputs.append(result)
        return supported, result

    with torch.inference_mode(), patch.object(lightop_gemm, "_ordered_gemm", return_value=gemm):
        actual = method.apply(layer, x)

    assert len(outputs) == 1
    assert outputs[0].shape[0] == max(32, (m + 15) // 16 * 16)
    assert actual.shape == (m, w.shape[0]) and actual.is_contiguous()
    assert actual.untyped_storage().data_ptr() == outputs[0].untyped_storage().data_ptr()
    assert torch.equal(actual, expected)


@requires_gfx936
@pytest.mark.parametrize("case", ["disabled", "legacy", "unaligned", "declined"])
def test_whole_projection_falls_back_to_bounded_chunks(monkeypatch, case):
    from hcu_sglang.multimodal_gen import minimax_h3_quant as quant

    x, w, sw = _inputs(n=80, k=112) if case == "unaligned" else _inputs()
    monkeypatch.delenv(FLAG, raising=False)
    with torch.inference_mode():
        expected = quant.linear(x, w, sw, chunk=32)
    monkeypatch.setenv(FLAG, "0" if case == "disabled" else "1")
    monkeypatch.setenv(WHOLE_FLAG, "1")

    def decline(*args, **kwargs):
        return False, None

    with (
        torch.inference_mode(),
        patch.object(quant, "quant_rows", wraps=quant.quant_rows) as quantize,
        patch.object(
            lightop_gemm, "_ordered_gemm", return_value=None if case == "legacy" else decline,
        ),
    ):
        actual = quant.linear(x, w, sw, chunk=32)

    rows = [call.args[0].shape[0] for call in quantize.call_args_list]
    assert rows == ([65, 32, 32, 1] if case == "declined" else [32, 32, 1])
    assert torch.equal(actual, expected)


@requires_gfx936
def test_whole_projection_graph_replays_changed_input(monkeypatch):
    from hcu_sglang.multimodal_gen.minimax_h3_quant import linear

    x, w, sw = _inputs()
    changed = -x.roll(1, dims=-1) * 2
    monkeypatch.delenv(FLAG, raising=False)
    with torch.inference_mode():
        expected_initial = linear(x, w, sw, chunk=32)
        expected_changed = linear(changed, w, sw, chunk=32)
    assert not torch.equal(expected_initial, expected_changed)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(WHOLE_FLAG, "1")
    native_gemm = lightop_gemm._ordered_gemm()
    assert native_gemm is not None
    hits = []

    def gemm(*args, **kwargs):
        supported, result = native_gemm(*args, **kwargs)
        assert supported
        hits.append(args[0].shape[0])
        return supported, result

    with (
        torch.inference_mode(),
        patch.object(lightop_gemm, "_ordered_gemm", return_value=gemm),
    ):
        current = torch.cuda.current_stream()
        warmup = torch.cuda.Stream()
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup):
            linear(x, w, sw, chunk=32)
        current.wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = linear(x, w, sw, chunk=32)
        graph.replay()
        assert torch.equal(actual, expected_initial)
        x.copy_(changed)
        graph.replay()

    assert hits == [80, 80]
    assert torch.equal(actual, expected_changed)
