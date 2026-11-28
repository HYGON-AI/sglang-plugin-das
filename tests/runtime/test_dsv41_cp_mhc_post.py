"""Check the residual conversion selected for the gfx938 V4.1 CP model."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hcu_sglang.environ import register_envs
from hcu_sglang.utils.detection import is_hcu_native_fp8_supported


@pytest.mark.skipif(not is_hcu_native_fp8_supported(), reason="requires gfx938")
def test_cp_post_truncates_and_replays_changed_inputs(monkeypatch):
    register_envs()
    from hcu_sglang.models.deepseek.current_v4.execution import mhc
    monkeypatch.delenv("SGLANG_OPT_USE_TILELANG_MHC_POST", raising=False)
    monkeypatch.setenv("SGLANG_OPT_USE_AITER_MHC_POST", "1")
    cfg = SimpleNamespace(mult=4, hidden=5120, cp_prefill=True, pre_from_prev=True)
    x = torch.full((20, 5120), 1.0078125, device="cuda", dtype=torch.bfloat16)
    residual = x.new_zeros((20, 4, 5120))
    post = torch.full((20, 4), .998, device="cuda")
    comb = post.new_zeros((20, 4, 4))
    with patch.object(mhc, "get_platform", return_value=SimpleNamespace(is_blackwell=False, is_sm90=False)):
        actual = mhc.post(cfg, x, residual, post, comb)
        torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = mhc.post(cfg, x, residual, post, comb)
        for value in (-1.0078125, 2.015625):
            x.fill_(value)
            graph.replay()
            torch.testing.assert_close(replay, torch.full_like(replay, int(value)), rtol=0, atol=0)
        monkeypatch.setenv("SGLANG_OPT_USE_AITER_MHC_POST", "0")
        with patch("sglang.kernels.ops.layernorm.mhc.mhc_post", return_value=residual) as legacy:
            assert mhc.post(cfg, x, residual, post, comb) is residual
            legacy.assert_called_once()


@pytest.mark.skipif(not is_hcu_native_fp8_supported(), reason="requires gfx938")
def test_combined_rows_select_hcu_norm_and_keep_explicit_override(monkeypatch):
    from unittest.mock import Mock
    from sglang.srt.layers.layernorm import RMSNorm

    register_envs()
    from hcu_sglang.models.deepseek.current_v4.execution import mhc

    cfg = SimpleNamespace(mult=4, hidden=5120, cp_prefill=True, pre_from_prev=True)
    norm = Mock(spec=RMSNorm)
    norm.weight = torch.ones(5120, dtype=torch.bfloat16, device="cuda")
    norm.cast_x_before_out_mul = False
    norm.variance_size_override = None
    hc = SimpleNamespace(cfg=cfg, norm=norm)
    rows = torch.ones((8, 4, 5120), dtype=torch.bfloat16, device="cuda")
    state = SimpleNamespace(residual=rows, pre=torch.full((8, 4), .25, device="cuda"))
    with patch.object(mhc, "get_platform", return_value=SimpleNamespace(is_blackwell=False, is_sm90=False)), patch(
        "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled", return_value=False
    ):
        monkeypatch.delenv("SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM", raising=False)
        assert mhc._combine(hc, state, None) is norm.forward_hip.return_value
        torch.testing.assert_close(norm.forward_hip.call_args.args[0], rows[:, 0])
        monkeypatch.setenv("SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM", "0")
        assert mhc._combine(hc, state, None) is norm.return_value
