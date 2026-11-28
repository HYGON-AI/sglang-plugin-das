# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat
from hcu_sglang.environ import register_envs

bootstrap_aiter_compat()
register_envs()

from hcu_sglang.kernels import mhc as kernels
from hcu_sglang.models.deepseek.current_v4.execution import mhc
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def test_cp_attention_post_carries_normalized_rows_and_precomputed_stats():
    y = torch.zeros(3, 2, dtype=torch.bfloat16)
    residual = torch.zeros(3, 4, 2, dtype=y.dtype)
    pre = torch.ones(3, 4)
    stats = (pre, pre + 1, torch.zeros(3, 4, 4))
    packed = torch.ones(4)
    hc = SimpleNamespace(cfg=SimpleNamespace(cp_prefill=True))
    norm = SimpleNamespace(weight=y[0], variance_epsilon=1e-6)
    ff = SimpleNamespace(fn=y, scale=pre, base=pre, norm=norm,
                         cfg=SimpleNamespace(rms_eps=1e-6, eps=1e-8, sinkhorn_iters=20))
    kwargs = dict(stats_stream=None, next=SimpleNamespace(hc=ff),
                  world_size=8, precomputed=stats)
    with patch.object(mhc, "_is_hcu", True), patch.object(
        kernels, "hcu_mhc_mega_boundary", return_value=(residual, y, stats)
    ) as fused, patch.object(mhc, "_post_fusion") as fallback:
        state = mhc.run_attn_post(hc, y, residual, mode=ForwardMode.EXTEND,
                                  fn_packed=packed, **kwargs)
        assert state.streams is residual and state.pre is pre and state.stats is stats
        assert fused.call_args.args[2] is pre and fused.call_args.kwargs["fn_packed"] is packed
        for mode, weight in ((ForwardMode.DECODE, packed), (ForwardMode.TARGET_VERIFY, packed),
                             (ForwardMode.EXTEND, None)):
            assert mhc.run_attn_post(hc, y, residual, mode=mode, fn_packed=weight,
                                     **kwargs) is fallback.return_value
        assert fused.call_count == 1 and fallback.call_count == 3


def test_pack_preserves_pointer_and_respects_disabled_mode():
    hc = SimpleNamespace(
        cfg=SimpleNamespace(pre_from_prev=True), fn=torch.empty((24, 20480)),
        norm=SimpleNamespace(cast_x_before_out_mul=False, variance_size_override=None,
                             weight=torch.ones(1, dtype=torch.bfloat16)),
    )
    previous = torch.zeros(3)
    with patch.object(mhc, "_is_hcu", True), patch.object(
        mhc.envs.SGLANG_HCU_MEGA_MHC_OPT, "get", return_value=False
    ):
        assert mhc.prepare_hcu_mega_weight(hc, previous) is None
    with patch.object(mhc, "_is_hcu", True), patch.object(
        mhc.envs.SGLANG_HCU_MEGA_MHC_OPT, "get", return_value=True
    ), patch(
        "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled", return_value=False
    ), patch(
        "sglang.srt.runtime_context.get_exec",
        return_value=SimpleNamespace(deterministic=SimpleNamespace(enable_deterministic_inference=False))
    ), patch.object(kernels, "hcu_mega_mhc_backend") as backend:
        backend.return_value.mega_mhc_prepare_fn.return_value = torch.ones(3)
        assert mhc.prepare_hcu_mega_weight(hc, previous) is previous
        torch.testing.assert_close(previous, torch.ones(3))
