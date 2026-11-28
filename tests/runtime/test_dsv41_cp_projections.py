# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def activate():
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu() or not torch.cuda.is_available():
        pytest.skip("HCU runtime required")
    from sglang.srt.plugins import load_plugins
    load_plugins()


def test_cp_gather_permutation_and_capture():
    from sglang.srt.environ import envs
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import hcu_cp_utils as cp

    for rows, limit, same_storage in ((128, 128, True), (120, 1, False)):
        payload = torch.arange(rows * 16, device="cuda").view(rows, 16)
        expected = payload.view(8, rows // 8, 16).transpose(0, 1).reshape(rows, 16)
        with envs.SGLANG_DSV4_CP_OUTPUT_INPLACE.override(True), envs.SGLANG_DSV4_CP_OUTPUT_INPLACE_MAX_CYCLE.override(limit):
            y = cp._interleave_cp_output(payload, 8)
        assert (payload.data_ptr() == y.data_ptr()) == same_storage
        torch.testing.assert_close(y, expected)
    for offset in (0, 100):
        payload.copy_(torch.arange(payload.numel(), device="cuda").view_as(payload) + offset)
        if offset == 0:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = cp._interleave_cp_output(payload, 8)
        graph.replay()
        assert y.data_ptr() != payload.data_ptr()
        torch.testing.assert_close(y, payload.view(8, -1, 16).transpose(0, 1).reshape_as(payload))


def test_mhc_entry_physical_rows():
    from hcu_sglang.models.deepseek.current_v4.execution import core

    x = torch.arange(18 * 64, device="cuda", dtype=torch.bfloat16).view(18, 64)
    rows = torch.arange(18)
    batch = SimpleNamespace(input_ids=rows, positions=rows, out_cache_loc=rows, extend_seq_lens_cpu=[13])
    for rank in range(8):
        assert core._can_defer_mhc_repeat_cp_split(x, batch, 4, 8, rank)
        actual = core._repeat_mhc_input_on_cp_rank(x, 4, 8, rank)
        torch.testing.assert_close(actual, x[:, None].repeat(1, 4, 1)[rank::8])
    batch.positions = rows[:-1]
    assert not core._can_defer_mhc_repeat_cp_split(x, batch, 4, 8, 0)


def test_mhc_bf16x3_threshold():
    import lightop
    from lightop import split_bf16_hc_weight, split_tf32_hc_weight
    from hcu_sglang.models.deepseek.current_v4.execution import mhc
    from sglang.srt.environ import envs

    torch.manual_seed(727)
    x = torch.randn(128, 4, 5120, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, 20480, device="cuda") * 0.003
    hc = SimpleNamespace(
        fn=fn, scale=torch.ones(3, device="cuda"), base=torch.zeros(24, device="cuda"),
        cfg=SimpleNamespace(sinkhorn_iters=20, rms_eps=1e-6, eps=1e-6, mult=4),
        tf32_parts=split_tf32_hc_weight(fn), bf16_parts=split_bf16_hc_weight(fn),
    )
    platform = SimpleNamespace(is_sm90=False, is_sm100=False, is_blackwell=False)
    with patch.object(mhc, "get_platform", return_value=platform), patch.object(mhc, "_use_hcu_hc_mix_stats", True):
        with envs.SGLANG_DSV4_HC_BF16X3_MIN_ROWS.override(128), patch.object(
            lightop, "hc_mix_stats_sinkhorn_bf16x3", wraps=lightop.hc_mix_stats_sinkhorn_bf16x3
        ) as bf16x3:
            fast = mhc.mix_stats(hc, x)
            bf16x3.assert_called_once()
        with envs.SGLANG_DSV4_HC_BF16X3_MIN_ROWS.override(129), patch.object(
            lightop, "hc_mix_stats_sinkhorn_deepgemm", wraps=lightop.hc_mix_stats_sinkhorn_deepgemm
        ) as tf32:
            baseline = mhc.mix_stats(hc, x)
            tf32.assert_called_once()
    for a, b in zip(fast, baseline, strict=True):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("ratio,local_rows,keep,expect_local",
                         [(1,4096,None,True), (2,4096,None,True),
                          (2,4096,[1],False), (1,2048,None,False)])
def test_compressor_gathers_one_payload(ratio, local_rows, keep, expect_local):
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import backend, dsv41_sparse
    from sglang.srt.environ import envs

    torch.manual_seed(727)
    cp_size, rank = 2, 1
    x = torch.randn(local_rows * cp_size, 128, device="cuda", dtype=torch.bfloat16)
    compressor = dsv41_sparse.DeepseekV41Compressor(128, 32, ratio, 1e-6, fused_compress=(ratio == 2)).cuda()
    layer = SimpleNamespace(compressor=compressor, compress_ratio=ratio)
    full_kv, full_score = compressor.project(x)
    peer = compressor.project_packed(x[::2]) if expect_local else x[::2]
    calls = []
    def gather(payload, batch, stream):
        calls.append(payload.shape)
        return torch.stack((peer, payload), dim=1).flatten(0, 1)
    pos = torch.arange(len(x), device=x.device)
    owner = SimpleNamespace(forward_metadata=SimpleNamespace(late_layer_tail=SimpleNamespace(
        local_lens_cpu=[local_rows], req_global=torch.zeros_like(pos), pos_global=pos)),
        _low_ratio_compress_torch=Mock())
    batch = SimpleNamespace(attn_cp_metadata=object(), extend_seq_lens_cpu=[len(x)],
                            encoder_swa_compress_rows=keep)
    with envs.SGLANG_OPT_DSV41_CP_LOCAL_COMPRESSOR_PROJECTION.override(True), patch.object(
        backend, "get_parallel", return_value=SimpleNamespace(attn_cp_size=cp_size, attn_cp_rank=rank)
    ), patch.object(backend, "cp_materialize_global_token_order", side_effect=gather):
        backend.DeepseekV4AttnBackend._forward_low_ratio_sources_cp(
            owner, layer=layer, x=x[rank::cp_size], q_lora=None, positions=None,
            forward_batch=batch, run_compressor=True, run_indexer=False)
    capture = owner._low_ratio_compress_torch.call_args
    assert calls == [(local_rows, ratio * 32 if expect_local else 128)]
    if expect_local:
        kv, score = capture.kwargs["projected"]
        torch.testing.assert_close(kv, full_kv, rtol=0.005, atol=0.01)
        if ratio == 2:
            torch.testing.assert_close(score, full_score, rtol=1e-4, atol=1e-4)
    else:
        assert capture.kwargs["projected"] is None
        torch.testing.assert_close(capture.args[1], x)


def test_draft_main_projection_prepack():
    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v4_dspark as draft
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    w = torch.nn.Parameter(torch.randn(128, 256, dtype=torch.bfloat16, device="cuda"), False)
    projection = SimpleNamespace(weight=w, quant_method=UnquantizedLinearMethod())
    owner = SimpleNamespace(stages=[SimpleNamespace(main_proj=projection)])
    pack = draft.DeepseekV4ForCausalLMDSpark._maybe_prepack_bf16_main_proj
    original = w.detach().clone()
    x = torch.randn(8, 256, dtype=w.dtype, device=w.device)
    expected = torch.nn.functional.linear(x, original)
    for enabled, mode, stride in ((False, "prefill", (256,1)), (True, "null", (256,1)),
                                  (True, "prefill", (1,128))):
        with envs.SGLANG_HCU_DSV41_BF16_MAIN_PROJ_PREPACK.override(enabled), patch.object(
            draft, "get_disagg", return_value=SimpleNamespace(disaggregation_mode=mode)
        ):
            pack(owner)
        assert projection.weight is w and w.stride() == stride
    torch.testing.assert_close(w, original, rtol=0, atol=0)
    torch.testing.assert_close(torch.nn.functional.linear(x, w), expected)
