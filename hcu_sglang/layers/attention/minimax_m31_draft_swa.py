import os

import torch

from hcu_sglang.utils.detection import is_hcu_native_fp8_supported

_WINDOW_LEFT = 4095


class _DraftSwaForward:
    def __init__(self, fallback, page_size):
        self.fallback = fallback
        self.page_size = page_size
        self.kernel = None

    def __call__(
        self, q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices, custom_mask, is_causal, mask_indptr,
        max_len_extend, k_scale, v_scale, sm_scale, *, logit_cap=0,
        sliding_window_size=-1, sinks=None, window_kv_offsets=None,
        xai_temperature_len=-1, max_bs=None,
    ):
        bs = qo_indptr.numel() - 1
        rows = bs * 7
        if (
            sliding_window_size != _WINDOW_LEFT
            or custom_mask is not None
            or sinks is not None
            or logit_cap
            or (xai_temperature_len is not None and xai_temperature_len > 0)
            or max_len_extend != 7
            or bs <= 0
            or kv_indptr.shape != qo_indptr.shape
            or not kv_indices.numel()
            or not q_extend.is_cuda
            or q_extend.dtype != torch.bfloat16
            or q_extend.shape != o_extend.shape
            or q_extend.shape != (rows, 64, 128)
            or any(t.shape != (rows, 4, 128) or t.dtype != q_extend.dtype
                   for t in (k_extend, v_extend))
            or k_buffer.shape != v_buffer.shape
            or k_buffer.dtype != v_buffer.dtype
            or k_buffer.dtype not in (torch.float8_e4m3fn, torch.bfloat16)
            or any(t.ndim != 3 or t.shape[1:] != (4, 128) or t.stride(-1) != 1
                   for t in (k_buffer, v_buffer))
            or any(t.stride(-1) != 1 for t in (q_extend, k_extend, v_extend, o_extend))
        ):
            if self.fallback is None:
                return False
            return self.fallback(
                q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
                qo_indptr, kv_indptr, kv_indices, custom_mask, is_causal, mask_indptr,
                max_len_extend, k_scale, v_scale, sm_scale, logit_cap=logit_cap,
                sliding_window_size=sliding_window_size, sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=xai_temperature_len, max_bs=max_bs,
            )

        if self.kernel is None:
            from ...kernels.minimax_draft_swa import VerifyMLA

            self.kernel = VerifyMLA(
                max(bs, max_bs or 0), 64, 128, 128, 7,
                device=q_extend.device, kv_group_num=16,
            )
        self.kernel.grow_buffers(bs)
        self.kernel(
            q_extend, k_extend, v_extend, k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices, sm_scale,
            o_out=o_extend, k_scale=k_scale, v_scale=v_scale,
            is_causal=is_causal, page_size=self.page_size,
            sliding_window_size=sliding_window_size,
        )
        return True


def _enable_draft_swa(_result, backend, model_runner, *args, **kwargs):
    arches = model_runner.model_config.hf_config.architectures or ()
    if (
        os.getenv("SGLANG_MINIMAX_M31_DRAFT_SPLITKV", "1") != "1"
        or not model_runner.is_draft_worker
        or "DSparkMiniMaxDraftModel" not in arches
        or not model_runner.spec_algorithm.is_dspark()
        or backend.topk != 1
        or backend.use_mla
        or backend.dcp_size != 1
        or not is_hcu_native_fp8_supported()
    ):
        return

    if backend.use_verify_shared_kv:
        fallback = backend.verify_shared_kv_fwd
    elif backend.use_verify_splitkv:
        fallback = backend.verify_splitkv_fwd
    else:
        fallback = None
    backend.verify_shared_kv_fwd = torch.compiler.disable(
        _DraftSwaForward(fallback, backend.page_size)
    )
    backend.use_verify_shared_kv = True


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.triton_backend.TritonAttnBackend.__init__",
        _enable_draft_swa,
        HookType.AFTER,
    )
