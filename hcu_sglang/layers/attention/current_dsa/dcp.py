# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple


def _validate_dsa_dcp_launch(
    *,
    dcp_enabled: bool,
    dcp_size: int,
    is_hcu_platform: bool,
    device_capability: tuple[int, int],
    dsa_prefill_impl: str,
    dsa_decode_impl: str,
    dsa_kv_cache_store_fp8: bool,
    page_size: int,
    enable_prefill_cp: bool,
    enable_hisparse: bool,
    enable_hierarchical_cache: bool,
    enable_symm_mem: bool,
    speculative_algorithm: Optional[str],
    fused_topk_enabled: bool,
    dcp_comm_backend: str,
    speculative_num_steps: Optional[int],
    speculative_eagle_topk: Optional[int],
    speculative_num_draft_tokens: Optional[int],
    pp_size: int,
    attn_cp_size: int,
    enable_dp_attention: bool,
    attn_tp_size: int,
    dcp_group_ranks: Tuple[int, ...],
    attn_tp_group_ranks: Tuple[int, ...],
    index_share_for_mtp_iteration: bool,
    decode_cuda_graph_backend: str,
    decode_cuda_graph_max_bs: Optional[int],
    allow_index_share_for_mtp_iteration: bool = False,
) -> None:
    if not dcp_enabled:
        return
    if dcp_size not in (2, 4, 8):
        raise ValueError(f"DSA DCP supports dcp size 2, 4, or 8; got {dcp_size}.")
    supported_capability = (
        device_capability == (9, 3) if is_hcu_platform else device_capability == (9, 0)
    )
    if not supported_capability:
        platform_name = "HCU" if is_hcu_platform else "CUDA"
        raise ValueError(
            f"DSA DCP first-phase support requires Hopper SM90 or HCU BW1000; got {platform_name} capability {device_capability}."
        )
    if (dsa_prefill_impl, dsa_decode_impl) != ("flashmla_kv", "flashmla_kv"):
        raise ValueError(
            f"DSA DCP first-phase support requires flashmla_kv for both prefill and decode; got {dsa_prefill_impl}/{dsa_decode_impl}."
        )
    if not dsa_kv_cache_store_fp8:
        raise ValueError("DSA DCP first-phase support requires an FP8 KV cache.")
    if page_size != 64:
        raise ValueError(
            f"DSA DCP first-phase support requires page size 64; got {page_size}."
        )
    if enable_prefill_cp:
        raise ValueError("DSA DCP does not support prefill CP in the first phase.")
    if enable_hisparse:
        raise ValueError("DSA DCP does not support HiSparse in the first phase.")
    if enable_hierarchical_cache:
        raise ValueError("DSA DCP does not support HiCache in the first phase.")
    if enable_symm_mem:
        raise ValueError(
            "DSA DCP does not support symmetric memory in the first phase."
        )
    if speculative_algorithm is not None:
        if not is_hcu_platform:
            raise ValueError("DSA DCP speculative decoding is only supported on HCU.")
        if speculative_algorithm != "EAGLE":
            raise ValueError(
                f"DSA DCP speculative decoding only supports EAGLE; got {speculative_algorithm}."
            )
        speculative_parameters = {
            "num_steps": speculative_num_steps,
            "eagle_topk": speculative_eagle_topk,
            "num_draft_tokens": speculative_num_draft_tokens,
        }
        invalid_parameters = {
            name: value
            for (name, value) in speculative_parameters.items()
            if value is None or value <= 0
        }
        if invalid_parameters:
            raise ValueError(
                f"DSA DCP EAGLE requires positive speculative parameters; got {invalid_parameters}."
            )
        if attn_tp_size != dcp_size:
            raise ValueError(
                f"DSA DCP EAGLE requires attn_tp_size == dcp_size; got dcp_size={dcp_size}, attn_tp_size={attn_tp_size}."
            )
        if tuple(dcp_group_ranks) != tuple(attn_tp_group_ranks):
            raise ValueError(
                f"DSA DCP EAGLE requires identical DCP and attention TP group ranks; got dcp={tuple(dcp_group_ranks)}, attn_tp={tuple(attn_tp_group_ranks)}."
            )
        if pp_size != 1 or attn_cp_size != 1 or (not enable_dp_attention):
            raise ValueError(
                f"DSA DCP EAGLE requires PP1, attnCP1, and DP attention; got pp_size={pp_size}, attn_cp_size={attn_cp_size}, enable_dp_attention={enable_dp_attention}."
            )
        if index_share_for_mtp_iteration and not (is_hcu_platform and allow_index_share_for_mtp_iteration):
            raise ValueError("DSA DCP MTP index sharing is unsupported for this model/cache configuration.")
        if decode_cuda_graph_backend not in ("disabled", "full"):
            raise ValueError(
                f"DSA DCP EAGLE only supports disabled or full decode CUDA Graph; got {decode_cuda_graph_backend}."
            )
        if decode_cuda_graph_backend == "full" and (
            decode_cuda_graph_max_bs is None or decode_cuda_graph_max_bs <= 0
        ):
            raise ValueError(
                f"DSA DCP EAGLE CUDA Graph requires a positive max BS; got {decode_cuda_graph_max_bs}."
            )
    if fused_topk_enabled:
        raise ValueError(
            "DSA DCP does not support fused DSA top-k in the first phase; set SGLANG_DSA_FUSE_TOPK=false."
        )
    if dcp_comm_backend != "ag_rs":
        raise ValueError(
            f"DSA DCP first-phase support requires the ag_rs communication backend; got {dcp_comm_backend}."
        )


def around_mla_dcp_phase(original, attn, forward_batch):
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu() or not attn.use_dsa:
        return original(attn, forward_batch)
    if attn.dcp_size == 1:
        return False
    from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import (
        effective_forward_mode,
    )

    mode = effective_forward_mode(forward_batch)
    return (
        mode.is_decode()
        or mode.is_target_verify()
        or mode.is_draft_extend_v2()
        or mode.is_extend_without_speculative()
    )
