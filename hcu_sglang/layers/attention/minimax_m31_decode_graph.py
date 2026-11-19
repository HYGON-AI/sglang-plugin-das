import logging
import os
from dataclasses import dataclass

import torch

from . import minimax_m31_flashmla as native

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Stage1GraphVariants:
    cap: int
    verify_width: int

    @property
    def capture_labels(self) -> tuple[str, str]:
        return f"msa_stage1_cap_{self.cap}", "msa_stage1_full"

    def select(self, forward_batch) -> str:
        compact, full = self.capture_labels
        lengths = forward_batch.seq_lens_cpu
        if (
            forward_batch.batch_size <= 0
            or lengths is None
            or lengths.device.type != "cpu"
            or lengths.numel() < forward_batch.batch_size
        ):
            return full
        longest = int(lengths.max())
        return compact if 0 < longest <= self.cap - self.verify_width else full


def _stage1_graph_variants(original, model_runner, capture_forward_mode, captured_req_width=0):
    setting = os.environ.get("M31_STAGE1_GRAPH_MAX_KV_LEN")
    if not setting or not torch.version.hip or model_runner.is_draft_worker:
        return original(model_runner, capture_forward_mode, captured_req_width)

    from sglang.srt.configs.model_config import is_minimax_sparse
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_memory

    model = model_runner.model_config
    if (
        model.quantization != "slimquant_w4a8_marlin"
        or not is_minimax_sparse(model.hf_config)
        or get_memory().enable_hisparse
        or capture_forward_mode not in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY)
    ):
        return original(model_runner, capture_forward_mode, captured_req_width)

    sparse = native.sparse_backend_for_mode(model_runner.attn_backend, capture_forward_mode)
    if (
        getattr(sparse, "block_size_k", None) != 128
        or getattr(sparse, "page_size", None) != 128
        or not native.flash_mla_gfx938_available()
        or not native.flash_mla_stage1_graph_available()
    ):
        return original(model_runner, capture_forward_mode, captured_req_width)

    cap = int(setting)
    if cap <= 0 or cap % sparse.page_size:
        raise ValueError("M31_STAGE1_GRAPH_MAX_KV_LEN must be a positive multiple of 128")
    if cap >= sparse.max_context_len:
        return original(model_runner, capture_forward_mode, captured_req_width)

    verify_width = (
        captured_req_width if capture_forward_mode == ForwardMode.TARGET_VERIFY else 0
    )
    LOG.info("MiniMax Stage 1 decode graphs enabled with compact KV limit %d", cap)
    return _Stage1GraphVariants(cap, verify_width)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.graph_variants.create_dsv41_candidate_graph_variants",
        _stage1_graph_variants,
        HookType.AROUND,
    )
