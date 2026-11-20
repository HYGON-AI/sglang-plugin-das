import os
from collections.abc import Mapping


_PACKED_EXPERT_BUFFERS = {
    "w13_weight_deepgemm",
    "w2_weight_deepgemm",
    "w13_weight_deepgemm_masked",
    "w2_weight_deepgemm_masked",
}


class _RoutedWeights(Mapping):
    def __init__(self, layers):
        self._layers = layers

    def __getitem__(self, layer_id):
        from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert

        moe = self._layers[layer_id]
        experts = moe.experts
        candidates = list(experts.named_parameters())
        candidates.extend(
            (name, tensor)
            for name, tensor in experts.named_buffers()
            if name in _PACKED_EXPERT_BUFFERS
        )
        num_local = experts.num_local_experts
        num_routed = num_local - moe.num_fused_shared_experts
        weights = []
        seen = set()
        for name, tensor in candidates:
            if name == "correction_bias" or not filter_moe_weight_param_global_expert(
                name, tensor, num_local
            ):
                continue
            view = tensor.detach()[:num_routed]
            alias = (view.device, view.dtype, view.data_ptr(), view.shape, view.stride())
            if alias not in seen:
                seen.add(alias)
                weights.append(view)
        return weights

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)


def _attach_routed_weights(_result, model, *args, **kwargs):
    from sglang.srt.models.minimax_m3 import MiniMaxM3MoE

    if hasattr(model, "routed_experts_weights_of_layer"):
        return

    core = model.model
    layers = {}
    if core is not None:
        for layer_id in range(core.start_layer, core.end_layer):
            moe = core.layers[layer_id].mlp
            if isinstance(moe, MiniMaxM3MoE):
                layers[layer_id] = moe
    model.routed_experts_weights_of_layer = _RoutedWeights(layers)


def _capture_forward_metadata(_result, gatherer, forward_batch):
    global_tokens = forward_batch.original_global_num_tokens_cpu
    if global_tokens is None:
        global_tokens = forward_batch.global_num_tokens_cpu
    gatherer._m31_forward_metadata = {
        "forward_mode": forward_batch.forward_mode.name,
        "batch_size": int(forward_batch.batch_size),
        "num_tokens": forward_batch.input_ids.numel() if forward_batch.input_ids is not None else 0,
        "is_extend_in_batch": bool(forward_batch.is_extend_in_batch),
        "global_num_tokens_cpu": (
            [int(tokens) for tokens in global_tokens] if global_tokens is not None else None
        ),
    }


def _collect_forward_metadata(result, gatherer):
    metadata = getattr(gatherer, "_m31_forward_metadata", None)
    if metadata is not None:
        result["forward_metadata"] = metadata
    return result


def _record_prefill(_original, forward_batch):
    from sglang.srt.eplb.expert_distribution import _is_model_capture_mode

    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return (
        not _is_model_capture_mode()
        and forward_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.SPLIT_PREFILL)
        and int(forward_batch.extend_num_tokens or 0) > 0
    )


def _advance_prefill_counter(_original, forward_batch):
    from sglang.srt.eplb.expert_distribution import _is_model_capture_mode

    return not _is_model_capture_mode() and forward_batch.is_extend_in_batch


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target in (
        "minimax_m3.MiniMaxM3SparseForCausalLM",
        "minimax_m3_vl.MiniMaxM3SparseForConditionalGeneration",
    ):
        HookRegistry.register(
            f"sglang.srt.models.{target}.__init__",
            _attach_routed_weights,
            HookType.AFTER,
        )

    recorder = "sglang.srt.eplb.expert_distribution"
    gatherer = f"{recorder}._LayerBasedGpuSinglePassGatherer"
    HookRegistry.register(
        f"{gatherer}.on_forward_pass_start", _capture_forward_metadata, HookType.AFTER
    )
    HookRegistry.register(f"{gatherer}.collect", _collect_forward_metadata, HookType.AFTER)
    if os.getenv("SGLANG_MINIMAX_EPLB_PREFILL_ONLY", "0") == "1":
        HookRegistry.register(
            f"{recorder}.should_record_forward_pass", _record_prefill, HookType.AROUND
        )
        HookRegistry.register(
            f"{recorder}.should_advance_eplb_counter", _advance_prefill_counter, HookType.AROUND
        )
