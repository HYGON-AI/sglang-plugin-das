import logging
import os
import time
from functools import cache, wraps
from types import FunctionType, SimpleNamespace

from hcu_sglang.utils.detection import is_hcu

_BASE_BACKENDS = ("none", "megamoe", "mori")
_MINIMAX_ARCHES = {"MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3SparseForCausalLM"}
_INT8_KV_ONLY = os.getenv("SGLANG_MINIMAX_DSPARK_INT8_KV_ONLY", "1") == "1"


@cache
def _with_deepep(original):
    code = original.__code__
    if code.co_consts.count(_BASE_BACKENDS) != 1:
        raise RuntimeError("v0.5.23 DSpark DP MoE backend check has changed")
    constants = tuple(
        (*value, "deepep") if value == _BASE_BACKENDS else value
        for value in code.co_consts
    )
    return FunctionType(
        code.replace(co_consts=constants),
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )


def _resolve_dspark(original, server_args):
    from sglang.srt.arg_groups.overrides import model_config_of, resolving_view

    cfg = resolving_view(server_args)
    if (
        cfg.moe_a2a_backend != "deepep"
        or cfg.attn_dp_size <= 1
        or cfg.quantization != "slimquant_w4a8_marlin"
    ):
        return original(server_args)

    import torch

    arch = model_config_of(server_args).hf_config.architectures[0]
    if arch not in _MINIMAX_ARCHES or not torch.version.hip:
        return original(server_args)
    return _with_deepep(original)(server_args)


def _minimax_hip_dspark():
    import torch
    from sglang.srt.runtime_context import get_model, get_spec

    return (
        torch.version.hip is not None
        and get_spec().speculative_algorithm == "DSPARK"
        and get_model().quantization == "slimquant_w4a8_marlin"
    )


@cache
def _with_native_hip(original):
    return FunctionType(
        original.__code__,
        {**original.__globals__, "_is_cuda": False, "_is_hip": True},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )


def _hip_future_map(original, relay, *args, **kwargs):
    # The private D2H copy stream is CUDA-only in the community relay.
    # Use its native HIP publish fence and synchronous CPU length read.
    if _minimax_hip_dspark():
        original = _with_native_hip(original)
    return original(relay, *args, **kwargs)


def _hip_resolve_seq_lens(original, relay, batch, stream_ordered=False):
    if _minimax_hip_dspark():
        from .minimax_m31_seq_lens import defer_resolution

        if defer_resolution(relay, batch, stream_ordered):
            return None
        original = _with_native_hip(original)
    return original(relay, batch, stream_ordered=stream_ordered)


@cache
def _with_hip_scheduler(original):
    return FunctionType(
        original.__code__,
        {**original.__globals__, "is_cuda": lambda: False, "_is_hip": True},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )


def _hip_scheduler_loop(original, scheduler):
    # Keep native HIP's stream alias check and its opt-in WAR barrier.
    # CUDA semantics would otherwise add an unconditional wait from the
    # scheduler stream to the previous forward, including on idle DP ranks.
    if _minimax_hip_dspark():
        original = _with_hip_scheduler(original)
    return original(scheduler)


def _scope_verify_hidden_commit(_result, worker, *args, **kwargs):
    from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

    if not (
        _minimax_hip_dspark()
        and worker._hosts_draft
        and get_parallel().attn_dp_enabled
        and get_disagg().disaggregation_mode in ("null", "decode")
        and not get_memory().enable_hisparse
        and worker.model_runner.model_config.hf_config.architectures[0]
        in _MINIMAX_ARCHES
    ):
        return

    executor = worker._verify_executor
    commit_hidden = executor.commit_hidden

    @wraps(commit_hidden)
    def commit_in_draft_scope(**commit_kwargs):
        with worker._draft_context():
            return commit_hidden(**commit_kwargs)

    executor.commit_hidden = commit_in_draft_scope


def _draft_int8_kv_only(original, attention, hidden_states):
    if not _INT8_KV_ONLY or not is_hcu():
        return original(attention, hidden_states)

    import torch

    projection = attention.qkv_proj
    weight = getattr(projection, "weight", None)
    if weight is None or weight.dtype != torch.int8:
        return original(attention, hidden_states)

    from compressed_tensors.quantization import QuantizationStrategy
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

    scheme = getattr(projection, "scheme", None)
    scales = getattr(projection, "weight_scale", None)
    output_size = attention.q_size + 2 * attention.kv_size
    if not (
        type(scheme) is CompressedTensorsW8A8Int8
        and scheme.strategy == QuantizationStrategy.CHANNEL
        and not scheme.is_static_input_scheme
        and scheme.input_symmetric
        and scales is not None
        and scales.shape == (output_size, 1)
        and scales.dtype == torch.float32
        and not projection.skip_bias_add
    ):
        return original(attention, hidden_states)

    rows = slice(attention.q_size, output_size)
    strategy = getattr(scheme, "w8a8_strategy", None)
    hidden_size = attention.hidden_size
    if strategy is None and (
        weight.shape == (hidden_size, output_size)
        and weight.stride() == (1, hidden_size)
    ):
        kv_weight = weight[:, rows]
    elif strategy == 3 and (
        weight.shape == (output_size, hidden_size) and weight.is_contiguous()
    ):
        kv_weight = weight[rows]
    else:
        return original(attention, hidden_states)
    kv_projection = SimpleNamespace(weight=kv_weight, weight_scale=scales[rows])
    bias = projection.bias
    kv = scheme.apply_weights(
        kv_projection, hidden_states, bias=bias[rows] if bias is not None else None
    )
    return kv.split(attention.kv_size, dim=-1)


def _warmup_accept_output(_result, worker):
    if not _minimax_hip_dspark() or not worker._hosts_draft:
        return
    from sglang.srt.runtime_context import get_disagg
    if (
        get_disagg().disaggregation_mode != "decode"
        or worker.verify_num_draft_tokens != worker.gamma + 1
        or not worker.draft_model_runner.decode_cuda_graph_captured
    ):
        return

    import torch
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens

    start = time.monotonic()
    max_bs = max(worker.draft_model_runner.decode_cuda_graph_runner.capture_bs)
    with torch.inference_mode():
        draft = torch.arange(max_bs * worker.gamma, device=worker.device).view(max_bs, worker.gamma)
        bonus = torch.full((max_bs,), 41, dtype=torch.int64, device=worker.device)
        correct_len = torch.arange(max_bs, device=worker.device) % worker.verify_num_draft_tokens
        for bs in range(1, max_bs + 1):
            kwargs = dict(
                draft_tokens=draft[:bs], correct_len=correct_len[:bs], bonus=bonus[:bs],
                verify_num_draft_tokens=worker.verify_num_draft_tokens, gamma=worker.gamma,
            )
            if not torch.equal(BuildOutTokens.triton(**kwargs), BuildOutTokens.torch(**kwargs)):
                raise RuntimeError(f"DSpark accept-output warmup mismatch at batch {bs}")
    logging.getLogger(__name__).info(
        "DSpark eager accept-output warmup finished for B=1..%d, T=%d in %.2f ms",
        max_bs, worker.verify_num_draft_tokens, (time.monotonic() - start) * 1000,
    )


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.arg_groups.speculative_hook._handle_dspark",
        _resolve_dspark,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.managers.overlap_utils.FutureMap.__init__",
        _hip_future_map,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.overlap_utils.FutureMap.resolve_seq_lens_cpu",
        _hip_resolve_seq_lens,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.run_event_loop",
        _hip_scheduler_loop,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2.__init__",
        _scope_verify_hidden_commit,
        HookType.AFTER,
    )

    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2.init_cuda_graphs",
        _warmup_accept_output,
        HookType.AFTER,
    )

    HookRegistry.register(
        "sglang.srt.models.dspark_minimax.DSparkMiniMaxAttention.kv_proj_only",
        _draft_int8_kv_only,
        HookType.AROUND,
    )
