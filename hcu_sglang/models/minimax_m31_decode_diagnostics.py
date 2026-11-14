from contextvars import ContextVar
import logging

from hcu_sglang.layers.moe import deepep_current
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import get_parallel

_logger = logging.getLogger(__name__)
_phase = ContextVar("m31_decode_diagnostic_phase", default=None)
_moe_layer = ContextVar("m31_decode_diagnostic_moe_layer", default=None)
_CALLS_PER_PHASE = 3
_INSTALLED = False


def _emit(event, **fields):
    state = _phase.get()
    if state is None:
        return
    phase, call = state
    _logger.info(
        "M31_DECODE_DIAG rank=%d phase=%s call=%d event=%s %s",
        get_parallel().attn_dp_rank,
        phase,
        call,
        event,
        " ".join(f"{key}={value}" for key, value in fields.items()),
    )


def _run_phase(original, worker, batch, **kwargs):
    phase = "prefill" if batch.forward_mode.is_extend() or batch.is_extend_in_batch else "decode"
    counters = getattr(worker, "_m31_diagnostic_phase_calls", None)
    if counters is None:
        counters = worker._m31_diagnostic_phase_calls = {}
    call = counters[phase] = counters.get(phase, 0) + 1
    if call > _CALLS_PER_PHASE:
        return original(worker, batch, **kwargs)

    context = _phase.set((phase, call))
    try:
        _emit("worker.begin", mode=batch.forward_mode.name, tokens=batch.global_num_tokens,
              width=batch.batch_size(), extend=batch.is_extend_in_batch)
        result = original(worker, batch, **kwargs)
        _emit("worker.end", graph=result.can_run_cuda_graph)
        return result
    finally:
        _phase.reset(context)


def _run_model(original, runner, batch, *args, **kwargs):
    if _phase.get() is None:
        return original(runner, batch, *args, **kwargs)
    _emit("model.begin", role="draft" if runner.is_draft_worker else "target",
          mode=batch.forward_mode.name, width=batch.batch_size,
          rows=None if batch.input_ids is None else batch.input_ids.numel(),
          extend=batch.is_extend_in_batch)
    result = original(runner, batch, *args, **kwargs)
    _emit("model.end", role="draft" if runner.is_draft_worker else "target",
          graph=result.can_run_graph)
    return result


def _run_moe(original, layer, hidden_states, forward_batch):
    if _phase.get() is None:
        return original(layer, hidden_states, forward_batch)
    context = _moe_layer.set(layer.layer_id)
    try:
        return original(layer, hidden_states, forward_batch)
    finally:
        _moe_layer.reset(context)


def _log_collective(original, dispatcher, *args, **kwargs):
    layer = _moe_layer.get()
    if layer is None:
        return original(dispatcher, *args, **kwargs)
    tensor = next(iter(args), None)
    if tensor is None:
        tensor = kwargs.get("hidden_states", kwargs.get("combine_input"))
    if not hasattr(tensor, "shape"):
        tensor = tensor.hidden_states
    event = "deepep." + original.__name__
    _emit(event + ".begin", layer=layer, shape=tuple(tensor.shape),
          mode=dispatcher.deepep_mode.resolve(get_is_extend_in_batch()).name)
    result = original(dispatcher, *args, **kwargs)
    _emit(event + ".end", layer=layer)
    return result


def _log_kv_injection(original, injector, *args, **kwargs):
    if _phase.get() is None:
        return original(injector, *args, **kwargs)
    _emit("draft_kv.begin", shape=tuple(kwargs["target_hidden"].shape))
    result = original(injector, *args, **kwargs)
    _emit("draft_kv.end")
    return result


def register_hooks():
    global _INSTALLED
    if _INSTALLED:
        return

    targets = {
        "speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2.forward_batch_generation":
            _run_phase,
        "model_executor.model_runner.ModelRunner._forward_raw": _run_model,
        "models.minimax_m3.MiniMaxM3MoE.forward_deepep": _run_moe,
        "speculative.dspark_components.dspark_kv_inject.TargetHiddenKvInjector.inject_prefill":
            _log_kv_injection,
    }
    for target, hook in targets.items():
        HookRegistry.register(f"sglang.srt.{target}", hook, HookType.AROUND)

    dispatcher = deepep_current.DeepEPDispatcher
    for method in ("dispatch", "combine"):
        original = getattr(dispatcher, method)

        def wrapped(self, *args, _original=original, **kwargs):
            return _log_collective(_original, self, *args, **kwargs)

        setattr(dispatcher, method, wrapped)
    _INSTALLED = True
