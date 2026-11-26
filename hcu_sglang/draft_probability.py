from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_spec

from hcu_sglang.utils.detection import is_hcu


def uses_native_draft_probabilities(worker):
    if (
        not envs.SGLANG_HCU_GLM5_FLASH_DRAFT_PROBABILITIES.get()
        or worker.topk != 1
        or getattr(worker, "draft_lm_head_vp", None) is not None
        or get_spec().speculative_use_rejection_sampling
    ):
        return False
    config = worker.target_worker.model_runner.model_config.hf_config
    config = getattr(config, "text_config", None) or config
    return (
        getattr(config, "model_type", None) == "glm5_next_text"
        and getattr(config, "hidden_size", None) == 4096
        and getattr(config, "n_routed_experts", None) == 288
        and getattr(config, "index_kpool", None) == 4
        and getattr(config, "qk_rope_head_dim", None) == 0
    )


def _hook(fn):
    def around(original, worker, *args, **kwargs):
        if uses_native_draft_probabilities(worker):
            return fn(worker, *args, **kwargs)
        return original(worker, *args, **kwargs)

    return around


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang import draft_probability_source

    for name in ("draft_forward", "_draft_extend_for_decode"):
        fn = getattr(draft_probability_source, name)
        HookRegistry.register(
            f"sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.{name}",
            _hook(fn),
            HookType.AROUND,
        )
