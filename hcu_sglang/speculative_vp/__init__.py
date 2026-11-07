from hcu_sglang.speculative_vp.config import vp_size
from hcu_sglang.utils.detection import is_hcu


def around_init_lm_head(original, self):
    result = original(self)
    self.draft_lm_head_vp = None
    size = vp_size()
    if not is_hcu() or size == 1:
        return result
    from sglang.srt.runtime_context import get_parallel
    from hcu_sglang.speculative_vp.core import DraftLMHeadVocabParallelTop1

    model = self.draft_runner.model
    processor = getattr(model, "logits_processor", None)
    if processor is None:
        raise RuntimeError("Draft LM-head VP requires a logits_processor")
    parallel = get_parallel()
    self.draft_lm_head_vp = DraftLMHeadVocabParallelTop1(
        full_weight=model.lm_head.weight,
        vocab_size=self.draft_runner.model_config.vocab_size,
        vp_size=size,
        max_rows_per_rank=self.draft_runner.req_to_token_pool.size,
        local_world_size=parallel.tp_size // parallel.nnodes,
    )
    processor.draft_lm_head_vp = self.draft_lm_head_vp
    return result


def around_draft_forward(original, self, forward_batch):
    if getattr(self, "draft_lm_head_vp", None) is None:
        return original(self, forward_batch)
    from hcu_sglang.speculative_vp.worker import draft_forward

    return draft_forward(self, forward_batch)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.speculative_vp import config, logits

    for target, hook, kind in (
        ("sglang.srt.server_args.ServerArgs.add_cli_args", config.after_add_cli_args, HookType.AFTER),
        ("sglang.srt.server_args.ServerArgs.from_cli_args", config.before_from_cli_args, HookType.BEFORE),
        ("sglang.srt.layers.logits_processor.LogitsMetadata.from_forward_batch", logits.after_metadata, HookType.AFTER),
        ("sglang.srt.layers.logits_processor.LogitsProcessor.forward", logits.around_forward, HookType.AROUND),
        ("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.init_lm_head", around_init_lm_head, HookType.AROUND),
        ("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.draft_forward", around_draft_forward, HookType.AROUND),
    ):
        HookRegistry.register(target, hook, kind)
