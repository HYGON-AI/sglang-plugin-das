import torch

from sglang.srt.layers import logits_processor as native
from sglang.srt.speculative.spec_info import SpecInputType

from hcu_sglang.prefill_output import LogitsProcessorOutput


def after_metadata(result, cls, forward_batch):
    result.spec_input_type = getattr(forward_batch.spec_info, "spec_input_type", None)
    return result


def around_forward(
    original, self, input_ids, hidden_states, lm_head, logits_metadata,
    aux_hidden_states=None, hidden_states_before_norm=None,
):
    helper = getattr(self, "draft_lm_head_vp", None)
    if helper is None:
        return original(
            self, input_ids, hidden_states, lm_head, logits_metadata,
            aux_hidden_states, hidden_states_before_norm,
        )
    metadata = logits_metadata
    if isinstance(metadata, native.ForwardBatch):
        spec_type = getattr(metadata.spec_info, "spec_input_type", None)
    else:
        spec_type = getattr(metadata, "spec_input_type", None)
    if isinstance(metadata, native.ForwardBatch):
        metadata = native.LogitsMetadata.from_forward_batch(metadata)
    if (
        not metadata.forward_mode.is_decode_or_idle()
        or spec_type != SpecInputType.EAGLE_DRAFT
        or metadata.extend_return_logprob
        or native._autotune_run_lm_head is False
    ):
        return original(
            self, input_ids, hidden_states, lm_head, logits_metadata,
            aux_hidden_states, hidden_states_before_norm,
        )
    hidden_states, hidden_states_before_norm = native.layernorm_sp.maybe_exit_gather(
        hidden_states=hidden_states,
        hidden_states_before_norm=hidden_states_before_norm,
        input_ids=input_ids,
        forward_mode=metadata.forward_mode,
    )
    (
        pruned_states, pruned_states_before_norm, aux_pruned_states,
        sample_indices, _, _,
    ) = self._get_pruned_states(
        hidden_states, hidden_states_before_norm, aux_hidden_states, metadata,
    )
    hidden_states_to_store = self._get_hidden_states_to_store(
        hidden_states, hidden_states_before_norm, aux_hidden_states,
        pruned_states, pruned_states_before_norm, aux_pruned_states,
        sample_indices, metadata,
    )
    top1_scores, top1_token_ids = helper.project_top1(
        pruned_states, lm_head.weight,
        logit_scale=self.logit_scale,
        final_logit_softcapping=self.final_logit_softcapping,
    )
    top1_token_ids = top1_token_ids.unsqueeze(-1)
    return LogitsProcessorOutput(
        next_token_logits=top1_scores.unsqueeze(-1),
        hidden_states=hidden_states_to_store,
        draft_top1_token_ids=top1_token_ids,
        draft_top1_probs=torch.ones_like(top1_token_ids, dtype=torch.float32),
        mm_input_embeds=metadata.mm_input_embeds,
    )
