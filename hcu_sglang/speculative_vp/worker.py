from __future__ import annotations

import contextlib
from typing import List

import torch
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftInput, IndexTopKShareState, ForwardContext, forward_context,
    get_spec, maybe_detect_nan, maybe_detect_inf, maybe_detect_oob,
    per_step_draft_out_cache_loc, select_top_k_tokens, sample_draft_proposal,
    draft_topk1_postprocess, renorm_draft_probs, fast_topk,
    organize_draft_results, _is_cuda, _is_hip,
)


def draft_forward(self, forward_batch):
    # Parse args
    spec_info: EagleDraftInput = forward_batch.spec_info
    if forward_batch.forward_mode.is_idle():
        return self._draft_forward_idle(forward_batch, spec_info)

    out_cache_loc = forward_batch.out_cache_loc
    topk_p, topk_index, hidden_states = (
        spec_info.topk_p,
        spec_info.topk_index,
        spec_info.hidden_states,
    )

    maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

    if self.hot_token_id is not None:
        topk_index = self.hot_token_id[topk_index]

    out_cache_loc = per_step_draft_out_cache_loc(
        out_cache_loc,
        forward_batch.batch_size,
        self.topk,
        self.speculative_num_steps,
    )

    # Return values
    score_list: List[torch.Tensor] = []
    token_list: List[torch.Tensor] = []
    parents_list: List[torch.Tensor] = []
    # Verify commits argmax for an all-greedy batch and never reads the
    # draft distribution, so skip building it.
    needs_draft_probs = (
        get_spec().speculative_use_rejection_sampling
        and not forward_batch.sampling_info.is_all_greedy
    )
    if needs_draft_probs:
        draft_probs_list: List[torch.Tensor] = [spec_info.draft_probs]

    topk1_chain_fits = (
        self.topk == 1
        and topk_index.shape[0] <= self._topk1_parents_prealloc.shape[0]
    )
    # Materialize the chain directly only when the CUDA kernel can write
    # every subsequent column. Other topk=1 paths retain the token list and
    # assemble it with one final cat instead of launching a copy per step.
    draft_tokens_topk1 = None
    if (
        topk1_chain_fits
        and _is_cuda
        and self.hot_token_id is None
        and not needs_draft_probs
        and getattr(self, "draft_lm_head_vp", None) is None
    ):
        draft_tokens_topk1 = torch.empty(
            (topk_index.shape[0], self.speculative_num_steps),
            dtype=topk_index.dtype,
            device=topk_index.device,
        )
        draft_tokens_topk1[:, :1].copy_(topk_index)

    # Forward multiple steps
    scores = None
    with IndexTopKShareState.mtp_iteration(
        forward_batch,
        enabled=self.index_share_for_mtp_iteration,
        keep_carry_seed=self.seed_dsa_topk_from_draft_extend,
    ):
        for i in range(self.speculative_num_steps):
            if draft_tokens_topk1 is not None:
                input_ids = topk_index.flatten()
            else:
                input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                    i, topk_p, topk_index, hidden_states, scores, self.topk
                )
                score_list.append(tree_info[0])
                token_list.append(tree_info[1])
                parents_list.append(tree_info[2])

            if i == self.speculative_num_steps - 1:
                break

            forward_batch.input_ids = input_ids
            # Qwen3-MoE MTP uses a fused RoPE + KV-store path whose cache_loc
            # argument must be contiguous.
            if (
                self.draft_runner.model_config.hf_config.architectures[0]
                == "Qwen3MoeForCausalLMMTP"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            spec_info.hidden_states = hidden_states

            canary_index_ctx = (
                c.with_active_single_forward_manager(i)
                if (c := self.draft_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with (
                forward_context(
                    ForwardContext(
                        attn_backend=self.draft_attn_backend.attn_backends[i]
                    )
                ),
                canary_index_ctx,
            ):
                logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output
            maybe_detect_nan(
                logits_output.next_token_logits, f"draft_forward step {i}"
            )
            maybe_detect_inf(
                logits_output.next_token_logits, f"draft_forward step {i}"
            )
            vp_top1_token_ids = getattr(logits_output, "draft_top1_token_ids", None)
            if vp_top1_token_ids is not None:
                topk_p = logits_output.draft_top1_probs
                topk_index = vp_top1_token_ids
                forward_batch.positions.add_(1)
            elif needs_draft_probs:
                probs, topk_p, topk_index = sample_draft_proposal(
                    logits_output.next_token_logits,
                    forward_batch.sampling_info.temperatures,
                    forward_batch.sampling_info.top_ks,
                )
                draft_probs_list.append(probs)
                forward_batch.positions.add_(1)
            elif self.topk == 1:
                if _is_cuda or _is_hip:
                    topk_p, topk_index = draft_topk1_postprocess(
                        logits_output.next_token_logits,
                        forward_batch.positions,
                        draft_tokens_topk1,
                        i + 1,
                    )
                else:
                    topk_index = torch.argmax(
                        logits_output.next_token_logits, dim=-1, keepdim=True
                    )
                    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
                    forward_batch.positions.add_(1)
            else:
                probs = renorm_draft_probs(
                    logits_output.next_token_logits,
                    forward_batch.sampling_info,
                    needs_draft_probs,
                )
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                forward_batch.positions.add_(1)
            if self.draft_runner.model_config.model_is_mrope:
                forward_batch.mrope_positions.add_(1)
            vocab_size = (
                self.draft_runner.model_config.vocab_size
                if vp_top1_token_ids is not None
                else logits_output.next_token_logits.shape[-1]
            )
            maybe_detect_oob(
                topk_index,
                0,
                vocab_size,
                f"draft_forward step {i}: topk_index OOB vs vocab_size={vocab_size}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

    draft_probs = (
        torch.stack(draft_probs_list, dim=1) if needs_draft_probs else None
    )

    # Organize the results
    if draft_tokens_topk1 is not None:
        bs = draft_tokens_topk1.shape[0]
        top_scores_index = self._topk1_score_indices_prealloc[:bs]
        parent_list = self._topk1_parents_prealloc[:bs]
        return parent_list, top_scores_index, draft_tokens_topk1, draft_probs

    if topk1_chain_fits:
        bs = token_list[0].shape[0]
        draft_tokens = torch.cat(token_list, dim=1)
        top_scores_index = self._topk1_score_indices_prealloc[:bs]
        parent_list = self._topk1_parents_prealloc[:bs]
        return parent_list, top_scores_index, draft_tokens, draft_probs

    parent_list, top_scores_index, draft_tokens = organize_draft_results(
        score_list, token_list, parents_list, self.speculative_num_draft_tokens
    )

    return parent_list, top_scores_index, draft_tokens, draft_probs
