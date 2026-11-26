from __future__ import annotations

from sglang.srt.speculative import eagle_worker_v2 as N


def draft_forward(self, forward_batch: N.ForwardBatch):
    # Parse args
    spec_info: N.EagleDraftInput = forward_batch.spec_info
    if forward_batch.forward_mode.is_idle():
        return self._draft_forward_idle(forward_batch, spec_info)

    out_cache_loc = forward_batch.out_cache_loc
    topk_p, topk_index, hidden_states = (
        spec_info.topk_p,
        spec_info.topk_index,
        spec_info.hidden_states,
    )

    N.maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

    if self.hot_token_id is not None:
        topk_index = self.hot_token_id[topk_index]

    out_cache_loc = N.per_step_draft_out_cache_loc(
        out_cache_loc,
        forward_batch.batch_size,
        self.topk,
        self.speculative_num_steps,
    )

    # Return values
    score_list: N.List[N.torch.Tensor] = []
    token_list: N.List[N.torch.Tensor] = []
    parents_list: N.List[N.torch.Tensor] = []
    # Verify commits argmax for an all-greedy batch and never reads the
    # draft distribution, so skip building it.
    needs_draft_probs = (
        N.get_spec().speculative_use_rejection_sampling
        and not forward_batch.sampling_info.is_all_greedy
    )
    if needs_draft_probs:
        draft_probs_list: N.List[N.torch.Tensor] = [spec_info.draft_probs]

    topk1_chain_fits = (
        self.topk == 1
        and topk_index.shape[0] <= self._topk1_parents_prealloc.shape[0]
    )
    # Forward multiple steps
    scores = None
    with N.IndexTopKShareState.mtp_iteration(
        forward_batch,
        enabled=self.index_share_for_mtp_iteration,
        keep_carry_seed=self.seed_dsa_topk_from_draft_extend,
    ):
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = N.select_top_k_tokens(
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
                else N.contextlib.nullcontext()
            )
            with (
                N.forward_context(
                    N.ForwardContext(
                        attn_backend=self.draft_attn_backend.attn_backends[i]
                    )
                ),
                canary_index_ctx,
            ):
                logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output
            N.maybe_detect_nan(
                logits_output.next_token_logits, f"draft_forward step {i}"
            )
            N.maybe_detect_inf(
                logits_output.next_token_logits, f"draft_forward step {i}"
            )
            if needs_draft_probs:
                probs, topk_p, topk_index = N.sample_draft_proposal(
                    logits_output.next_token_logits,
                    forward_batch.sampling_info.temperatures,
                    forward_batch.sampling_info.top_ks,
                )
                draft_probs_list.append(probs)
                forward_batch.positions.add_(1)
            else:
                probs = N.renorm_draft_probs(
                    logits_output.next_token_logits,
                    forward_batch.sampling_info,
                    needs_draft_probs,
                )
                topk_p, topk_index = N.fast_topk(probs, self.topk, dim=-1)
                forward_batch.positions.add_(1)
            if self.draft_runner.model_config.model_is_mrope:
                forward_batch.mrope_positions.add_(1)
            N.maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

    draft_probs = (
        N.torch.stack(draft_probs_list, dim=1) if needs_draft_probs else None
    )

    # Organize the results
    if topk1_chain_fits:
        bs = token_list[0].shape[0]
        draft_tokens = N.torch.cat(token_list, dim=1)
        top_scores_index = self._topk1_score_indices_prealloc[:bs]
        parent_list = self._topk1_parents_prealloc[:bs]
        return parent_list, top_scores_index, draft_tokens, draft_probs

    parent_list, top_scores_index, draft_tokens = N.organize_draft_results(
        score_list, token_list, parents_list, self.speculative_num_draft_tokens
    )

    return parent_list, top_scores_index, draft_tokens, draft_probs


def _draft_extend_for_decode(
    self,
    batch: N.ScheduleBatch,
    batch_result: N.GenerationBatchResult,
    *,
    kv_loc_plan: N.Optional[N.KVLocPlan] = None,
):
    # Cast to int64 before entering plan stream to avoid cross-stream
    # synchronization issues with .to() inside the plan stream context.
    if batch_result.prepared_draft_extend_inputs is not None:
        num_correct_drafts, select_index, next_token_ids = (
            batch_result.prepared_draft_extend_inputs
        )
    elif batch_result.accept_lens.is_cuda:
        from sglang.kernels.ops.speculative.eagle import prepare_draft_extend_inputs

        num_correct_drafts, select_index, next_token_ids = (
            prepare_draft_extend_inputs(
                batch_result.accept_lens,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
            )
        )
    else:
        num_correct_drafts = batch_result.accept_lens - 1
        select_index = (
            N.torch.arange(
                0,
                len(batch.seq_lens) * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                device=self.device,
            )
            + batch_result.accept_lens
            - 1
        )

        next_token_ids = batch_result.next_token_ids.to(N.torch.int64)

    # Batch 2: Draft extend
    draft_extend_input = N.EagleDraftExtendInput(
        hidden_states=batch_result.logits_output.hidden_states,
        # accept_lens includes the bonus token; correct drafts exclude it.
        num_correct_drafts=num_correct_drafts,
        num_accept_tokens=batch_result.accept_lens,
        # Draft-extend fills the whole tree width (num_draft_tokens) per req,
        # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
        num_tokens_per_req=self.speculative_num_draft_tokens,
        num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
    )
    # Prepare for draft extend in a separate stream
    if self.plan_stream:
        self.plan_stream.wait_stream(
            N.torch.get_device_module(self.device).current_stream()
        )
    with self.plan_stream_ctx:
        forward_batch = N.prepare_for_draft_extend(
            draft_extend_input,
            batch,
            next_token_ids,
            self.speculative_num_draft_tokens,
            self.draft_runner,
            self.cuda_graph_runner_for_draft_extend,
            return_hidden_states_before_norm=False,
            kv_loc_plan=kv_loc_plan,
        )

    if self.plan_stream:
        N.torch.get_device_module(self.device).current_stream().wait_stream(
            self.plan_stream
        )

    # Run draft extend batch in the main compute stream
    can_run_decode_cuda_graph = (
        self.cuda_graph_runner_for_draft_extend
        and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
    )

    # Eager path publishes the indexer top-k into a worker buffer (the graph
    # path uses the runner's static buffer). Gathered at select_index below.
    if self.seed_dsa_topk_from_draft_extend and not can_run_decode_cuda_graph:
        forward_batch.spec_info.dsa_seed_topk_capture = (
            self._get_dsa_extend_topk_buf(forward_batch.input_ids.shape[0])
        )

    canary_ctx = (
        N.context_tuple(
            c.with_ops_outside_graph(
                single_forward_indices=[0],
                maybe_inaccurate_forward_batch=forward_batch,
            ),
            c.with_active_single_forward_manager(0),
        )
        if (c := self.draft_runner.canary_manager) is not None
        else N.contextlib.nullcontext()
    )
    with canary_ctx:
        if can_run_decode_cuda_graph:
            draft_logits_output = self.cuda_graph_runner_for_draft_extend.execute(
                forward_batch, select_index
            )
        else:
            draft_logits_output = self.draft_runner.forward(
                forward_batch
            ).logits_output

    N.maybe_detect_nan(
        draft_logits_output.next_token_logits,
        f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
    )
    N.maybe_detect_inf(
        draft_logits_output.next_token_logits,
        f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
    )

    # Gather the per-request last-position indexer top-k as the next loop's
    # seed (select_index already picks the last accepted position per req).
    dsa_seed_topk_indices = None
    if self.seed_dsa_topk_from_draft_extend:
        if can_run_decode_cuda_graph:
            dsa_extend_topk_capture = self.cuda_graph_runner_for_draft_extend.buffers.dsa_seed_topk_capture
        else:
            dsa_extend_topk_capture = forward_batch.spec_info.dsa_seed_topk_capture
        # Fancy indexing returns a fresh tensor (detached from the buffer).
        dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]

    # Reorganize the spec info for the next batch
    if not can_run_decode_cuda_graph:
        draft_logits_output.next_token_logits = (
            draft_logits_output.next_token_logits[select_index]
        )
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
    # Selected-row top-k remains worker-owned for both graph and eager
    # paths; the graph runner only moves the row selection before lm_head.
    if N.get_spec().speculative_use_rejection_sampling:
        ret_draft_probs, ret_topk_p, ret_topk_index = N.sample_draft_proposal(
            draft_logits_output.next_token_logits,
            batch.sampling_info.temperatures,
            batch.sampling_info.top_ks,
        )
    else:
        probs = N.renorm_draft_probs(
            draft_logits_output.next_token_logits,
            batch.sampling_info,
            N.get_spec().speculative_use_rejection_sampling,
        )
        ret_topk_p, ret_topk_index = N.fast_topk(probs, self.topk, dim=-1)
        ret_draft_probs = None
    ret_hidden_states = draft_logits_output.hidden_states

    # Construct the return values
    next_draft_input = batch_result.next_draft_input
    (
        next_draft_input.topk_p,
        next_draft_input.topk_index,
        next_draft_input.hidden_states,
    ) = (
        ret_topk_p,
        ret_topk_index,
        ret_hidden_states,
    )
    if N.get_spec().speculative_use_rejection_sampling:
        next_draft_input.draft_probs = ret_draft_probs
    if self.seed_dsa_topk_from_draft_extend:
        next_draft_input.dsa_topk_indices = dsa_seed_topk_indices
    next_draft_input.cuda_graph_compatible = not (
        self.seed_dsa_topk_from_draft_extend and dsa_seed_topk_indices is None
    )
