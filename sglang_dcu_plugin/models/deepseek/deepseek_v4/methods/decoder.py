"""DCU DeepSeek V4 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_v4 import constants as C
from sglang_dcu_plugin.models.deepseek.deepseek_v4.methods._runtime import *  # noqa: F403

def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: Optional[nn.Module] = None,
    ):
        """Return MHC-pre outputs and whether the optional RMSNorm was fused."""

        @compile_in_capture_mode
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        shape, dtype = x.size(), x.dtype

        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=dtype, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=dtype, device=x.device
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            if C._is_dcu and C._use_aiter_tilelang_mhc:
                post, comb, y = mhc_pre_big_fuse(
                    residual=x,
                    fn=hc_fn,
                    mhc_scale=hc_scale,
                    mhc_base=hc_base,
                    rms_eps=self.rms_norm_eps,
                    mhc_pre_eps=self.hc_eps,
                    mhc_sinkhorn_eps=self.hc_eps,
                    mhc_post_mult_value=2.0,
                    sinkhorn_repeat=self.hc_sinkhorn_iters,
                    n_splits=16,
                )
                # AITER MHC pre does not fuse the decoder-layer RMSNorm.
                norm_fused = False
            else:
                from sglang.srt.layers.mhc import mhc_pre

                norm_kwargs = {}
                if norm is not None:
                    norm_kwargs["norm_weight"] = norm.weight.data
                    norm_kwargs["norm_eps"] = norm.variance_epsilon

                post, comb, y = mhc_pre(
                    residual=x,
                    fn=hc_fn,
                    hc_scale=hc_scale,
                    hc_base=hc_base,
                    rms_eps=self.rms_norm_eps,
                    hc_pre_eps=self.hc_eps,
                    hc_sinkhorn_eps=self.hc_eps,
                    hc_post_mult_value=2.0,
                    sinkhorn_repeat=self.hc_sinkhorn_iters,
                    **norm_kwargs,
                )
                norm_fused = norm is not None
            return y, post.squeeze(-1), comb, norm_fused

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            # import deep_gemm
            import deepgemm as deep_gemm

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            deep_gemm.tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        from sglang.srt.layers.mhc import hc_split_sinkhorn

        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1), False

def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):

        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            if C._is_dcu and C._use_aiter_tilelang_mhc:
                out = mhc_post_fwd(
                    x, 
                    residual,
                    post, 
                    comb, 
                )
                return out
            else:
                from sglang.srt.layers.mhc import mhc_post
                return mhc_post(x, residual, post, comb)
                # return mhc_post_torch(x, residual, post, comb)

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        @compile_in_capture_mode
        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)

        return hc_post_torch_impl(x, residual, post, comb)

def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            norm=self.input_layernorm,
        )
        if not norm_fused:
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            x=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
        )

        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        residual = hidden_states
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm=self.post_attention_layernorm,
        )
        if not norm_fused:
            hidden_states = self.post_attention_layernorm(hidden_states)

        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_attention_dp_size() > 1
            and get_moe_a2a_backend().is_none()
        )
        _use_tp_attn_a2a_scatter = (
            not _use_cp
            and envs.SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER.get()
            and get_attention_tp_size() > 1
            and not get_moe_a2a_backend().is_none()
        )
        if _use_cp:
            assert get_moe_a2a_backend().is_deepep(), (
                "CP requires DeepEP (moe_a2a_backend == deepep). "
                "Only DeepEP is tested with CP's per-rank token split."
            )
            cp_rank = get_attention_cp_rank()
            cp_size = get_attention_cp_size()
            input_ids = input_ids[cp_rank::cp_size].contiguous()
            input_ids_global = input_ids
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
        _a2a_scatter_chunks: Optional[List[torch.Tensor]] = None
        if _use_tp_attn_a2a_scatter:
            s, r = get_attention_tp_size(), get_attention_tp_rank()
            _a2a_scatter_chunks = list(hidden_states.tensor_split(s))
            hidden_states = _a2a_scatter_chunks[r].contiguous()
            input_ids = input_ids.tensor_split(s)[r].contiguous()
            input_ids_global = input_ids_global.tensor_split(s)[r].contiguous()
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )
        if _use_tp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
        if _use_tp_attn_a2a_scatter:
            assert _a2a_scatter_chunks is not None
            gathered = [torch.empty_like(t) for t in _a2a_scatter_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)

        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        dsv4_2604_submode = getattr(envs, "SGLANG_DSV4_2604_SUBMODE", None)
        if dsv4_2604_submode is not None and dsv4_2604_submode.get() == "2604B":
            # assert deepseek_v4_moe_code_path_checker.observed == 1
            deepseek_v4_moe_code_path_checker.observed = 0

        return hidden_states

