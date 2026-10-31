"""HCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_v2 import constants as C
from sglang_hcu_plugin.models.deepseek.deepseek_v2.methods._runtime import *  # noqa: F403

def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        update_hd: Optional[bool] = False,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            if C._use_fused_rms_quant and rms_weight is not None and residual is not None:
                return x, None, None, None
            return x

        if (
            gemm_output_zero_allocator is not None
            and x.shape[0] <= 256
            and self.gate_up_proj.weight.dtype == torch.uint8
        ):
            y = gemm_output_zero_allocator.allocate(
                x.shape[0] * self.gate_up_proj.output_size_per_partition
            ).view(x.shape[0], self.gate_up_proj.output_size_per_partition)
            x = (x, None, y)

        gate_up, _ = self.gate_up_proj(x)
        # Fast path: fused silu+clamp+fp8_quant+deepgemm when conditions met.
        # Only valid when down_proj does NOT need an all-reduce and its weights
        # are fp8 (uint8 storage with weight_scale_inv).
        if (
            self.swiglu_limit is not None
            and not self.down_proj.reduce_results
            and self.down_proj.weight.dtype == torch.uint8
            and hasattr(self.down_proj, "weight_scale_inv")
        ):
            M, N = gate_up.shape
            down_input_fp8 = gate_up.new_empty((M, N // 2), dtype=torch.float8_e4m3fn)
            scale_block_size = 128
            down_input_scale = create_per_token_group_quant_fp8_output_scale(
                x_shape=(M, N // 2),
                device=gate_up.device,
                group_size=scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            silu_and_mul_contig_post_quant(
                input=gate_up,
                output=down_input_fp8,
                output_scale=down_input_scale,
                quant_group_size=scale_block_size,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                swiglu_limit=float(self.swiglu_limit),
            )
            down_output = gate_up.new_empty(
                (M, self.down_proj.output_size), dtype=torch.bfloat16
            )
            deep_gemm_wrapper.gemm_nt_f8f8bf16(
                (down_input_fp8, down_input_scale),
                (self.down_proj.weight, self.down_proj.weight_scale_inv),
                down_output,
            )
            return down_output
        # Fallback: fused silu+clamp kernel (still faster than unfused)
        if self.swiglu_limit is not None:
            M, N = gate_up.shape
            x = gate_up.new_empty((M, N // 2))
            silu_and_mul_clamp(gate_up, x, float(self.swiglu_limit))
        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return x

