"""Compare the active CP C2 pooling path before either KV writer consumes it."""
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.skipif(not (torch.version.hip and torch.cuda.is_available()), reason="HCU required")
def test_c2_fused_pool_norm_preserves_bf16_rounding():
    import lightop
    from hcu_sglang.environ import register_envs
    register_envs()
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import backend
    from sglang.srt.environ import envs

    torch.manual_seed(616)
    n = 63
    # CP-local packed projection produces row-strided contiguous columns.
    a, b = torch.randn(n, 1024, device="cuda").chunk(2, -1)
    c, d = torch.randn(n, 1024, device="cuda").chunk(2, -1)
    weight = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    pos = torch.arange(n, device="cuda")
    core = NS(raw_out_loc=pos, c2_out_loc=pos // 2)
    compressor = NS(norm=NS(weight=weight, eps=1e-6),
                    pool_pairs=lambda x,s: (x * s.softmax(1)).sum(1))
    layer = NS(layer_id=1, compress_ratio=2, compressor=compressor)
    owner = NS(forward_metadata=NS(core_metadata=core),
               _low_ratio_write_group=Mock(), _low_ratio_pair_partners=Mock(return_value=(c,d)))
    def run(enable):
        with envs.SGLANG_HCU_OPT_DSV41_CP_POOL_NORM.override(enable):
            backend.DeepseekV4AttnBackend._low_ratio_compress_torch(
                owner, layer, None, pos * 0, pos, (a,b), fuse_rope_fq_store=True)
        args, kwargs = owner._low_ratio_write_group.call_args
        return args[1], kwargs["normalized_latent"]
    with patch.object(backend, "get_parallel", return_value=NS(attn_cp_size=8)):
        base, absent = run(False)
        assert absent is None
        with patch.object(lightop, "dsv41_pool_pairs_norm", wraps=lightop.dsv41_pool_pairs_norm) as fused:
            absent, actual = run(True)
            fused.assert_called_once()
        assert absent is None
    x = base.bfloat16().float()
    ref = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).bfloat16()
    torch.testing.assert_close(actual, ref, rtol=0.008, atol=0.001)
