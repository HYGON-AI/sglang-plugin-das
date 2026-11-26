# SPDX-License-Identifier: Apache-2.0

import os
import socket
from unittest.mock import patch

import pytest
import torch

FLAG = "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION"
requires_gfx936 = pytest.mark.skipif(
    os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1"
    or torch.version.hip is None
    or not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).gcnArchName.partition(":")[0] != "gfx936",
    reason="Requires an isolated gfx936 GPU lane with HCU_PLUGIN_RUN_GPU_TESTS=1",
)


@pytest.fixture(scope="module")
def tp1():
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized():
        yield
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.MonkeyPatch.context() as env:
        for name, value in {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
        }.items():
            env.setenv(name, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        try:
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.fixture
def block_inputs(monkeypatch, tp1):
    from hcu_sglang.multimodal_gen import minimax_h3_modulation as modulation
    from hcu_sglang.multimodal_gen.minimax_h3_quant import MiniMaxH3Int8Config
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        component_attn_backend_context_manager,
    )
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.srt.plugins.hook_registry import _wrap_fn

    class Registry:
        @staticmethod
        def register(target, hook, kind):
            path = target.removeprefix(dit.__name__ + ".").split(".")
            owner = dit if len(path) == 1 else getattr(dit, path[0])
            name = path[-1]
            monkeypatch.setattr(owner, name, _wrap_fn(getattr(owner, name), hook, kind))

    modulation.register_hooks(Registry)
    monkeypatch.setenv("SGLANG_MINIMAX_H3_QUANT_OPTIMIZE", "1")
    monkeypatch.delenv(FLAG, raising=False)
    arch = MiniMaxH3DiTArchConfig(
        hidden_size=256,
        num_attention_heads=2,
        attention_head_dim=128,
        checkpoint_qkv_layout="native",
        ffn_hidden_size=512,
        adaln_curve_grid=1025,
        time_embed_dim=8,
    )
    config = MiniMaxH3Int8Config(
        targets={
            f"blocks.0.{name}"
            for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
        },
        compressed_adaln=True,
    )
    generator = torch.Generator(device="cuda").manual_seed(47)
    with (
        component_attn_backend_context_manager(
            AttentionBackendEnum.TORCH_SDPA, require_backend_selection=False,
        ),
        torch.device("cuda"),
    ):
        block = dit.MiniMaxH3DiTBlock(
            arch, config, prefix="blocks.0", use_adaln_cache=True,
        ).eval()
    with torch.no_grad():
        for name, param in block.named_parameters():
            if param.dtype == torch.int8:
                param.random_(-8, 9, generator=generator)
            elif name.endswith("weight_scale_inv"):
                param.fill_(0.005)
            else:
                param.fill_(1)
    rows = 65
    x = torch.randn((rows, arch.hidden_size), device="cuda", dtype=torch.bfloat16, generator=generator)
    table = torch.randn((4, 6 * arch.hidden_size), device=x.device, generator=generator) * 0.125
    kwargs = dict(
        adaln_input=None,
        combined_indices=torch.arange(rows, device=x.device) % 4,
        rope_cache=None,
        cu_seqlens=torch.tensor([0, rows], device=x.device, dtype=torch.int32),
        cu_seqlens_host=(0, rows),
        max_seqlen=rows,
        adaln_params=table.chunk(6, dim=-1),
    )
    return block, x, kwargs


@requires_gfx936
def test_fused_affine_rows_preserve_bf16_rounding_and_padding():
    from hcu_sglang.multimodal_gen import minimax_h3_quant as quant

    generator = torch.Generator(device="cuda").manual_seed(47)
    x = torch.randn((65, 112), device="cuda", dtype=torch.bfloat16, generator=generator)
    table = torch.randn((4, 224), device=x.device, generator=generator)
    shift, scale = table.chunk(2, dim=-1)
    indices = torch.arange(len(x), device=x.device, dtype=torch.int32) % len(table)
    expected_q, expected_scales = quant.quant_rows(quant.affine(x, shift, scale, indices))
    q, scales = quant.quant_affine_rows(x, shift, scale, indices)
    assert torch.equal(q, expected_q)
    assert torch.equal(scales, expected_scales)

    w = torch.randint(-8, 9, (80, x.shape[1]), device=x.device, dtype=torch.int8, generator=generator)
    sw = torch.full((len(w), 1), 0.005, device=x.device)
    expected = quant.linear(quant.affine(x, shift, scale, indices), w, sw, chunk=32)
    actual = quant.linear(x, w, sw, chunk=32, input_affine=(shift, scale, indices))
    assert torch.equal(actual, expected)


@requires_gfx936
def test_dit_fc1_affine_handoff_is_exact(monkeypatch, block_inputs):
    from hcu_sglang.multimodal_gen import minimax_h3_quant as quant

    block, x, kwargs = block_inputs
    with torch.inference_mode():
        expected = block(x.clone(), **kwargs)
        monkeypatch.setenv(FLAG, "1")
        with patch.object(quant, "quant_affine_rows", wraps=quant.quant_affine_rows) as fused:
            actual = block(x.clone(), **kwargs)
        fused.assert_called_once()
        assert fused.call_args.args[1] is kwargs["adaln_params"][3]
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@requires_gfx936
@pytest.mark.parametrize("case", ["upstream_fused", "projection_hook"])
def test_dit_preserves_existing_affine_consumers(monkeypatch, block_inputs, case):
    from hcu_sglang.multimodal_gen import minimax_h3_quant as quant
    from sglang.kernels.ops.diffusion.sites.minimax_h3_norm_modulate_site import (
        mount_minimax_h3_norm_modulate,
    )

    block, x, kwargs = block_inputs
    if case == "upstream_fused":
        assert mount_minimax_h3_norm_modulate(block)
    else:
        block.mlp.fc1.register_forward_pre_hook(lambda *_: None)
    with torch.inference_mode():
        expected = block(x.clone(), **kwargs)
        monkeypatch.setenv(FLAG, "1")
        with patch.object(quant, "quant_affine_rows", wraps=quant.quant_affine_rows) as fused:
            actual = block(x.clone(), **kwargs)
        fused.assert_not_called()
    assert torch.equal(actual, expected)
