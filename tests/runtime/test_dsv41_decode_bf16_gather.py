from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hcu_sglang.environ import register_envs
from hcu_sglang.models.deepseek.current_v4.flat_kv_dequant import (
    KVLayout,
    configure_bf16_decode_gather,
    use_bf16_decode_gather,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

FLASH = "SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA"
NATIVE = "SGLANG_DSV4_HCU_USE_LIGHTOP_BF16_GATHER"


def options(quant="fp8", layout=KVLayout.V4, bf16=False, model="deepseek_v41"):
    config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type=model),
        hf_quant_config={"quant_method": quant} if isinstance(quant, str) else quant,
    )
    return SimpleNamespace(
        model_runner=SimpleNamespace(model_config=config),
        token_to_kv_pool=SimpleNamespace(kv_layout=layout, is_bf16_attention_kv_cache=bf16),
    )


def test_defaults_and_rollback(monkeypatch):
    register_envs()
    monkeypatch.delenv(FLASH, raising=False)
    monkeypatch.delenv(NATIVE, raising=False)
    from lightop import op
    for backend, hcu in [(options(), False), (options(layout=KVLayout.V41), True),
                         (options(bf16=True), True), (options(model="deepseek_v4"), True),
                         (options(quant={"quant_method": "fp8", "routed_experts_quant_method": "mxfp4"}), True)]:
        configure_bf16_decode_gather(backend, hcu)
        assert not backend._dsv4_bf16_flashmla_decode
    backend = options()
    configure_bf16_decode_gather(backend, True)
    assert backend._dsv4_lightop_kvcache_op is op
    for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
        assert use_bf16_decode_gather(backend, mode)
    for mode in (ForwardMode.EXTEND, ForwardMode.IDLE):
        assert not use_bf16_decode_gather(backend, mode)
    monkeypatch.setenv(FLASH, "0")
    configure_bf16_decode_gather(backend, True)
    assert not backend._dsv4_bf16_flashmla_decode and not backend._dsv4_lightop_bf16_gather
    monkeypatch.setenv(NATIVE, "1")
    with pytest.raises(RuntimeError, match="requires"):
        configure_bf16_decode_gather(backend, True)
    monkeypatch.delenv(FLASH)
    monkeypatch.delenv(NATIVE)
    with patch.object(op, "dsv4_gather_upconvert_k_cache_paged", None):
        configure_bf16_decode_gather(backend, True)
        assert backend._dsv4_bf16_flashmla_decode and not backend._dsv4_lightop_bf16_gather
        monkeypatch.setenv(NATIVE, "1")
        with pytest.raises(RuntimeError, match="missing"):
            configure_bf16_decode_gather(backend, True)


def test_routed_family_scope_and_explicit_legacy_mode(monkeypatch):
    register_envs()
    monkeypatch.delenv(FLASH, raising=False)
    monkeypatch.delenv(NATIVE, raising=False)
    def with_weights(kind, bits):
        return {
            "targets": ["re:.*ffn.experts.*"],
            "weights": {"type": kind, "num_bits": bits},
            "input_activations": {"type": kind, "num_bits": 8},
        }
    cfg = {"quant_method": "compressed-tensors", "config_groups": {"routed": with_weights("float", 8)}}
    backend = options(quant=cfg)
    configure_bf16_decode_gather(backend, True)
    assert backend._dsv41_fp8_decode_gather
    cfg["config_groups"]["routed"] = with_weights("int", 4)
    configure_bf16_decode_gather(backend, True)
    assert not backend._dsv4_bf16_flashmla_decode
    monkeypatch.setenv(FLASH, "1")
    configure_bf16_decode_gather(backend, True)
    assert not backend._dsv4_lightop_bf16_gather
    assert use_bf16_decode_gather(backend, ForwardMode.EXTEND)
    assert not use_bf16_decode_gather(backend, ForwardMode.IDLE)


def test_normalized_channel_checkpoint(monkeypatch):
    from hcu_sglang.configs import after_normalize_deepseek_v41_config
    from sglang.srt.configs.deepseek_v41 import normalize_deepseek_v41_config
    monkeypatch.setattr("hcu_sglang.configs.is_hcu", lambda: True)
    register_envs()
    monkeypatch.delenv(FLASH, raising=False)
    monkeypatch.delenv(NATIVE, raising=False)
    normalized = normalize_deepseek_v41_config({
        "model_type": "deepseek_v41", "text_config": {"model_type": "deepseek_v41_text"},
        "quantization_config": {"quant_method": "fp8", "is_per_channel": True, "activation_scheme": "dynamic"},
    })
    quant = after_normalize_deepseek_v41_config(normalized, {})["quantization_config"]
    assert quant["config_groups"]["group_0"]["targets"] == ["Linear"]
    backend = options(quant=quant, model=normalized["model_type"])
    configure_bf16_decode_gather(backend, True)
    assert backend._dsv41_fp8_decode_gather and backend._dsv4_bf16_flashmla_decode
    quant["config_groups"]["group_0"]["weights"]["num_bits"] = 4
    configure_bf16_decode_gather(backend, True)
    assert not backend._dsv4_bf16_flashmla_decode


def make_cache(page, shift):
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.dequant_k_cache import fp8_dtype

    stride = ((page * 584 + 575) // 576) * 576
    buf = torch.zeros((2, stride), dtype=torch.uint8, device="cuda")
    payload = buf[:, :page * 576].view(2, page, 576)
    scale = buf[:, page * 576:page * 584].view(2, page, 8)
    loc = torch.arange(2 * page, device="cuda").view(2, page, 1)
    values = ((loc + torch.arange(512, device="cuda") + shift) % 15 - 7).to(torch.bfloat16)
    exps = (loc + torch.arange(7, device="cuda") + shift) % 3 - 1
    scale[:, :, :7].copy_((exps + 127).to(torch.uint8))
    payload[:, :, :448].copy_(values[:, :, :448].to(fp8_dtype).view(torch.uint8))
    payload[:, :, 448:].copy_(values[:, :, 448:].contiguous().view(torch.uint8))
    expected = values.clone()
    expected[:, :, :448].mul_(torch.exp2(exps.float()).repeat_interleave(64, dim=-1))
    return buf[:, :page*584].view(2, page, 1, 584), expected.view(-1, 512)


@pytest.mark.parametrize("queries,extra_page", [(1, None), (3, 128), (2, 256)])
@pytest.mark.parametrize("native", [True, False])
def test_decode_gather_graph_changes_and_reference(queries, extra_page, native, monkeypatch):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu() or not torch.cuda.is_available():
        pytest.skip("requires HCU")
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.backend import DeepseekV4AttnBackend
    register_envs()
    monkeypatch.setenv(FLASH, "1")
    monkeypatch.setenv(NATIVE, str(int(native)))
    backend = object.__new__(DeepseekV4AttnBackend)
    for key, value in vars(options()).items():
        setattr(backend, key, value)
    backend.device = "cuda"
    configure_bf16_decode_gather(backend, True)
    assert backend._dsv41_fp8_decode_gather
    swa, swa_ref = make_cache(256, 3)
    extra, extra_ref = make_cache(extra_page, 11) if extra_page else (None, None)
    i = torch.arange(queries * 128, device="cuda", dtype=torch.int32).view(queries, 1, 128)
    j = i[:, :, :64].clone()
    lengths = torch.full((queries,), 9, device="cuda", dtype=torch.int32)
    elengths = lengths.clone()
    def gather():
        return backend._prepare_dsv4_bf16_flashmla_inputs(
            swa_k_cache=swa, swa_indices=i, swa_topk_lengths=lengths, extra_k_cache=extra,
            extra_indices=j if extra_page else None, extra_topk_lengths=elengths if extra_page else None,
        )
    gather()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gather()
    for step in (0, 1, 2):
        i.add_(137 + step).remainder_(512)
        j.add_(13 + step).remainder_(2 * (extra_page or 1))
        lengths.copy_((torch.arange(queries, device="cuda") + 5 + step).int())
        elengths.copy_((torch.arange(queries, device="cuda") + 1 + step).int())
        eager = gather()
        graph.replay()
        for result in (eager, captured):
            for row in range(queries):
                n = int(lengths[row])
                e = int(elengths[row]) if extra_page else 0
                expect_swa = swa_ref[i[row, 0, :n].long()]
                expect_extra = extra_ref[j[row, 0, :e].long()] if extra_page else None
                ids = result[1][row, 0, :n].long()
                torch.testing.assert_close(result[0].view(-1, 512)[ids], expect_swa, rtol=0, atol=0)
                if extra_page:
                    if native:
                        storage, ids = result[3], result[4][row, 0, :e]
                    else:
                        storage, ids = result[0], result[1][row, 0, n:n + e]
                    torch.testing.assert_close(storage.view(-1, 512)[ids.long()], expect_extra, rtol=0, atol=0)
                assert int(result[2][row]) == (n if native else n + e)
        assert backend._dsv4_bf16_flashmla_workspaces == {}


def test_native_flashmla_decode_uses_gathered_rows(monkeypatch):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu() or not torch.cuda.is_available():
        pytest.skip("requires HCU")
    from sglang.srt.plugins import load_plugins
    load_plugins()
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import backend as attention
    monkeypatch.setenv(FLASH, "1")
    monkeypatch.setenv(NATIVE, "1")
    backend = object.__new__(attention.DeepseekV4AttnBackend)
    for key, value in vars(options(model="deepseek_v41_text")).items():
        setattr(backend, key, value)
    backend.device = "cuda"
    backend.softmax_scale = 512 ** -0.5
    backend.head_dim_v = 512
    configure_bf16_decode_gather(backend, True)
    assert backend._dsv41_fp8_decode_gather
    packed, expected = make_cache(256, 3)
    extra, extra_expected = make_cache(128, 8)
    ids = torch.zeros(2, 128, device="cuda", dtype=torch.int32)
    extra_ids = torch.zeros(2, 64, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
    extra_lengths = torch.tensor([3, 2], device="cuda", dtype=torch.int32)
    q = torch.zeros(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(64, device="cuda")
    import flash_mla
    metadata = flash_mla.get_mla_metadata()[0]
    def forward():
        return backend._forward_flash_mla_decode(
            q=q, swa_k_cache=packed, swa_page_indices=ids, swa_topk_lengths=lengths,
            flashmla_metadata=metadata, attn_sink=sink, extra_k_cache=extra,
            extra_indices=extra_ids, extra_topk_lengths=extra_lengths, compress_ratio=2,
            layer_id=0, forward_batch=SimpleNamespace(forward_mode=ForwardMode.DECODE),
        )
    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for step in (0, 1, 2):
        ids.copy_((torch.arange(256, device="cuda").view(2, 128) + 253 + step) % 512)
        extra_ids.copy_((torch.arange(128, device="cuda").view(2, 64) + 126 + step) % 256)
        lengths.add_(1)
        extra_lengths.add_(1)
        eager = forward()
        graph.replay()
        for row in range(2):
            n, e = int(lengths[row]), int(extra_lengths[row])
            rows = torch.cat([expected[ids[row, :n].long()], extra_expected[extra_ids[row, :e].long()]])
            mean = rows.float().sum(0).div(n + e + 1).to(torch.bfloat16)
            torch.testing.assert_close(eager[row].float(), mean.expand_as(eager[row]).float(), rtol=.01, atol=.015625)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
