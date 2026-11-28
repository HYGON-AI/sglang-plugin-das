# SPDX-License-Identifier: Apache-2.0
"""Compare both Engram call paths to official torch, including history side effects."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def runtime():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires a ROCm device")
    from hcu_sglang.environ import register_envs
    from hcu_sglang.layers import engram_hash
    from sglang.srt.layers import engram
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    register_envs()
    return engram, engram_hash, ForwardMode


def _hasher(engram):
    h = object.__new__(engram.EngramHasher)
    torch.nn.Module.__init__(h)
    h.max_ngram_size, h.pad_id, h.image_token_id = 4, 3, 7
    h.token_map = torch.arange(32, device="cuda").roll(3)
    h.primes = torch.tensor([31, 37, 41, 43, 47, 53, 59, 61], device="cuda").repeat(2, 3, 1)
    h.multipliers = engram.compute_hash_multipliers((0, 3), 4, 32).to("cuda")
    sizes = h.primes.flatten(1)
    h.offsets = torch.cumsum(sizes, 1) - sizes
    h.init_history(4, "cuda")
    h.history.copy_(torch.arange(15, dtype=torch.int32, device="cuda").reshape(5, 3))
    return h


@pytest.mark.parametrize("native_dispatch", [False, True], ids=["torch-seam", "kernel-seam"])
@pytest.mark.parametrize("case", ["slots", "scheduler", "verify"])
def test_hash_and_history_match_torch(runtime, monkeypatch, native_dispatch, case):
    engram, hooks, mode = runtime
    real_op, block_supported = hooks._provider()
    if real_op is None or (case == "verify" and not block_supported):
        pytest.skip("installed LightOp does not export this hash interface")
    device = {"device": "cuda", "dtype": torch.int32}
    h = _hasher(engram)
    baseline = _hasher(engram)
    ids = torch.tensor([5, 3, engram.MM_PAD_SHIFT_VALUE + 17, 9, 8, 10, 1, 2], **device)
    positions = torch.tensor([3, 4, 0, 1, 2, 3, 0, 0], device="cuda")
    batch = SimpleNamespace(
        forward_mode=mode.TARGET_VERIFY if case == "verify" else mode.EXTEND,
        req_pool_indices=torch.tensor([3, 1], **device),
        spec_info=SimpleNamespace(draft_token_num=4),
        extend_seq_lens=torch.tensor([2, 3], **device),
        extend_start_loc=torch.tensor([0, 2], **device),
        extend_seq_lens_cpu=[2, 3],
        out_cache_loc=torch.tensor([1, 2, 3, 4, 0, 0, 0, 0], **device),
        engram_history=None if case != "scheduler" else h.history[[1, 2]].clone(),
        positions=positions,
    )
    monkeypatch.setenv("SGLANG_OPT_ENGRAM_EXTEND_HASH", "0")
    monkeypatch.setattr(engram, "_cuda_kernels", lambda _: False)
    monkeypatch.setenv("SGLANG_HCU_OPT_ENGRAM_HASH_IDS", "0")
    expected = baseline(ids, batch)
    op = Mock(wraps=real_op)
    monkeypatch.setattr(hooks, "_provider", lambda: (op, block_supported))
    monkeypatch.setenv("SGLANG_HCU_OPT_ENGRAM_HASH_IDS", "1")
    monkeypatch.setattr(engram, "_cuda_kernels", lambda _: native_dispatch)
    if native_dispatch:
        def dispatch(*args, **kwargs):
            hashes, tokens = hooks.around_hash_ids(None, *args, **kwargs)
            assert tokens.dtype == torch.int32
            return hashes, tokens

        monkeypatch.setattr(engram, "engram_hash_ids", dispatch)
    else:
        def dispatch(owner, *args):
            hashes, tokens = hooks.around_torch_hash_ids(None, owner, *args)
            assert tokens.dtype == torch.int64
            return hashes, tokens

        monkeypatch.setattr(engram.EngramHasher, "_torch_hash_ids", dispatch)
    result = h(ids, batch)
    op.assert_called_once()
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(h.history, baseline.history, atol=0, rtol=0)


def test_old_verify_export_and_disabled_fallback(runtime, monkeypatch):
    engram, hooks, _ = runtime
    h = _hasher(engram)
    ids = torch.arange(4, device="cuda")
    op = Mock()
    monkeypatch.setattr(hooks, "_provider", lambda: (op, False))
    original = Mock(return_value=object())
    for setting in ("1", "0"):
        monkeypatch.setenv("SGLANG_HCU_OPT_ENGRAM_HASH_IDS", setting)
        assert hooks.around_hash_ids(original, ids, ids, mode=engram.MODE_VERIFY,
            history=h.history[:1], token_map=h.token_map, multipliers=h.multipliers,
            primes=h.primes, offsets=h.offsets, pad_id=0, block=4) is original.return_value
    assert original.call_count == 2
    op.assert_not_called()


def test_gate_matches_torch(runtime, monkeypatch):
    engram, hooks, _ = runtime
    monkeypatch.setattr(engram, "_cuda_kernels", lambda _: False)
    g = torch.Generator(device="cuda").manual_seed(45)
    x, kv, qw, kw = [torch.randn(shape, generator=g, device="cuda", dtype=torch.bfloat16)
                     for shape in [(7, 4, 256), (7, 5 * 256), (4, 256), (4, 256)]]
    args = (x, kv, qw, kw, 1e-6, 1e-6)
    monkeypatch.setenv("SGLANG_HCU_OPT_ENGRAM_GATE", "0")
    baseline = hooks.around_gate(engram.engram_gate, *args)
    monkeypatch.setenv("SGLANG_HCU_OPT_ENGRAM_GATE", "1")
    result = hooks.around_gate(None, *args)
    torch.testing.assert_close(result, baseline, atol=0.02, rtol=0.01)
