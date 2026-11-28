# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def activate():
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu() or not torch.cuda.is_available():
        pytest.skip("HCU runtime required")
    from sglang.srt.plugins import load_plugins
    load_plugins()


@pytest.mark.parametrize("page_size", [128, 256])
def test_v4_rope_store_matches_two_stage_cache(page_size):
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool, DeepSeekV4SingleKVPool
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.dsv41_sparse import _rope_fq4
    from hcu_sglang.mem_cache.hcu_v4_pool import DeepSeekV4TokenToKVPool as OwnedPool

    assert DeepSeekV4TokenToKVPool is OwnedPool and OwnedPool.supports_fp8_rope_fq_store
    def make_pool():
        child = object.__new__(DeepSeekV4SingleKVPool)
        child.page_size = page_size
        child.dtype = torch.float8_e4m3fn
        child.kv_layout = KVLayout.V4
        cache = torch.full((3, KVLayout.V4.page_bytes(page_size)), 0x55, device="cuda", dtype=torch.uint8)
        child.kv_buffer = [cache]
        parent = object.__new__(DeepSeekV4TokenToKVPool)
        parent.layer_mapping = [(None,0,child)]
        return parent, cache

    torch.manual_seed(727)
    latent = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    original = latent.clone()
    phase = torch.randn(7,32,device="cuda")
    freqs = torch.polar(torch.ones_like(phase), phase)
    loc = torch.tensor([1, page_size-1, page_size, page_size+2, 0, 2*page_size+1, 9], device="cuda")
    keep = torch.tensor([True,True,True,True,False,True,False],device="cuda")
    ref_pool, ref_bytes = make_pool()
    fast_pool, fast_bytes = make_pool()
    rotated = _rope_fq4(latent[keep].contiguous(),freqs[keep],64,compressed_kv=True)
    ref_pool.set_extra_key_buffer_fused(0,loc[keep],rotated)
    fast_pool.set_extra_key_buffer_fused(0,loc,latent,valid_mask=keep,freqs_cis=freqs)
    torch.testing.assert_close(fast_bytes, ref_bytes, rtol=0, atol=0)
    torch.testing.assert_close(latent,original,rtol=0,atol=0)


def test_layer_split_owner_controls_rope_store():
    from hcu_sglang.mem_cache.cp_cache_layer_split.deepseek_v4_pool import CpCacheLayerSplitDeepSeekV4TokenToKVPool as Split
    loc, rows, freq = object(), object(), object()
    for owns in (True,False):
        child = Mock()
        handle = SimpleNamespace(compress_layer_id=5,compress_kv_pool=child)
        pool = SimpleNamespace(compression_ratios=[2],_extra_full_read_key=object(),
                               _owns_extra_key_layer_id=lambda _: owns,
                               _owner_compress_layer_item=lambda _: handle)
        Split.set_extra_key_buffer_fused(pool,0,loc,rows,freqs_cis=freq)
        assert pool._extra_full_read_key is None
        if owns:
            child.set_key_buffer_fused.assert_called_once_with(5,loc,rows,None,freq)
        else:
            child.set_key_buffer_fused.assert_not_called()
