import json
import os
import unittest

if os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS') != '1':
    raise SystemExit('Set HCU_PLUGIN_RUN_GPU_TESTS=1 for the GPU regression')

import torch
from sglang.srt.plugins import load_plugins

load_plugins()

from hcu_sglang.layers.attention.debug_flash_mla_adapter import (
    _FP8_DTYPE,
    flash_mla_sparse_fwd_entrypoint,
)
from sglang.srt.layers.attention.hip_flash_mla import (
    flash_mla_with_kvcache_entrypoint,
)
from sglang.test.test_utils import CustomTestCase


def packed_cache(value, poison):
    data = torch.zeros((1, 8 * 584), dtype=torch.uint8)
    data[:, 8 * 576:] = 127
    for token in range(8):
        offset = token * 576
        data[0, offset:offset + 448] = torch.full((448,), value).to(
            _FP8_DTYPE
        ).view(torch.uint8)
        data[0, offset + 448:offset + 576] = torch.full(
            (64,), value, dtype=torch.bfloat16
        ).view(torch.uint8)
    if poison:
        data[0, :576] = 255
    return data.reshape(1, 8, 584)


def run_attention(mode, query, cache, indices, lengths, sink):
    if mode == 'paged':
        return flash_mla_with_kvcache_entrypoint(
            backend='torch', q=query, k_cache=cache, head_dim_v=512,
            softmax_scale=0.01, indices=indices, topk_length=lengths,
            attn_sink=sink,
        )[0]
    return flash_mla_sparse_fwd_entrypoint(
        q=query, kv=cache, indices=indices.unsqueeze(1), sm_scale=0.01,
        d_v=512, attn_sink=sink, topk_length=lengths,
    )[0]


class TestFlashMLAMaskedKV(CustomTestCase):
    def test_masked_values_and_graph_updates(self):
        cases = []
        for mode in ['paged', 'sparse']:
            for heads in [8, 64]:
                for name, ids, length in [
                    ('finite', [-1, 1], 2),
                    ('masked_nan', [-1, 1], 2),
                    ('length_nan', [1, 0], 1),
                    ('multiple_tiles', [-1] * 64 + [1], 65),
                    ('all_invalid', [-1, -1], 2),
                ]:
                    poison = name != 'finite'
                    query = torch.ones((1, heads, 512), device='cuda', dtype=torch.bfloat16)
                    indices = torch.tensor([ids], device='cuda', dtype=torch.int32)
                    lengths = torch.tensor([length], device='cuda', dtype=torch.int32)
                    sink = torch.zeros(heads, device='cuda')

                    def values(value):
                        if mode == 'paged':
                            return packed_cache(value, poison)
                        result = torch.full((8, 1, 512), value, dtype=torch.bfloat16)
                        if poison:
                            result[0] = float('nan')
                        return result

                    cache = values(0.25).cuda()
                    call = lambda: run_attention(mode, query, cache, indices, lengths, sink)
                    call()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = call()
                    snapshots = []
                    for state, value in enumerate([0.25, 0.5, 1.0]):
                        cache.copy_(values(value))
                        eager = call()
                        graph.replay()
                        torch.cuda.synchronize()
                        eager_cpu = eager.cpu()
                        expected = 0.0 if name == 'all_invalid' else float(
                            value * torch.sigmoid(torch.tensor(5.12 * value))
                        )
                        torch.testing.assert_close(
                            eager_cpu.float(), torch.full_like(eager_cpu.float(), expected),
                            rtol=0.01, atol=0.005,
                        )
                        torch.testing.assert_close(eager_cpu, captured.cpu(), rtol=0, atol=0)
                        snapshots.append(float(eager_cpu.flatten()[0]))
                        cases.append({'mode': mode, 'heads': heads, 'case': name, 'state': state})
                    if name != 'all_invalid':
                        self.assertEqual(len(set(snapshots)), 3)
                cache = packed_cache(1.0, True).cuda() if mode == 'paged' else torch.full(
                    (8, 1, 512), float('nan'), device='cuda', dtype=torch.bfloat16
                )
                indices = torch.tensor([[0]], device='cuda', dtype=torch.int32)
                lengths = torch.ones(1, device='cuda', dtype=torch.int32)
                result = run_attention(mode, query, cache, indices, lengths, sink)
                self.assertTrue(bool(torch.isnan(result).all()))
                cases.append({'mode': mode, 'heads': heads, 'case': 'valid_nan_preserved'})
        print(json.dumps({'passed': True, 'scope': 'public entry GPU regression only', 'cases': cases}), flush=True)


if __name__ == '__main__':
    unittest.main()
