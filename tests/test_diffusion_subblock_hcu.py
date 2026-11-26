from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hcu_sglang.multimodal_gen import subblock_sparse as bsa


SGLANG_BACKEND = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn"
)


class SubBlockHcuTests(unittest.TestCase):
    def test_only_active_prefix_is_sorted(self):
        index = torch.tensor([[[[4, 2, 0, 1, 3], [4, 1, 3, 0, 2]]]], dtype=torch.int32)
        original = index.clone()
        counts = torch.tensor([[[2, 5]]], dtype=torch.int32)
        ordered = bsa._sorted_active_indices(index, counts, 5)
        self.assertEqual(ordered.tolist(), [[[[2, 4, 0, 1, 3], [0, 1, 2, 3, 4]]]])
        self.assertTrue(torch.equal(index, original))

    def test_other_platform_uses_original_runner(self):
        original = Mock()
        device = torch.device("cuda")
        with patch.object(bsa, "is_hcu", return_value=False):
            runner = bsa.around_get_runner(original, device, "sage_fp8")
        self.assertIs(runner, original.return_value)
        original.assert_called_once_with(device, "sage_fp8")

    def test_aiter_runner_returns_output_tensor(self):
        q = torch.empty(1, 1, 1, 128, dtype=torch.bfloat16)
        k = v = torch.empty(1, 65, 1, 128, dtype=q.dtype)
        index = torch.tensor([[[[1, 0]]]], dtype=torch.int32)
        counts = torch.tensor([[[2]]], dtype=torch.int32)
        sizes = torch.tensor([64, 1], dtype=torch.int32)
        output = torch.empty_like(q)
        kernel = Mock(return_value=output)
        cached_sizes = Mock(return_value=sizes)
        with patch.dict(
            "sys.modules",
            {
                "aiter": SimpleNamespace(bsa_attn_blk64_fwd=kernel),
                SGLANG_BACKEND: SimpleNamespace(_cached_block_sizes=cached_sizes),
            },
        ):
            self.assertIs(bsa.hcu_sparse_attention(q, k, v, index, 2, 0.125, counts), output)
        cached_sizes.assert_called_once_with(65, k.device)
        actual_q, actual_k, actual_v, actual_index, topk = kernel.call_args.args
        self.assertIs(actual_q, q)
        self.assertIs(actual_k, k)
        self.assertIs(actual_v, v)
        self.assertEqual(actual_index.tolist(), [[[[0, 1]]]])
        self.assertEqual(topk, 2)
        self.assertIs(kernel.call_args.kwargs["block_sizes"], sizes)
        self.assertIs(kernel.call_args.kwargs["q2k_block_nums"], counts)
        self.assertEqual(kernel.call_args.kwargs["softmax_scale"], 0.125)

    @unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires HIP")
    def test_partial_key_block_and_heterogeneous_counts(self):
        if find_spec("aiter") is None:
            self.skipTest("requires AITER with SubBlock sparse attention")
        from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
            _cached_block_sizes,
        )

        torch.manual_seed(1101)
        q = torch.randn(1, 129, 2, 256, device="cuda", dtype=torch.bfloat16)[..., :128]
        k, v = torch.randn(1, 257, 2, 256, device=q.device, dtype=q.dtype).split(128, -1)
        row = torch.tensor([4, 1, 3, 2, 0], device=q.device, dtype=torch.int32)
        index = row.expand(1, 2, 3, 5).contiguous()
        counts = torch.tensor([2, 5, 0], device=q.device, dtype=torch.int32).expand(1, 2, 3)
        sizes = _cached_block_sizes(k.shape[1], k.device)
        self.assertEqual(sizes.tolist(), [64, 64, 64, 64, 1])

        query, key, value = (tensor[0].cpu().float() for tensor in (q, k, v))
        expected = torch.zeros_like(query)
        for block, count in enumerate((2, 5)):
            tokens = (row[:count].cpu()[:, None] * 64 + torch.arange(64)).flatten()
            tokens = tokens[tokens < key.shape[0]]
            q_slice = slice(block * 64, (block + 1) * 64)
            scores = torch.einsum("qhd,khd->hqk", query[q_slice], key[tokens]) * 0.125
            expected[q_slice] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value[tokens])

        actual = bsa.hcu_sparse_attention(q, k, v, index, 5, 0.125, counts)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual[0].cpu().float(), expected, rtol=0.03, atol=0.03)
