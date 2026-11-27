import unittest
from unittest.mock import Mock, patch

import torch
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

from hcu_sglang.models.deepseek.current_v4.flat_kv_dequant import dequantize_sparse_prefill_rows


class TestFlatKVDequant(unittest.TestCase):
    def test_page_and_layout_routing_preserves_destination_slice(self):
        buf = torch.zeros(32, dtype=torch.bfloat16)
        idx = torch.tensor([1, 0])
        dest = torch.empty(3, 1, 512, dtype=torch.bfloat16)[1:]
        cases = [(size, KVLayout.V4, True, True) for size in (2, 64, 256)]
        cases += [(size, KVLayout.V4, True, False) for size in (1, 32, 128)]
        cases += [(64, KVLayout.V41, True, False), (64, KVLayout.V4, False, False)]
        for size, layout, enabled, optimized in cases:
            fallback = Mock()
            with self.subTest(size=size, layout=layout, enabled=enabled), patch(
                "lightop.kvcache.dsv4_dequantize_k_cache_paged_out"
            ) as vendor:
                dequantize_sparse_prefill_rows(buf, idx, dest, size, layout,
                                              lightop=enabled, fallback=fallback)
                if optimized:
                    fallback.assert_not_called()
                    packed, tokens, output, page = vendor.call_args.args
                    self.assertEqual((packed.dtype, packed.data_ptr()), (torch.uint8, buf.data_ptr()))
                    self.assertIs(tokens, idx)
                    self.assertIs(output, dest)
                    self.assertEqual(page, size)
                else:
                    vendor.assert_not_called()
                    fallback.assert_called_once_with(buf, idx, page_size=size, out=dest, layout=layout)


if __name__ == "__main__":
    unittest.main()
