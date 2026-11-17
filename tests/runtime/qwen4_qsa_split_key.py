"""Key-split QSA prefill logits: the final partial row block must stay exact.

The kernel grid is ``ceildiv(rows, block_q)``, so the last CTA covers rows past
``rows-1`` whenever the caller's row count is not a multiple of ``block_q=32``.
The meta reduction over that block polls ``Starts``/``Ends``, so this file pins
the row-clamped load on the real device: the partial block must score exactly
the same interval as the rows it actually owns, and must not write a single
column outside its rows' ``[start, end)`` ranges.

CPU collection stays free of tilelang and torch device work; the kernel cases
are gated on the explicit GPU lane.
"""

import os
import unittest

import torch

GPU = os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1" and torch.cuda.is_available()


def _reference(q, k, starts, ends):
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(-1) / (128**0.5)
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    outside = (columns < starts.reshape(-1, 1)) | (columns >= ends.reshape(-1, 1))
    return logits.masked_fill(outside, -float("inf"))


@unittest.skipUnless(GPU, "explicit GPU lane")
class SplitKeyPartialBlockTests(unittest.TestCase):
    def _check(self, rows, keys, key_span):
        from hcu_sglang.models.qwen.qsa_mqa_split_key import prefill

        generator = torch.Generator(device="cuda").manual_seed(7)
        q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        k = torch.randn(keys, 1, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        starts = torch.full((rows,), keys - keys // 10, device="cuda", dtype=torch.int32)
        starts += torch.arange(rows, device="cuda", dtype=torch.int32) % 97
        ends = torch.full((rows,), keys, device="cuda", dtype=torch.int32)
        ends -= torch.arange(rows, device="cuda", dtype=torch.int32) % 31
        self.assertLess(int(starts.max()), int(ends.min()))

        actual = prefill(q, k, starts, ends, key_span=key_span)
        torch.testing.assert_close(actual, _reference(q, k, starts, ends), atol=1e-4, rtol=1e-4)

        columns = torch.arange(keys, device="cuda").unsqueeze(0)
        inside = (columns >= starts.reshape(-1, 1)) & (columns < ends.reshape(-1, 1))
        self.assertEqual(int(torch.isfinite(actual).sum()), int(inside.sum()))

    def test_rows_below_one_block(self):
        self._check(17, 16384, 1024)

    def test_rows_one_past_a_block(self):
        self._check(33, 16384, 4096)

    def test_rows_unaligned_with_key_span_split(self):
        self._check(65, 32768, 2048)

    def test_aligned_rows_stay_unchanged(self):
        self._check(64, 16384, 4096)


@unittest.skipUnless(GPU, "explicit GPU lane")
class SplitKeyRowClampSourceTests(unittest.TestCase):
    """The clamp is what keeps the partial block from starting at key 0, and it
    is invisible in the numbers once the mask kernel has run, so pin the emitted
    row index itself."""

    def test_starts_and_ends_loads_are_row_clamped(self):
        from hcu_sglang.models.qwen.qsa_mqa_split_key import split_key_kernel

        source = split_key_kernel(1024).get_kernel_source()
        for tensor in ("Starts[", "Ends["):
            loads = [line for line in source.splitlines() if tensor in line]
            self.assertTrue(loads, tensor)
            for line in loads:
                self.assertIn("min(", line, f"{tensor} load is not clamped: {line}")
                self.assertIn("rows) - ", line, f"{tensor} load is not clamped by rows: {line}")


if __name__ == "__main__":
    unittest.main()
