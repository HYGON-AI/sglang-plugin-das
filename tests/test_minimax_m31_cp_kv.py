# SPDX-License-Identifier: Apache-2.0
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from hcu_sglang.layers.attention.minimax_m31_cp_kv import _gather_payload, _workspace


class TestMiniMaxCPKVExchange(unittest.TestCase):
    def test_rank_major_packets_restore_request_order(self):
        meta = SimpleNamespace(
            per_rank_logical_token=[5, 2],
            per_rank_actual_token=[8, 8],
            reverse_split_len=[2, 1, 1, 1, 1, 0, 1, 0],
            cp_reverse_index=[0, 4, 6, 2, 1, 5, 7, 3],
        )
        rank_tokens = ([10, 11, 20, 14, 21], [12, 13])
        counts = meta.per_rank_logical_token
        ref = torch.tensor([10, 11, 12, 13, 14, 20, 21], dtype=torch.bfloat16)
        device = torch.device("cpu")
        width = 1152
        workspace = _workspace(meta, counts, device, 0, ref.dtype, width, nullcontext)

        def inputs(tokens):
            rows = torch.tensor(tokens, dtype=ref.dtype)[:, None, None]
            return (
                rows.expand(-1, 4, 128),
                (rows + 32).expand(-1, 4, 128),
                (rows + 64).expand(-1, 1, 128),
            )

        class Gather:
            def __init__(self):
                self.sends = []

            def all_gather_into_tensor(self, raw, send):
                self.sends.append(send.clone())
                if len(self.sends) == len(counts):
                    torch.cat(self.sends, out=raw)

        for offset in (0, 1):
            group = Gather()
            for tokens in rank_tokens:
                logical = [token + offset for token in tokens]
                _gather_payload(
                    group, workspace, inputs(logical + [-1] * (8 - len(tokens))),
                    len(tokens), None, (None, None),
                )
            self.assertEqual(len(group.sends), 2)
            self.assertEqual(group.sends[1][2:].count_nonzero().item(), 0)
            restored = workspace.raw.index_select(0, workspace.reorder)
            for start, end, delta in ((0, 512, 0), (512, 1024, 32), (1024, 1152, 64)):
                expected = (ref + offset + delta)[:, None].expand(-1, end - start)
                torch.testing.assert_close(restored[:, start:end], expected)

        self.assertIs(
            workspace, _workspace(meta, counts, device, 0, ref.dtype, width, nullcontext)
        )
        self.assertIsNot(
            workspace, _workspace(meta, counts, device, 1, ref.dtype, width, nullcontext)
        )

    def test_empty_rank_still_sends_a_zero_packet(self):
        meta = SimpleNamespace(reverse_split_len=[0, 2], cp_reverse_index=[0, 1])
        workspace = _workspace(meta, [0, 2], "cpu", 0, torch.uint8, 1280, nullcontext)
        workspace.send.fill_(255)

        class Gather:
            def all_gather_into_tensor(self, raw, send):
                raw[: len(send)].copy_(send)
                raw[len(send) :].fill_(99)

        def unexpected_pack(*args):
            self.fail("Empty ranks do not invoke the native packer")

        _gather_payload(Gather(), workspace, (), 0, unexpected_pack, (None, None))
        self.assertEqual(workspace.raw[:2].count_nonzero().item(), 0)
        self.assertEqual(workspace.reorder.tolist(), [2, 3])
