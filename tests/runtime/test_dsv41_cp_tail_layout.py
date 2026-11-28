"""Check the device CP tail against a row-by-row host round-robin oracle."""
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch
from hcu_sglang.environ import register_envs

register_envs()

from hcu_sglang.models.deepseek.current_v4.hcu_backend import backend


@skipUnless(torch.version.hip and torch.cuda.is_available(), "HCU required")
class TailLayoutTest(TestCase):
    def test_multiple_requests_uneven_tails_and_empty_ranks(self):
        for extends, tails in (([12, 25, 4], [3, 9, 4]), ([1, 2], [1, 1]), ([4096, 4096], [128, 128])):
            start = 0
            idx, request = [], []
            for i, (length, retained) in enumerate(zip(extends, tails)):
                idx.extend(range(start+length-retained, start+length))
                request.extend([i] * retained)
                start += length
            owners = [i % 8 for i in idx]
            counts = [owners.count(r) for r in range(8)]
            batch = SimpleNamespace(extend_seq_lens_cpu=extends, positions=torch.arange(start, device="cuda")+19, batch_size=len(tails))
            token_indices = torch.tensor(idx, device="cuda")
            for rank in range(8):
                with patch.object(backend, "get_parallel", return_value=SimpleNamespace(attn_cp_rank=rank, attn_cp_size=8)):
                    row = backend.DeepseekV4AttnBackend._late_layer_tail_cp_layout(None, batch, token_indices, tails)
                keep = [i for i, owner in enumerate(owners) if owner == rank]
                pad = max(counts) - counts[rank]
                local_lens = [sum(request[i] == j for i in keep) for j in range(len(tails))]
                self.assertEqual(row["local_lens_cpu"], local_lens)
                self.assertEqual(row["local_lens"].tolist(), local_lens)
                self.assertEqual(row["local_token_indices"].tolist(), [idx[i]//8 for i in keep])
                self.assertEqual(row["local_positions"].tolist(), [idx[i]+19 for i in keep] + [0]*pad)
                cp = row["cp_metadata"]
                first_pad = len(idx) + sum(max(counts)-c for c in counts[:rank])
                self.assertEqual(cp.local_index.tolist(), keep + list(range(first_pad,first_pad+pad)))
                self.assertEqual(cp.gather_index.tolist(), [r*max(counts)+owners[:i].count(r) for i,r in enumerate(owners)])
                self.assertEqual(cp.per_rank_logical_token, counts)

if __name__ == "__main__":
    main()
