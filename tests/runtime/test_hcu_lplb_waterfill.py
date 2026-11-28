"""HCU compact replica probabilities and empty/padded rank participation."""
import unittest
from unittest.mock import patch

import torch

from hcu_sglang.environ import register_envs
from hcu_sglang.eplb.lplb_solver import LPLBSolver, _HostWaterfill, assert_lplb_supported_model


class TestHcuWaterfill(unittest.TestCase):
    def test_architecture(self):
        for arch in ("DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"):
            assert_lplb_supported_model(arch)
        with self.assertRaises(NotImplementedError):
            assert_lplb_supported_model("unknown")

    @unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "HCU required")
    def test_device_splits_and_padded_counts(self):
        register_envs()
        phy = torch.tensor([0, 1, 2, 1, 2, 3], device="cuda")
        mapping = torch.tensor([[0,-1,-1,-1], [1,3,-1,-1], [2,4,-1,-1], [5,-1,-1,-1]], device="cuda")
        with patch.dict("os.environ", SGLANG_LPLB_HIP_SOLVER="device", SGLANG_LPLB_HOST_MIN_PAIRS="1"):
            solver = LPLBSolver(phy, mapping, 2, logical_to_all_physical_map_num_valid=(mapping >= 0).sum(-1))
            host = _HostWaterfill(phy, mapping[:, :2], 2)
            empty = torch.empty((0, 2), dtype=torch.int32, device="cuda")
            uniform = solver.solve(empty)
            batch = torch.tensor([[0,1], [2,1], [0,2], [99,-1]], device="cuda", dtype=torch.int32)
            actual = solver.solve(batch, num_token_non_padded=torch.tensor(3, device="cuda"))
            counts = torch.tensor([2,2,2,0], dtype=torch.float32, device="cuda")
            reference = host.solve(counts, torch.empty_like(actual))
        self.assertEqual(tuple(actual.shape), (4,2))
        torch.testing.assert_close(actual.sum(1), torch.ones(4, device="cuda"))
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(uniform[1], torch.full((2,), 0.5, device="cuda"))
        self.assertEqual(solver.dispatch_map.tolist(), [[0,-1],[1,3],[2,4],[5,-1]])
