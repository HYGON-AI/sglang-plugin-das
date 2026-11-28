# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from hcu_sglang.multimodal_gen import minimax_h3, minimax_h3_dit_qknorm_rope, qknorm_rope


class MiniMaxH3QKNormDispatchTests(unittest.TestCase):
    def test_qk_norm_uses_upstream_probe_on_gfx936(self):
        query = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda:2"))
        key = SimpleNamespace(dtype=torch.bfloat16)
        q_norm, k_norm = Mock(), Mock()
        original = Mock()
        target = "sglang.kernels.ops.layernorm.norm"
        probe = Mock()
        with (
            patch.object(minimax_h3, "is_hcu", return_value=True),
            patch.object(minimax_h3, "get_gcn_arch") as arch,
            patch.dict("sys.modules", {target: SimpleNamespace(can_use_fused_inplace_qknorm=probe)}),
        ):
            arch.return_value = "gfx936:sramecc+"
            probe.return_value = True
            result = minimax_h3.around_apply_qk_norm(original, query, key, q_norm, k_norm, 128)
            self.assertIs(result, original.return_value)
            probe.assert_called_once_with(128, torch.bfloat16)
            arch.assert_called_once_with(2)
            original.assert_called_once_with(query, key, q_norm, k_norm, 128)

            original.reset_mock()
            probe.return_value = False
            result = minimax_h3.around_apply_qk_norm(original, query, key, q_norm, k_norm, 128)
            self.assertEqual(result, (q_norm.return_value, k_norm.return_value))
            original.assert_not_called()

            probe.reset_mock()
            arch.return_value = "gfx938"
            result = minimax_h3.around_apply_qk_norm(original, query, key, q_norm, k_norm, 128)
            self.assertEqual(result, (q_norm.return_value, k_norm.return_value))
            probe.assert_not_called()
            original.assert_not_called()



class MiniMaxH3QKNormRoPE4HeadsTests(unittest.TestCase):
    def test_dispatch_is_scoped_and_falls_back_to_the_wrapped_kernel(self):
        attention = SimpleNamespace(_use_fused_qknorm_rope=True)
        original_kernel = Mock()
        args = tuple(object() for _ in range(6))
        kwargs = {"is_neox": True, "round_norm_before_rope": True}

        def run_kernel():
            return qknorm_rope.around_fused_inplace_qknorm_rope(
                original_kernel, *args, **kwargs
            )

        def attend(_attention):
            return run_kernel()

        with (
            patch.dict("os.environ", {qknorm_rope._H3_4HEADS_ENV: "0"}),
            patch.object(
                minimax_h3_dit_qknorm_rope, "try_fused_inplace_qknorm_rope_4heads"
            ) as fused,
            torch.no_grad(),
        ):
            self.assertIs(
                qknorm_rope.around_minimax_h3_attend(attend, attention),
                original_kernel.return_value,
            )
            fused.assert_not_called()
            original_kernel.assert_called_once_with(*args, **kwargs)
            original_kernel.reset_mock()

            with patch.dict("os.environ", {qknorm_rope._H3_4HEADS_ENV: "1"}):
                fused.return_value = True
                self.assertIsNone(qknorm_rope.around_minimax_h3_attend(attend, attention))
                fused.assert_called_once_with(*args, **kwargs)
                original_kernel.assert_not_called()

                self.assertIs(run_kernel(), original_kernel.return_value)
                fused.assert_called_once()
                original_kernel.reset_mock()

                fused.return_value = False
                self.assertIs(
                    qknorm_rope.around_minimax_h3_attend(attend, attention),
                    original_kernel.return_value,
                )
                original_kernel.assert_called_once_with(*args, **kwargs)

                fused.side_effect = RuntimeError("launch failed")
                with self.assertRaisesRegex(RuntimeError, "launch failed"):
                    qknorm_rope.around_minimax_h3_attend(attend, attention)
                self.assertIs(run_kernel(), original_kernel.return_value)

    def test_packed_stride_grid_and_unsupported_contract(self):
        module = minimax_h3_dit_qknorm_rope
        packed = torch.empty((3, 3, 5, 128), dtype=torch.bfloat16)
        q = Mock(wraps=packed[:, 0], is_cuda=True, ndim=3)
        k = Mock(wraps=packed[:, 1], is_cuda=True)
        for view, tensor in ((q, packed[:, 0]), (k, packed[:, 1])):
            view.shape, view.device, view.dtype = tensor.shape, tensor.device, tensor.dtype
        weight = torch.ones(128, dtype=torch.bfloat16)
        cache = torch.empty((3, 96), dtype=torch.bfloat16)
        positions = torch.arange(3, dtype=torch.int64)
        args = (q, k, weight, weight, cache, positions)
        kwargs = {"is_neox": True, "round_norm_before_rope": True, "eps": 1e-6}
        with (
            patch.object(module, "is_hcu", return_value=True),
            patch.object(module, "get_gcn_arch", return_value="gfx936:sramecc+") as arch,
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(
                module, "_minimax_h3_dit_qknorm_rope_4heads_kernel", new_callable=MagicMock
            ) as kernel,
        ):
            self.assertTrue(module.try_fused_inplace_qknorm_rope_4heads(*args, **kwargs))
            kernel.__getitem__.assert_called_once_with((6,))
            kernel.__getitem__.return_value.assert_called_once_with(
                *args, 1920, 1920, 128, 128, 5, 1e-6,
                num_warps=1, enable_fp_fusion=False,
            )
            kernel.reset_mock()
            self.assertFalse(
                module.try_fused_inplace_qknorm_rope_4heads(
                    *args, **{**kwargs, "cache_has_full_width": True}
                )
            )
            arch.return_value = "gfx938"
            self.assertFalse(module.try_fused_inplace_qknorm_rope_4heads(*args, **kwargs))
            kernel.__getitem__.assert_not_called()


if __name__ == '__main__':
    unittest.main()
