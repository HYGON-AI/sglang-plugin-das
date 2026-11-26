from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hcu_sglang.multimodal_gen import minimax_h3


class MiniMaxH3RopeTests(unittest.TestCase):
    def test_dedicated_rope_resolves_native_op_or_package_export(self):
        q, k = (Mock(is_cuda=True, shape=(4, 2, 128)) for _ in range(2))
        cache, positions = Mock(), Mock()
        original = Mock()
        for has_native_op in (True, False):
            with self.subTest(has_native_op=has_native_op):
                native, exported = Mock(), Mock()
                namespace = SimpleNamespace(
                    minimax_h3_rotary_embedding=native if has_native_op else None
                )
                package = SimpleNamespace(minimax_h3_rotary_embedding=exported)
                with (
                    patch.object(minimax_h3, "is_hcu", return_value=True),
                    patch("torch.ops.sgl_kernel", namespace),
                    patch.dict("sys.modules", {"sgl_kernel": package}),
                ):
                    result = minimax_h3.around_apply_rope_qk(original, q, k, cache, positions)
                self.assertEqual(result, (q, k))
                kernel, unused = (native, exported) if has_native_op else (exported, native)
                kernel.assert_called_once_with(
                    positions, q.view.return_value, k.view.return_value, 128, cache, True
                )
                unused.assert_not_called()
        original.assert_not_called()
