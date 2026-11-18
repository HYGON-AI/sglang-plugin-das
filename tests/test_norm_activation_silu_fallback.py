# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock, patch

from hcu_sglang.layers import norm_activation


class SiluAndMulFallbackTests(unittest.TestCase):
    def test_missing_lightop_uses_upstream_forward(self):
        marker = object()
        original = Mock(return_value=marker)
        layer = object()
        value = object()

        # ``from lightop import fuse_silu_and_mul`` raises ImportError when the
        # optional package is unavailable; a None entry makes that deterministic
        # even on hosts where lightop is installed.
        with patch.object(norm_activation, "is_hcu", return_value=True), patch.dict(
            sys.modules, {"lightop": None}
        ):
            result = norm_activation.around_silu_and_mul_forward_cuda(
                original, layer, value
            )

        self.assertIs(result, marker)
        original.assert_called_once_with(layer, value)


if __name__ == "__main__":
    unittest.main()
