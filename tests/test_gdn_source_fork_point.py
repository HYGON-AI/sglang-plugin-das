# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pin the fork point of the vendored GDN backend.

``gdn_source.forward_decode`` is sglang's target-decode loop plus the HCU
deltas; ``forward_extend``/``_verify_attempt``/``_convolve_prefill``
are the plugin's overlay. An upstream sync changes the copied body, and these
hashes make that change show up as a failing test instead of silent drift.

Update after a reviewed sync: run this file directly, paste the new digests,
and say which upstream commit the copy now matches.

    python3 tests/test_gdn_source_fork_point.py
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[1]
          / "hcu_sglang/layers/attention/linear/gdn_source.py")

# Digests of the definitions as merged in #72 (b8ea993). The digests first
# committed with this test did not match that file on any main commit, so the
# test was red from the day it landed; re-pinned to the reviewed merge state.
PINNED = {
    "_convolve_prefill": "685e9a08dbf278b2765f528dbeec6eeb3ce66d92fb4a1cf40d939692353039b0",
    "forward_extend": "ece02c94499c87efb13b614e74d2615cc5e89fbe312952d6473f9bfb23282b67",
    "_verify_attempt": "7699ece34c2620ff555fa8f606577e5ff51aa1f3fbbfc307fb28867f856c4185",
}

# The deltas that make the copy ours; losing one of these is the regression.
MARKERS = {
    "forward_decode": "conv_update(",
    "_convolve_prefill": "initial_states=conv_states",
    "forward_extend": "_upstream_forward_extend(",
    "_verify_attempt": "_verify_plan(",
}


def _functions():
    tree = ast.parse(SOURCE.read_text())
    return {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _digest(node) -> str:
    return hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()


class ForkPointTest(unittest.TestCase):
    def setUp(self):
        self.functions = _functions()

    def test_the_copied_and_overlay_definitions_are_pinned(self):
        for name, expected in PINNED.items():
            with self.subTest(definition=name):
                self.assertIn(name, self.functions)
                self.assertEqual(_digest(self.functions[name]), expected)

    def test_the_module_imports(self):
        try:
            from hcu_sglang.layers.attention.linear import gdn_source
        except ImportError as exc:  # the pinned sglang tree is not installed
            self.skipTest(str(exc))
        self.assertTrue(hasattr(gdn_source, "_VerifyPlan"))

    def test_each_definition_keeps_its_hcu_delta(self):
        for name, marker in MARKERS.items():
            with self.subTest(definition=name):
                self.assertIn(marker, ast.unparse(self.functions[name]))


if __name__ == "__main__":
    nodes = _functions()
    for name in sorted(PINNED):
        print(f'    "{name}": "{_digest(nodes[name])}",')
