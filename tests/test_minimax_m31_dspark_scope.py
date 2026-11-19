# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from hcu_sglang.models import minimax_m31_dspark


class TestMiniMaxDSparkCommitScope(unittest.TestCase):
    def _run_commit(self, *, hisparse, mode):
        active = False
        entries = []

        @contextmanager
        def draft_context():
            nonlocal active
            if active:
                raise AssertionError("Draft TP scope entered twice")
            active = True
            entries.append("draft")
            try:
                yield
            finally:
                active = False

        def commit_hidden(*, commit_lens):
            self.assertTrue(active)
            return commit_lens

        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForCausalLM"]),
        )
        worker = SimpleNamespace(
            _hosts_draft=True,
            _draft_context=draft_context,
            _verify_executor=SimpleNamespace(commit_hidden=commit_hidden),
            model_runner=SimpleNamespace(model_config=model_config),
        )
        runtime = SimpleNamespace(
            get_disagg=lambda: SimpleNamespace(disaggregation_mode=mode),
            get_memory=lambda: SimpleNamespace(enable_hisparse=hisparse),
            get_parallel=lambda: SimpleNamespace(attn_dp_enabled=True),
        )
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": runtime}),
            patch.object(minimax_m31_dspark, "_minimax_hip_dspark", return_value=True),
        ):
            minimax_m31_dspark._scope_verify_hidden_commit(None, worker)
            if hisparse:
                with draft_context():
                    result = worker._verify_executor.commit_hidden(commit_lens=3)
            else:
                result = worker._verify_executor.commit_hidden(commit_lens=3)

        self.assertEqual(result, 3)
        self.assertEqual(entries, ["draft"])
        self.assertFalse(active)
        self.assertEqual(worker._verify_executor.commit_hidden is commit_hidden, hisparse)

    def test_ifb_enters_draft_scope_around_commit(self):
        self._run_commit(hisparse=False, mode="null")

    def test_pd_decode_without_hisparse_uses_draft_scope(self):
        self._run_commit(hisparse=False, mode="decode")

    def test_hisparse_commit_uses_its_existing_scope(self):
        for mode in ("null", "decode"):
            with self.subTest(mode=mode):
                self._run_commit(hisparse=True, mode=mode)
