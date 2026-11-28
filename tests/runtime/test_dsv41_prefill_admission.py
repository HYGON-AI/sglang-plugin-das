"""HCU overlap admission must preserve the result snapshot and middle chunks."""
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from hcu_sglang import batch_overlap
from hcu_sglang.environ import register_envs
from sglang.srt.environ import envs


class AdmissionTest(TestCase):
    def setUp(self):
        register_envs()
        self.last, self.middle, self.done, self.long = [Mock() for _ in range(4)]
        for req in (self.last, self.middle, self.done, self.long):
            req.finished.return_value = req is self.done
            req.inflight_middle_chunks = int(req is self.middle)
        self.batch = SimpleNamespace(reqs=[self.last, self.middle, self.done, self.long], batch_is_full=True)
        self.snapshot = self.batch.reqs[:]
        self.batch.filter_batch = Mock(side_effect=self.filter_batch)
        self.scheduler = SimpleNamespace(
            enable_skip_finishing_decode=True,
            server_args=SimpleNamespace(enable_prefill_cp=True, dp_size=1, pp_size=1),
            spec_algorithm=Mock(), model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
            result_queue=[(SimpleNamespace(reqs=self.snapshot, forward_mode=Mock()), None)],
            _reqs_finishing_inflight=Mock(return_value=[self.last, self.middle]),
        )
        self.scheduler.result_queue[0][0].forward_mode.is_extend.return_value = True
        self.scheduler.spec_algorithm.is_none.return_value = True
        self.original = Mock()

    def filter_batch(self, *, reqs_to_exclude):
        self.batch.reqs = [r for r in self.batch.reqs if not r.finished() and r not in reqs_to_exclude]

    def call_hook(self, enabled):
        with patch.object(batch_overlap, "is_hcu", return_value=True), patch.object(
            envs.SGLANG_DSV41_CP_PRUNE_FINISHING_PREFILL, "get", return_value=enabled
        ):
            result = batch_overlap.around_dsv41_prefill_admission(self.original, self.scheduler, self.batch)
        self.original.assert_called_once_with(self.scheduler, self.batch)
        self.assertIs(result, self.original.return_value)

    def test_final_and_finished_release_slots_but_preserve_result_snapshot(self):
        self.call_hook(True)
        self.assertEqual(self.batch.reqs, [self.middle, self.long])
        self.assertFalse(self.batch.batch_is_full)
        self.assertEqual(self.snapshot, [self.last, self.middle, self.done, self.long])

    def test_decode_result_only_prunes_already_finished(self):
        self.scheduler.result_queue[0][0].forward_mode.is_extend.return_value = False
        self.call_hook(True)
        self.assertEqual(self.batch.reqs, [self.last, self.middle, self.long])
        self.scheduler._reqs_finishing_inflight.assert_not_called()

    def test_quick_disable_preserves_batch(self):
        self.call_hook(False)
        self.batch.filter_batch.assert_not_called()
        self.assertTrue(self.batch.batch_is_full)


if __name__ == "__main__":
    main()
