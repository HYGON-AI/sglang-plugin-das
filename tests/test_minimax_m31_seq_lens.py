# SPDX-License-Identifier: Apache-2.0
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hcu_sglang.models import minimax_m31_seq_lens as seq


class TestMiniMaxSeqLens(unittest.TestCase):

    def test_publish_records_event_after_copy_and_capture_invalidates_mirror(self):
        calls = []
        device_lengths = SimpleNamespace(is_cuda=True)
        mirror = seq._HostMirror(
            SimpleNamespace(copy_=lambda src, *, non_blocking: calls.append(("copy", src, non_blocking))),
            SimpleNamespace(record=lambda: calls.append("record")),
            ready=True,
        )
        relay = SimpleNamespace(
            new_seq_lens_buf=device_lengths,
            needs_confidence_relay=False,
            needs_cpu_seq_lens=True,
            _m31_async_seq_lens=True,
            _m31_seq_lens_mirror=mirror,
        )
        indices, lengths = torch.tensor([1]), torch.tensor([10])

        def original(future_map, slots, new_lengths, confidence):
            self.assertIs(future_map, relay)
            calls.append("publish")

        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            seq._publish(original, relay, indices, lengths)
        self.assertEqual(calls, ["publish", ("copy", device_lengths, True), "record"])
        self.assertTrue(mirror.ready)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            seq._publish(original, relay, indices, lengths)
        self.assertFalse(mirror.ready)
        self.assertEqual(calls[-1], "publish")

    def test_deferred_read_precedes_verify_window_and_survives_isolation(self):
        calls = []
        event = SimpleNamespace(synchronize=lambda: calls.append("wait_d2h"))
        relay = SimpleNamespace(
            new_seq_lens_buf=torch.tensor([3, 17, 29]),
            _publish_fresh=True,
            _m31_seq_lens_mirror=seq._HostMirror(torch.tensor([3, 17, 29]), event, ready=True),
        )
        scheduler = SimpleNamespace(
            future_map=relay, enable_overlap=True, _confidence_budget_prepare=None
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([0, 0]),
            seq_lens_cpu=torch.tensor([0, 0]),
            seq_lens_sum=0,
            req_pool_indices_cpu=torch.tensor([2, 1]),
            spec_info=SimpleNamespace(future_indices=torch.tensor([2, 1])),
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            is_extend_in_batch=False,
            prefill_input_ids_cpu=None,
        )

        @contextmanager
        def isolate(scheduler, batch, *, overlap):
            torch.testing.assert_close(batch.seq_lens, torch.tensor([29, 17]))
            original = batch.seq_lens_cpu, batch.seq_lens_sum
            try:
                yield
            finally:
                batch.seq_lens_cpu, batch.seq_lens_sum = original

        def decode(worker, batch):
            calls.append("build_verify_window")
            torch.testing.assert_close(batch.seq_lens_cpu, torch.tensor([29, 17]))
            self.assertEqual(batch.seq_lens_sum, 46)
            return "ok"

        def run(scheduler, batch):
            self.assertTrue(seq.defer_resolution(relay, batch, False))
            self.assertFalse(seq.defer_resolution(relay, batch, False))
            calls.append("scheduler_prepared_inputs")
            with seq._forward_isolation(isolate, scheduler, batch, overlap=True):
                return seq._forward_decode(decode, None, batch)

        with patch.object(seq, "_enabled", return_value=True):
            self.assertEqual(seq._run_batch(run, scheduler, batch), "ok")
        self.assertEqual(calls, ["scheduler_prepared_inputs", "wait_d2h", "build_verify_window"])
        self.assertEqual(batch.seq_lens_sum, 46)
        self.assertIsNone(seq._decode_step.get())
