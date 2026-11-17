# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import unittest

import numpy as np
import torch

from hcu_sglang.disagg.async_indices import (
    PendingTransferIndices,
    defer_page_indices,
    resolve_transfer_indices,
)


class TransferIndicesCpuTests(unittest.TestCase):
    def test_cpu_page_indices_preserve_dtype_and_floor(self):
        for dtype in (torch.int32, torch.int64):
            value = torch.tensor([-1, 0, 3, 4, 8, 9], dtype=dtype)
            actual = defer_page_indices(value, 2)
            np.testing.assert_array_equal(actual, (value[::2] // 2).numpy())
            self.assertEqual(actual.dtype, value.numpy().dtype)

    def test_sender_metadata_does_not_wait_or_read(self):
        class Event:
            waits = 0

            def synchronize(self):
                self.waits += 1

        event = Event()
        pending = PendingTransferIndices(torch.tensor([7, 8, 9]), event, object())
        self.assertEqual(len(pending), 3)
        tail = pending[1:]
        self.assertEqual(len(tail), 2)
        self.assertEqual(event.waits, 0)
        with self.assertRaises(TypeError):
            pending[0]
        actual = resolve_transfer_indices([None, (tail,)])
        np.testing.assert_array_equal(actual[1][0], [8, 9])
        self.assertEqual(event.waits, 1)

    def test_worker_wait_error_is_preserved(self):
        error = RuntimeError("metadata copy failed")

        class Event:
            def synchronize(self):
                raise error

        pending = PendingTransferIndices(torch.tensor([1]), Event(), None)
        with self.assertRaises(RuntimeError) as caught:
            pending.resolve()
        self.assertIs(caught.exception, error)


@unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1", "explicit GPU opt-in required")
class TransferIndicesGpuTests(unittest.TestCase):
    def test_async_snapshot_survives_source_update_and_worker_thread(self):
        for dtype in (torch.int32, torch.int64):
            for page_size in (1, 64, 256):
                value = torch.arange(32768, dtype=dtype, device="cuda") + 1024
                expected = (value[::page_size] // page_size).cpu().numpy()
                pending = defer_page_indices(value, page_size)
                value.fill_(-1)
                result, errors = [], []

                def consume():
                    try:
                        result.append(pending.resolve())
                    except BaseException as error:
                        errors.append(error)

                worker = threading.Thread(target=consume)
                worker.start()
                worker.join(timeout=30)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                np.testing.assert_array_equal(result[0], expected)


if __name__ == "__main__":
    unittest.main()
