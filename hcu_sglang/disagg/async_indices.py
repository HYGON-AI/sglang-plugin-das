# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import torch


class PendingTransferIndices:
    def __init__(self, host, event, source):
        self.host = host
        self.event = event
        self.source = source

    def __len__(self):
        return len(self.host)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("Pending transfer indices support positional slices only")
        return type(self)(self.host[key], self.event, self.source)

    def resolve(self):
        self.event.synchronize()
        return self.host.numpy()


def defer_page_indices(kv_indices, page_size):
    pages = (kv_indices[::page_size] // page_size).contiguous()
    if not pages.is_cuda:
        return pages.numpy()
    host = torch.empty_like(pages, device="cpu", pin_memory=True)
    host.copy_(pages, non_blocking=True)
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(pages.device))
    return PendingTransferIndices(host, event, pages)


def resolve_transfer_indices(value):
    if isinstance(value, PendingTransferIndices):
        return value.resolve()
    if isinstance(value, list):
        return [resolve_transfer_indices(item) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_transfer_indices(item) for item in value)
    return value
