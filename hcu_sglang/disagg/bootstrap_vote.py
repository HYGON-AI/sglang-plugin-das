# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch.distributed as dist


def all_reduce_bootstrap_vote(tensor, op, group, peer_group, *, hcu):
    if (
        hcu
        and tensor.device.type == 'cpu'
        and group is not None
        and peer_group is not None
        and str(dist.get_backend(group)).lower() == 'gloo'
        and str(dist.get_backend(peer_group)).lower() == 'gloo'
        and dist.get_world_size(group) == 1
        and dist.get_world_size(peer_group) > 1
    ):
        return None
    return dist.all_reduce(tensor, op=op, group=group)
