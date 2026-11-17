# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.


def agree_pd_hidden_capture_layers(
    model_runner, layer_ids, cpu_group, cp_size, torch, *, cache_consensus=False
):
    """Reuse a consensus only after every CP rank confirms unchanged input."""
    dist = torch.distributed
    use_cache = (
        cache_consensus
        and cpu_group is not None
        and dist.get_backend(cpu_group) == "gloo"
    )
    if use_cache:
        local = None if layer_ids is None else tuple(layer_ids)
        cached = getattr(model_runner, "_pd_hidden_capture_consensus", None)
        unchanged = (
            cached is not None
            and cached[0] is cpu_group
            and cached[1] == cp_size
            and cached[2] == local
        )
        changed = torch.tensor([not unchanged], dtype=torch.int32, device="cpu")
        dist.all_reduce(changed, op=dist.ReduceOp.MAX, group=cpu_group)
        if not changed.item():
            agreed = cached[3]
            return layer_ids if agreed is None else list(agreed)
        # A failed consensus must not leave an adoptable cached result.
        model_runner._pd_hidden_capture_consensus = None

    gathered = [None] * cp_size
    dist.all_gather_object(gathered, layer_ids, group=cpu_group)
    nonempty = [[int(x) for x in ids] for ids in gathered if ids]
    agreed = None
    if nonempty:
        agreed = nonempty[0]
        if any(ids != agreed for ids in nonempty[1:]):
            raise RuntimeError(
                "PD hidden capture layers disagree across prefill CP ranks: "
                f"{gathered}"
            )
    if use_cache:
        model_runner._pd_hidden_capture_consensus = (
            cpu_group, cp_size, local, None if agreed is None else tuple(agreed)
        )
    return layer_ids if agreed is None else list(agreed)
