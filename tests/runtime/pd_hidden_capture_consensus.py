"""Real two-rank CPU Gloo regression; no device initialization or model."""
import datetime
import importlib.util
import json
import pathlib
import sys
import types


def load_helper(path):
    spec = importlib.util.spec_from_file_location("consensus_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agree_pd_hidden_capture_layers


def worker(rank, directory, helper, init_url):
    root = pathlib.Path(directory)
    (root / ("progress" + str(rank) + ".txt")).write_text("before torch import")
    import torch
    import torch.distributed as dist

    root = pathlib.Path(directory)
    dist.init_process_group(
        "gloo", init_method=init_url,
        rank=rank, world_size=2, timeout=datetime.timedelta(seconds=40),
    )
    (root / ("progress" + str(rank) + ".txt")).write_text("group ready")
    agree = load_helper(helper)
    counts = {"reduce": 0, "gather": 0}

    class ObservedDist:
        ReduceOp = dist.ReduceOp
        get_backend = staticmethod(dist.get_backend)

        @staticmethod
        def all_reduce(*args, **kwargs):
            counts["reduce"] += 1
            return dist.all_reduce(*args, **kwargs)

        @staticmethod
        def all_gather_object(*args, **kwargs):
            counts["gather"] += 1
            return dist.all_gather_object(*args, **kwargs)

    observed = types.SimpleNamespace(
        distributed=ObservedDist, tensor=torch.tensor, int32=torch.int32
    )
    owner = types.SimpleNamespace()
    group = dist.new_group([0, 1], backend="gloo")
    cases = []

    def check(name, value, expected, gathers, model=owner, current_group=group, cache=True):
        (root / ("progress" + str(rank) + ".txt")).write_text(name)
        before = dict(counts)
        result = agree(model, value, current_group, 2, observed, cache_consensus=cache)
        assert result == expected, (name, result, expected)
        assert counts["gather"] - before["gather"] == gathers, (name, counts, before)
        assert counts["reduce"] - before["reduce"] == int(cache and current_group is not None)
        cases.append(name)
        return result

    try:
        check("initial_empty_consensus", None, None, 1)
        check("unchanged_empty_cache", None, None, 0)
        value = [1, 14] if rank == 0 else None
        out = check("rank0_only_input_propagates", value, [1, 14], 1)
        out.append(999)
        check("output_mutation_isolated", value, [1, 14], 0)
        value = [2, 15] if rank == 0 else None
        check("one_rank_change_invalidates_all", value, [2, 15], 1)
        check("unchanged_nonempty_cache", value, [2, 15], 0)
        try:
            agree(owner, [3 + rank], group, 2, observed, cache_consensus=True)
        except RuntimeError as error:
            assert str(error).startswith("PD hidden capture layers disagree across prefill CP ranks: ")
        else:
            raise AssertionError("mismatched ranks accepted")
        assert owner._pd_hidden_capture_consensus is None
        cases.append("mismatch_rejected_on_both_ranks")
        check("recovery_after_mismatch", [6], [6], 1)
        check("empty_list_abi", [], [], 1)
        check("unchanged_empty_list_abi", [], [], 0)
        check("new_model_invalidates", [], [], 1, model=types.SimpleNamespace())
        new_group = dist.new_group([0, 1], backend="gloo")
        check("new_group_invalidates", [], [], 1, current_group=new_group)
        check("default_group_original_fallback", None, None, 1, current_group=None)
        check("non_hcu_original_fallback", [7], [7], 1, cache=False)
        assert not torch.cuda.is_initialized()
        (root / ("rank" + str(rank) + ".json")).write_text(json.dumps({
            "passed": True, "rank": rank, "cases": cases, "counts": counts,
            "cuda_initialized": torch.cuda.is_initialized(),
            "actual_backend": dist.get_backend(group),
        }, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    root = pathlib.Path(sys.argv[1])
    helper = str(pathlib.Path(sys.argv[2]).resolve())
    import socket
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        port = handle.getsockname()[1]
    mp.spawn(worker, args=(str(root.resolve()), helper, "tcp://127.0.0.1:" + str(port)), nprocs=2, join=True)
    rows = [json.loads((root / ("rank" + str(i) + ".json")).read_bytes()) for i in range(2)]
    assert rows[0]["cases"] == rows[1]["cases"]
    result = dict(passed=True, ranks=rows, cases_per_rank=len(rows[0]["cases"]), scope="Actual CPU Gloo collectives, no HCU/native device initialization or model.")
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
