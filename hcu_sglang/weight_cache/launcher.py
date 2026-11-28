# SPDX-License-Identifier: Apache-2.0
from hcu_sglang.compat.aiter import bootstrap_aiter_compat


def bootstrap():
    bootstrap_aiter_compat()
    from sglang.srt.plugins import load_plugins
    load_plugins()


def run_worker(server_args, gpu_id, tp_rank, pp_rank, dist_init_method=None):
    bootstrap()
    from sglang.srt.weight_cache.daemon import _await_cluster_ready_to_serve
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.utils import kill_itself_when_parent_died
    from .daemon import WeightCacheDaemon

    import logging
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [Daemon gpu={gpu_id} tp_rank={tp_rank}] %(levelname)s %(message)s")
    kill_itself_when_parent_died()
    daemon = WeightCacheDaemon(server_args, gpu_id, tp_rank, pp_rank, dist_init_method)
    daemon.load()
    _await_cluster_ready_to_serve(timeout=resolving_view(server_args).weight_cache_timeout)
    daemon.serve()


def main():
    import argparse
    import logging

    bootstrap()
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.weight_cache import daemon as upstream
    parser = argparse.ArgumentParser(add_help=False)
    upstream.WeightCacheDaemonArgs.add_cli_args(parser)
    options, argv = parser.parse_known_args()
    args = prepare_server_args(argv)
    worker = upstream.WeightCacheDaemonArgs.from_cli_args(options)
    logging.basicConfig(level=logging.INFO)
    if worker.gpu_id is not None or worker.tp_rank is not None:
        gpu_id = worker.gpu_id if worker.gpu_id is not None else worker.tp_rank
        tp_rank = worker.tp_rank if worker.tp_rank is not None else worker.gpu_id
        upstream.cleanup_stale_daemon_files(upstream.current_platform.get_device_uuid(gpu_id), force=worker.force)
        run_worker(args, gpu_id, tp_rank, worker.pp_rank, worker.dist_init_method)
        return
    previous = upstream.run_weight_cache_daemon
    upstream.run_weight_cache_daemon = run_worker
    try:
        upstream.launch_weight_cache_daemons(args, dist_init_method=worker.dist_init_method,
                                             timeout=worker.timeout, force=worker.force)
    finally:
        upstream.run_weight_cache_daemon = previous
