# SPDX-License-Identifier: Apache-2.0
import os
import socket
import time
from typing import Dict, Tuple

import torch
from sglang.srt.weight_cache import daemon as upstream
from sglang.srt.weight_cache.daemon import (
    LoadConfig, SpawnRanks, compute_env_stamp, current_platform,
    get_parallel, get_quant_method_name, hash_quant_config, logger, publish,
    recv_msg, send_msg, spawn_world_rank,
)

from .protocol import CacheConfig, check_ipc_quant_support, collect_layout_env, compute_weight_layout_stamp
from .transport import TorchIpcTransportBackend


class WeightCacheDaemon(upstream.WeightCacheDaemon):
    def load(self):
        """Full loading pipeline: disk → TP shard → quantize → export IPC handles."""
        # CUDA IPC weight sharing relies on torch's _share_cuda_ handle export,
        # which only exists on CUDA-alike platforms (CUDA / ROCm). Fail loud here
        # instead of dying deep inside the export with an opaque error.
        if not current_platform.is_cuda_alike():
            raise RuntimeError(
                f"[WeightCacheDaemon] the weight cache daemon requires a CUDA-alike "
                f"platform (CUDA or ROCm) for CUDA IPC weight sharing, but the "
                f"active platform device type is {current_platform.device_type!r}. "
                f"Disable the weight cache (--weight-cache-mode off)."
            )
        # expandable_segments makes torch's caching allocator hand out memory
        # that cannot be exported via _share_cuda_, so the IPC export below would
        # die mid-way with an opaque CUDA error. Fail fast with an actionable
        # message before touching the device.
        self._assert_ipc_compatible_allocator()
        current_platform.set_device(current_platform.get_device(self.gpu_id))

        # Reduce thread contention during multi-process loading
        torch.set_num_threads(1)

        # Lazy imports to avoid circular dependencies and speed up startup
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.loader import get_model_loader

        server_args = self.server_args
        publish(
            server_args,
            role="weight_cache_daemon",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(
                    server_args, tp_rank=self.tp_rank, pp_rank=self.pp_rank
                ),
                gpu_id=self.gpu_id,
            ),
        )

        from sglang.srt.layers.moe import initialize_moe_config

        initialize_moe_config()

        # Initialize distributed backend for model loading
        # (must be done after server_args and model_config are available)
        # Build model config first, then init distributed
        # Use the same context length and model overrides as the target runner.
        model_config = ModelConfig.from_server_args(server_args)

        # Build cache config fingerprint before loading the model.
        # Loading may mutate hf_config.quantization_config (e.g. via
        # process_weights_after_loading), which would produce a different
        # hash than what the engine computes from the original config.
        # ModelConfig always exposes hf_config/quantization directly;
        # quantization_config is the only genuinely-optional attribute.
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        quant_method = get_quant_method_name(
            self.quantization or model_config.quantization
        )
        if not quant_method and quant_config is not None:
            quant_method = get_quant_method_name(quant_config)

        # Refuse unsupported quant methods before creating distributed groups
        # or touching model weights.
        check_ipc_quant_support(quant_method, quant_config, where="daemon")

        # The initialized groups are the authority for rank identity. This
        # avoids maintaining a second copy of the model-parallel hierarchy.
        self._init_distributed(server_args, model_config)
        self._initialize_eplb_expert_location_metadata(model_config)
        moe_dp_rank = get_parallel().moe_dp_rank
        moe_ep_rank = get_parallel().moe_ep_rank
        self.config = CacheConfig(
            model_path=self.model_path,
            model_arch=(
                model_config.hf_config.architectures[0]
                if model_config.hf_config.architectures
                else ""
            ),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank,
            dp_size=get_parallel().dp_size,
            # Resolved widths, as the engine states them: the raw CLI values
            # do not reflect topologies that derive EP from CP (megamoe).
            ep_size=get_parallel().moe_ep_size,
            moe_dp_size=get_parallel().moe_dp_size,
            moe_dp_rank=moe_dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_dp_size=get_parallel().attn_dp_size,
            enable_dp_lm_head=self.enable_dp_lm_head,
            attn_cp_size=self.attn_cp_size,
            moe_dense_tp_size=get_parallel().moe_dense_tp_size,
            moe_a2a_backend=self.moe_a2a_backend,
            quant_method=quant_method,
            quant_config_hash=hash_quant_config(quant_config),
            dtype=str(model_config.dtype),
            revision=self.revision or "",
            **compute_env_stamp(),
            weight_layout_stamp=compute_weight_layout_stamp(),
        )

        current_platform.empty_cache()
        memory_before_load = torch.cuda.memory_reserved(self.gpu_id)

        # Build load config
        load_config = LoadConfig(
            load_format=self.load_format,
            model_loader_extra_config=self.model_loader_extra_config,
            tp_rank=self.tp_rank,
        )

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Loading model from disk: {self.model_path}"
        )
        tic = time.perf_counter()

        # Load model using DefaultModelLoader (includes TP sharding + quant post-process)
        loader = get_model_loader(load_config=load_config, model_config=model_config)
        self.model = loader.load_model(
            model_config=model_config,
            device_config=DeviceConfig(current_platform.device_type, self.gpu_id),
        )

        elapsed = time.perf_counter() - tic
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Model loaded from disk in {elapsed:.2f}s"
        )

        # Ensure every post-processing kernel has retired before we export the
        # memory: clients map these tensors read-only via IPC and would otherwise
        # risk observing half-written weights.
        current_platform.synchronize()
        current_platform.empty_cache()
        self.preloaded_weights_bytes = max(
            0, torch.cuda.memory_reserved(self.gpu_id) - memory_before_load
        )

        # Export all parameters and buffers as IPC handles
        self._export_state()

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Exported {len(self.state_entries)} tensors as IPC handles."
        )

    def _export_state(self):
        """Export model state entries through the selected transport backend."""
        self.state_entries.clear()

        # remove_duplicate=False so tied weights are recognized as parameters
        # under every name. state_dict() below emits both tied keys, and with the
        # deduped set the duplicate would be mis-registered as a buffer, not a
        # parameter, on the client.
        param_names = set(
            name for name, _ in self.model.named_parameters(remove_duplicate=False)
        )
        state_dict_names = set(self.model.state_dict().keys())
        state_tensors: Dict[str, Tuple[torch.Tensor, bool]] = {}

        # Export all items from state_dict (parameters + persistent buffers)
        for name, tensor in self.model.state_dict().items():
            state_tensors[name] = (tensor.data, name in param_names)

        # Also export non-persistent buffers (not in state_dict but needed
        # for inference, e.g. rotary embedding cos_sin_cache)
        non_persistent_count = 0
        # remove_duplicate=False: a buffer object shared by several modules
        # (e.g. one rotary table for all layers) is needed under every name.
        for name, buf in self.model.named_buffers(remove_duplicate=False):
            if name not in state_dict_names:
                state_tensors[name] = (buf.data, False)
                non_persistent_count += 1

        from .extra_state import export_extra_state, export_host_state

        # Modules that keep weights in shareable host memory hand those out
        # themselves; their tensors must not go through device IPC.
        host_state, host_covered = export_host_state(self.model)
        for name in host_covered:
            state_tensors.pop(name, None)
        on_cpu = [n for n, (t, _) in state_tensors.items() if t.device.type == "cpu"]
        if on_cpu:
            raise RuntimeError(
                f"[WeightCacheDaemon] {len(on_cpu)} state tensor(s) live in host "
                f"memory and no module exports them (export_ipc_host_state): "
                f"{on_cpu[:10]}. Device IPC cannot share them."
            )

        self.transport_backend = TorchIpcTransportBackend()
        self.state_entries = self.transport_backend.prepare_export(state_tensors)

        self.extra_state = export_extra_state(
            self.model, self.transport_backend.export_extra_tensor
        )
        self.extra_state["host"] = host_state
        self.extra_state["env"] = collect_layout_env()
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] Extra (non state_dict) state: "
            f"{len(self.extra_state['tensors'])} tensors, "
            f"{len(self.extra_state['scalars'])} scalars, "
            f"{len(self.extra_state['opaque'])} opaque attributes, "
            f"{len(host_state)} host-memory module(s)"
        )

        # Log approximate serialized metadata size (not payload-backed bytes).
        # Only the handle blob carries real weight, so measure it directly:
        # stringifying every entry would allocate a copy of all handles.
        total_bytes = sum(
            len(handle)
            for handle in (entry.get("handle") for entry in self.state_entries.values())
            if isinstance(handle, (str, bytes, bytearray))
        )
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] "
            f"Exported {len(self.state_entries)} tensors "
            f"({non_persistent_count} non-persistent buffers), "
            f"transport={self.transport_backend.name}, "
            f"metadata size ~{total_bytes / 1024 / 1024:.1f} MB"
        )

    def _initialize_eplb_expert_location_metadata(self, model_config) -> None:
        """Build the same initial physical expert layout as the engine.

        Not gated on --enable-eplb: like ModelRunner, a static
        --init-expert-location (with redundant experts) decides which expert
        each physical slot loads even when rebalancing is off. Skipping it
        here loads a trivial placement the engine then routes against with
        the configured one.
        """
        from sglang.srt.eplb.expert_location import (
            compute_initial_expert_location_metadata,
            set_global_expert_location_metadata,
        )

        set_global_expert_location_metadata(
            compute_initial_expert_location_metadata(
                model_config=model_config,
                moe_ep_rank=get_parallel().moe_ep_rank,
            )
        )

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection."""
        req = recv_msg(conn)

        if req.get("type") == "query_config":
            # Client asks for config without requesting handles
            send_msg(conn, {"status": "ok", "config": self.config.to_dict()})

        elif req.get("type") == "fetch_state":
            # Client requests full state with IPC handles
            engine_config = CacheConfig.from_dict(req["config"])
            if not self.config.matches(engine_config):
                # Log detailed mismatch info for debugging
                daemon_dict = self.config.to_dict()
                engine_dict = engine_config.to_dict()
                mismatches = {
                    k: (daemon_dict.get(k), engine_dict.get(k))
                    for k in daemon_dict
                    if daemon_dict.get(k) != engine_dict.get(k)
                }
                logger.warning(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] "
                    f"Config mismatch: {mismatches}"
                )
                send_msg(
                    conn, {"status": "mismatch", "daemon_config": self.config.to_dict()}
                )
                return

            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"Serving {len(self.state_entries)} tensors via "
                f"{self.transport_backend.name} transport"
            )
            self.transport_backend.send_fetch_state_response(
                conn,
                config=self.config.to_dict(),
                entries=self.state_entries,
                # PID so the client can watch daemon liveness: if this
                # process dies while clients hold IPC mappings, their
                # param.data (and any CUDA-graph-captured addresses) dangle.
                pid=os.getpid(),
                preloaded_weights_bytes=self.preloaded_weights_bytes,
                extra_state=self.extra_state,
            )

        elif req.get("type") == "ping":
            send_msg(conn, {"status": "ok"})

        else:
            send_msg(
                conn,
                {
                    "status": "error",
                    "message": f"Unknown request type: {req.get('type')}",
                },
            )
