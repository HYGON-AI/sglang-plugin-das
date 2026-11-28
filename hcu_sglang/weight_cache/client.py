# SPDX-License-Identifier: Apache-2.0
import os
import stat
import time
from typing import Optional

import torch
import torch.nn as nn
from sglang.srt.weight_cache import ipc_loader as upstream
from sglang.srt.weight_cache.ipc_loader import (
    TORCH_IPC_BACKEND, _initialize_model, compute_env_stamp, current_platform, get_exec,
    get_parallel, get_socket_path, hash_quant_config, logger, recv_msg, send_msg,
)
from .protocol import CacheConfig, check_ipc_quant_support, collect_layout_env, compute_weight_layout_stamp
from .transport import get_client_transport_backend


class ChannelwiseIpcLoader(upstream.IpcModelLoader):
    def load_model(
        self,
        *,
        model_config,
        device_config,
    ) -> nn.Module:
        """Load model weights from the weight cache daemon.

        In daemon mode, raises RuntimeError if the daemon is unavailable
        (fallback to disk loading would cause OOM on shared GPUs).
        In client mode, falls back to DefaultModelLoader.
        """
        self.preloaded_weights_bytes = 0
        tic = time.perf_counter()

        # Hard-gate unsupported quant methods before touching the daemon, so an
        # unsupported model fails explicitly instead of silently disk-loading
        # (client mode) or serving wrong-numerics IPC weights. Checked here so
        # it applies regardless of whether the daemon is reachable.
        quant_method, engine_quant_config = self._resolve_engine_quant(model_config)
        check_ipc_quant_support(quant_method, engine_quant_config, where="client")

        # Try to fetch state from daemon
        cache_data = self._fetch_from_cache(model_config, device_config)

        if cache_data is None:
            if self.weight_cache_mode == "daemon":
                raise RuntimeError(
                    f"[IpcModelLoader] Weight cache daemon not available at "
                    f"{self.socket_path}. In daemon mode, fallback to disk "
                    f"loading is disabled because the daemon process already "
                    f"holds weights on the same GPU — loading from disk would "
                    f"cause OOM. Please ensure the weight cache daemon is "
                    f"running and the config matches."
                )
            logger.warning(
                "[IpcModelLoader] Weight cache not available or config mismatch, "
                "falling back to disk load"
            )
            return self._fallback_load(model_config, device_config)

        entries = cache_data["entries"]
        # Older daemons omit this field; missing metadata means no correction.
        preloaded_weights_bytes = cache_data.get("preloaded_weights_bytes", 0)
        if preloaded_weights_bytes is None:
            preloaded_weights_bytes = 0
        if (
            isinstance(preloaded_weights_bytes, bool)
            or not isinstance(preloaded_weights_bytes, int)
            or preloaded_weights_bytes < 0
        ):
            raise RuntimeError(
                "[IpcModelLoader] Daemon returned invalid weight-memory metadata: "
                f"{preloaded_weights_bytes=}"
            )
        logger.info(
            f"[IpcModelLoader] Fetched {len(entries)} tensors from daemon "
            f"(transport={self._transport_backend.name}) "
            f"in {time.perf_counter() - tic:.2f}s"
        )

        from sglang.srt.model_loader.loader import (
            _get_quantization_config,
        )

        quant_config = _get_quantization_config(model_config, self.load_config)

        extra_state = cache_data["extra_state"]
        model = self._load_zero_copy_mode(
            model_config,
            device_config,
            entries,
            quant_config,
            host_state=(extra_state or {}).get("host"),
            extra_state=extra_state,
        )
        self.preloaded_weights_bytes = preloaded_weights_bytes
        self._apply_extra_state(model, extra_state)

        # Skip _post_load_weights: the daemon already ran
        # process_weights_after_loading on the weights before exporting
        # IPC handles. Running it again would double-process (e.g.,
        # re-quantize already-quantized weights), corrupting tensor data.

        # Rebuild stale tensor views. Some modules store tensor views as
        # plain attributes (not parameters/buffers) during __init__. When
        # the model is initialized on meta device and then weights are
        # replaced via IPC mapping, these views still point to the old
        # meta storage. We must recreate them from the now-valid tensors.
        self._rebuild_stale_views(model)

        # The model now points into the daemon's GPU memory via CUDA IPC. If the
        # daemon dies, those pointers dangle, so watch it and fail loud.
        self._start_daemon_liveness_watchdog(cache_data.get("pid"))

        logger.info(
            f"[IpcModelLoader] Loaded model via IPC (mode={self.weight_cache_mode}), "
            f"total={time.perf_counter() - tic:.2f}s"
        )

        return model.eval()

    def _apply_extra_state(self, model, extra_state) -> None:
        """Reproduce the daemon's load-time state that is not in state_dict.

        Order matters: flags and derived tensors first, then the model's own
        per-process fix-up, which reads those flags to skip the one-shot weight
        rewrites the daemon already applied to the shared memory.
        """
        from .extra_state import apply_extra_state, verify_deferred_state

        result = None
        if extra_state:
            daemon_env = extra_state.get("env") or {}
            local_env = collect_layout_env()
            diff = {
                k: (daemon_env.get(k), local_env.get(k))
                for k in sorted(set(daemon_env) | set(local_env))
                if daemon_env.get(k) != local_env.get(k)
            }
            if diff:
                logger.warning(
                    "[IpcModelLoader] Environment differs from the weight cache "
                    "daemon's (daemon, engine). If one of these selects a weight "
                    "layout, the mapped weights do not match this engine: %s",
                    diff,
                )
            result = apply_extra_state(
                model, extra_state, self._transport_backend.import_extra_tensor
            )
            if result.keep_alive:
                model._ipc_imported_extra_tensors = result.keep_alive

        hook = getattr(model, "post_ipc_weight_load", None)
        if hook is not None:
            hook()
        self._check_deferred_aliases(model)
        if result is not None:
            verify_deferred_state(model, result)

    @staticmethod
    def _has_parent_module(model, name: str) -> bool:
        obj = model
        for part in name.split(".")[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                return False
        return True

    def _check_deferred_aliases(self, model) -> None:
        deferred = getattr(self, "_deferred_alias_entries", None) or {}
        if not deferred:
            return
        broken = []
        for name, entry in deferred.items():
            obj = model
            for part in name.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if (
                not isinstance(obj, torch.Tensor)
                or obj.device.type == "meta"
                or list(obj.shape) != list(entry["shape"])
                or str(obj.dtype).removeprefix("torch.") != entry["dtype"]
                or isinstance(obj, nn.Parameter) != entry["is_param"]
            ):
                broken.append(name)
        if broken:
            raise RuntimeError(
                f"[IpcModelLoader] {len(broken)} of {len(deferred)} tensor name(s) "
                f"the daemon exported through post-load module links were not "
                f"reproduced on this engine (the model needs a "
                f"post_ipc_weight_load hook that rebuilds those links): "
                f"{broken[:10]}"
            )
        logger.info(
            f"[IpcModelLoader] {len(deferred)} aliased tensor name(s) resolved "
            f"through rebuilt module links"
        )
        self._deferred_alias_entries = {}

    def _load_zero_copy_mode(
        self,
        model_config,
        device_config,
        entries,
        quant_config,
        host_state=None,
        extra_state=None,
    ) -> nn.Module:
        """Zero-copy load: map IPC tensors directly as param.data.

        The model is initialized on the meta device (no memory allocation),
        then each parameter's data is replaced with the IPC-mapped GPU tensor.
        The engine and daemon share the same physical GPU memory via CUDA IPC.
        """
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        # Initialize model on meta device to avoid any GPU/CPU memory allocation.
        # This creates the model structure with the correct parameter shapes/dtypes
        # but without allocating actual storage.
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

        # Build lookup dicts of existing parameter/buffer names in the
        # meta-device model. Post-quantization parameters (e.g. weight_scale
        # from FP8) are created by process_weights_after_loading, which the
        # daemon already ran. These params exist in the daemon's entries but
        # NOT in the meta-device model — we must register them as new attrs.
        # Use dicts (not sets) so we can do O(1) shape/dtype validation
        # without re-traversing the model tree on every lookup.
        # remove_duplicate=False mirrors the daemon's export (which keys tied
        # weights under every name) so a tied parameter is recognized under all
        # of its names here too.
        existing_params = {
            name: param
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        existing_buffers = {name: buf for name, buf in model.named_buffers()}
        existing_names = set(existing_params) | set(existing_buffers)

        imported_refs = []
        imported_count = 0
        mismatched = []
        new_params_count = 0
        map_tic = time.perf_counter()

        # Iterate over ALL daemon entries (not just model params/buffers).
        # This ensures post-quantization parameters (weight_scale, etc.)
        # that were created by process_weights_after_loading are also mapped.
        # Names that run through a module link the model only creates after
        # loading (e.g. a layer registering its successor as a submodule) have
        # no parent yet. They alias tensors mapped under their primary name,
        # so they are checked, not mapped, once the model has rebuilt its links.
        self._deferred_alias_entries = {}
        quant_method, _ = self._resolve_engine_quant(model_config)
        adopt_layout = quant_method == "compressed-tensors"
        adopted_count = 0
        # id(meta buffer) -> mapped tensor, to re-point the other modules that
        # share one buffer object (it is exported once, under its first name).
        replaced_buffers = {}
        for name, entry in entries.items():
            if not self._has_parent_module(model, name):
                self._deferred_alias_entries[name] = entry
                continue
            imported_tensor = self._transport_backend.import_tensor(entry)
            is_param = entry.get("is_param", True)

            if name in existing_names:
                # Existing parameter/buffer — validate shape/dtype
                if name in existing_params:
                    ref_param = existing_params[name]
                else:
                    ref_param = existing_buffers[name]
                layout_differs = (
                    imported_tensor.shape != ref_param.shape
                    or imported_tensor.dtype != ref_param.dtype
                )
                if layout_differs and adopt_layout:
                    # The daemon's post-processed layout; see
                    # IPC_QUANT_POSTPROCESSED_LAYOUT.
                    adopted_count += 1
                elif layout_differs:
                    mismatched.append(
                        f"  {name}: IPC={imported_tensor.shape}/{imported_tensor.dtype} "
                        f"vs model={ref_param.shape}/{ref_param.dtype}"
                    )
                    del imported_tensor
                    continue

            if name in existing_buffers:
                replaced_buffers[id(existing_buffers[name])] = imported_tensor

            # Replace or register the tensor in the model
            self._set_module_tensor(model, name, imported_tensor, is_param=is_param)
            imported_refs.append(imported_tensor)
            imported_count += 1

            if name not in existing_names:
                new_params_count += 1

        if mismatched:
            raise RuntimeError(
                f"[IpcModelLoader] {len(mismatched)} tensor(s) have shape/dtype "
                f"mismatch between the IPC daemon and the meta-initialized model. "
                f"The quantization method passed the IPC allowlist gate "
                f"(check_ipc_quant_support), so this is NOT an unsupported-quant "
                f"case — it indicates the daemon's weight fingerprint is "
                f"incomplete or the daemon/client configs drifted (a bug to fix), "
                f"not merely uninitialized weights:\n" + "\n".join(mismatched)
            )

        for name, buf in list(model.named_buffers(remove_duplicate=False)):
            if buf.device.type == "meta" and id(buf) in replaced_buffers:
                self._set_module_tensor(
                    model, name, replaced_buffers[id(buf)], is_param=False
                )

        # Weights the daemon keeps in shareable host memory are attached by
        # their modules, not mapped as device tensors.
        from .extra_state import attach_host_state, canonical_modules

        host_modules = attach_host_state(model, host_state)
        if host_modules:
            logger.info(
                f"[IpcModelLoader] Attached host-memory state of {host_modules} "
                f"module(s) from the daemon"
            )

        # Post-processing schemes on the adopt_layout allowlist may repack
        # canonical parameters into runtime buffers and delete the originals
        # (e.g. HCU DeepEP MoE deleting w13_weight/w2_weight after building
        # w13_weight_deepgemm/w2_weight_deepgemm). Drop only meta tensors that
        # the daemon's own post-processed module explicitly removed.
        module_tensors = (extra_state or {}).get("module_tensors")
        removed_count = 0
        if adopt_layout and isinstance(module_tensors, dict):
            for module_name, module in canonical_modules(model):
                kept = module_tensors.get(module_name)
                if kept is None:
                    continue
                kept_set = set(kept)
                for attr, param in list(module._parameters.items()):
                    if (
                        param is not None
                        and param.device.type == "meta"
                        and attr not in kept_set
                    ):
                        delattr(module, attr)
                        removed_count += 1
                for attr, buf in list(module._buffers.items()):
                    if (
                        buf is not None
                        and buf.device.type == "meta"
                        and attr not in kept_set
                    ):
                        delattr(module, attr)
                        removed_count += 1
        if removed_count:
            logger.info(
                f"[IpcModelLoader] Removed {removed_count} meta tensor(s) deleted "
                f"by the daemon's post-load repacking"
            )

        # After mapping every daemon entry, any tensor still on the meta device
        # is one the daemon did NOT provide. Filling it with torch.empty() would
        # hand the model uninitialized GPU memory — silently producing wrong
        # output, the worst failure mode for a load path. Hard-error and list the
        # offenders instead.
        #
        # The daemon exports the full state_dict AND non-persistent buffers
        # (e.g. rotary embedding cos_sin_cache), so a correct setup leaves nothing
        # on meta here. A non-empty list means the daemon's export is incomplete,
        # or the model has a genuinely-recomputable buffer that must be recomputed
        # explicitly (not filled with garbage) — add that handling here if needed.
        still_on_meta_params = [
            name
            for name, param in model.named_parameters()
            if param.device.type == "meta"
        ]
        still_on_meta_buffers = [
            name for name, buf in model.named_buffers() if buf.device.type == "meta"
        ]

        if still_on_meta_params or still_on_meta_buffers:
            raise RuntimeError(
                f"[IpcModelLoader] After IPC mapping, "
                f"{len(still_on_meta_params)} parameter(s) and "
                f"{len(still_on_meta_buffers)} buffer(s) remain on the meta device "
                f"— the daemon did not export them. Refusing to fill them with "
                f"uninitialized memory, which would silently produce wrong output. "
                f"This means the daemon's export is incomplete, or a recomputable "
                f"buffer needs explicit recompute logic here.\n"
                f"  params: {still_on_meta_params[:10]}"
                f"{'...' if len(still_on_meta_params) > 10 else ''}\n"
                f"  buffers: {still_on_meta_buffers[:10]}"
                f"{'...' if len(still_on_meta_buffers) > 10 else ''}"
            )

        map_elapsed = time.perf_counter() - map_tic

        # Stash IPC refs on the model to prevent GC (which would unmap the memory)
        if imported_refs:
            model._ipc_imported_tensors = imported_refs
        # Keep transport backend alive for the model lifetime (VMM backend owns
        # VA mappings that must stay mapped while tensors are in use).
        model._weight_cache_transport_backend = self._transport_backend

        logger.info(
            f"[IpcModelLoader] Zero-copy: mapped {imported_count} tensors "
            f"({new_params_count} new post-quant, {adopted_count} in the daemon's "
            f"post-processed layout), time={map_elapsed:.3f}s"
        )

        return model

    def _fetch_from_cache(self, model_config, device_config) -> Optional[dict]:
        """Connect to daemon, validate config, fetch IPC handles.

        Returns the daemon response dict on success, None if the daemon is
        genuinely absent (socket file doesn't exist). Raises on all other
        failures so they are never silently swallowed as a disk-load fallback.
        """
        import socket as socket_mod

        if self.socket_path is None:
            device_uuid = current_platform.get_device_uuid(int(device_config.gpu_id))
            self.socket_path = get_socket_path(device_uuid)

        # Only connect to a real socket node owned by us: reject a symlink, a
        # plain file, or another user's socket planted at this /tmp path. An
        # absent socket means no daemon -> fall back to disk (return None).
        try:
            st = os.lstat(self.socket_path)
        except FileNotFoundError:
            logger.info(
                f"[IpcModelLoader] Daemon socket not found at {self.socket_path}."
            )
            return None
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
            raise RuntimeError(
                f"[IpcModelLoader] Refusing to connect: {self.socket_path} is not "
                f"a socket owned by this user."
            )

        sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        try:
            sock.settimeout(30)
            sock.connect(self.socket_path)
        except FileNotFoundError:
            # Raced: socket removed between lstat and connect -> treat as absent.
            sock.close()
            return None
        except ConnectionRefusedError:
            sock.close()
            raise RuntimeError(
                f"[IpcModelLoader] Daemon socket exists at {self.socket_path} but "
                f"refused the connection. The daemon may have crashed after "
                f"creating the socket. Check daemon logs."
            )
        except Exception as e:
            sock.close()
            raise RuntimeError(
                f"[IpcModelLoader] Failed to connect to daemon at "
                f"{self.socket_path}: {e}"
            ) from e

        try:
            # Build engine's config fingerprint
            parallel = get_parallel()
            tp_size = parallel.tp_size
            tp_rank = parallel.tp_rank

            pp_size = parallel.pp_size
            pp_rank = parallel.pp_rank

            ep_size = parallel.moe_ep_size
            # The configured width, as the daemon states it: the MoE
            # communicator is wider when it aliases attention CP.
            moe_dp_size = parallel.moe_dp_size
            moe_dp_rank = parallel.moe_dp_rank
            moe_ep_rank = parallel.moe_ep_rank

            quant_method, quant_config = self._resolve_engine_quant(model_config)

            engine_config = CacheConfig(
                model_path=model_config.model_path,
                model_arch=(
                    model_config.hf_config.architectures[0]
                    if model_config.hf_config.architectures
                    else ""
                ),
                tp_size=tp_size,
                tp_rank=tp_rank,
                pp_size=pp_size,
                pp_rank=pp_rank,
                dp_size=parallel.dp_size,
                ep_size=ep_size,
                moe_dp_size=moe_dp_size,
                moe_dp_rank=moe_dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_dp_size=parallel.attn_dp_size,
                enable_dp_lm_head=parallel.enable_dp_lm_head,
                attn_cp_size=parallel.attn_cp_size,
                moe_dense_tp_size=parallel.moe_dense_tp_size,
                moe_a2a_backend=get_exec().moe.moe_a2a_backend,
                quant_method=quant_method,
                quant_config_hash=hash_quant_config(quant_config),
                dtype=str(model_config.dtype),
                revision=model_config.revision or "",
                **compute_env_stamp(),
                weight_layout_stamp=compute_weight_layout_stamp(),
            )

            logger.info(
                f"[IpcModelLoader] Requesting weights from daemon at "
                f"{self.socket_path} with config: "
                f"model={engine_config.model_path}, "
                f"arch={engine_config.model_arch}, "
                f"tp={engine_config.tp_size}/{engine_config.tp_rank}, "
                f"quant={engine_config.quant_method}, "
                f"dtype={engine_config.dtype}"
            )

            send_msg(sock, {"type": "fetch_state", "config": engine_config.to_dict()})
            result = recv_msg(sock)

            if result.get("status") != "ok":
                daemon_config = result.get("daemon_config", {})
                raise RuntimeError(
                    f"[IpcModelLoader] Daemon config mismatch!\n"
                    f"  Engine config: {engine_config.to_dict()}\n"
                    f"  Daemon config: {daemon_config}"
                )

            backend_name = result.get("transport_backend", TORCH_IPC_BACKEND)
            self._transport_backend = get_client_transport_backend(backend_name)
            result = self._transport_backend.recv_fetch_state_response(sock, result)
            return result

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[IpcModelLoader] Error communicating with daemon at "
                f"{self.socket_path}: {e}"
            ) from e
        finally:
            sock.close()
