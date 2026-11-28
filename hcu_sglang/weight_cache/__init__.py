# SPDX-License-Identifier: Apache-2.0
def around_ipc_load(original_fn, loader, *, model_config, device_config):
    if getattr(model_config.hf_config, "model_type", None) != "deepseek_v41":
        return original_fn(loader, model_config=model_config, device_config=device_config)
    from .client import ChannelwiseIpcLoader
    from .protocol import _hcu_channelwise_fp8_round_trips_via_ipc

    method, config = loader._resolve_engine_quant(model_config)
    if method != "compressed-tensors" or not _hcu_channelwise_fp8_round_trips_via_ipc(config):
        return original_fn(loader, model_config=model_config, device_config=device_config)
    client = ChannelwiseIpcLoader(load_config=loader.load_config, socket_path=loader.socket_path,
                                  weight_cache_mode=loader.weight_cache_mode,
                                  fallback_loader_cls=loader._fallback_loader_cls,
                                  fallback_load_format=loader._fallback_load_format)
    model = client.load_model(model_config=model_config, device_config=device_config)
    loader.preloaded_weights_bytes = client.preloaded_weights_bytes
    return model


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if is_hcu():
        HookRegistry.register("sglang.srt.weight_cache.ipc_loader.IpcModelLoader.load_model",
                              around_ipc_load, HookType.AROUND)
