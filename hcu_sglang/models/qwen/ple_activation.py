from __future__ import annotations

import functools
import ast
import hashlib
import inspect
import logging
import textwrap
import types


logger = logging.getLogger(__name__)
_SOURCE_SHA256 = "9340ba26cbb2b12ec431bb85a456ca3ff33885e86eab200534a724d7512c38ab"


class _PrivateFunctional:
    def __init__(self, functional):
        self.functional = functional

    def __getattr__(self, name):
        return getattr(self.functional, name)

    def silu(self, input, inplace=False):
        if inplace:
            raise RuntimeError("PLE activation upstream inplace contract changed")
        return self.functional.silu(input, inplace=True)


@functools.lru_cache(maxsize=None)
def _build(original):
    if not isinstance(original, types.FunctionType) or original.__closure__:
        raise RuntimeError("PLE activation requires the verified community method")
    source = textwrap.dedent(inspect.getsource(original)).strip()
    syntax = ast.dump(ast.parse(source), include_attributes=False)
    if hashlib.sha256(syntax.encode()).hexdigest() != _SOURCE_SHA256:
        raise RuntimeError("PLE activation community method changed; review temporary ownership")
    namespace = dict(original.__globals__)
    namespace["F"] = _PrivateFunctional(namespace["F"])
    result = types.FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    logger.info("Qwen4 PLE prefill inplace activation active: private temporary, community conv/state path")
    return result


def around_short_conv(original, layer, x, forward_batch, batch):
    from sglang.srt.environ import envs

    if not envs.SGLANG_QWEN4_PLE_PREFILL_INPLACE_SILU.get():
        return original(layer, x, forward_batch, batch)
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_parallel

    if (x.device.type != "cuda" or torch.is_grad_enabled() or x.shape[0] == 0
            or forward_batch.forward_mode != ForwardMode.EXTEND
            or batch.use_decode_fast_path or get_parallel().attn_cp_size != 1):
        return original(layer, x, forward_batch, batch)
    return _build(original)(layer, x, forward_batch, batch)
