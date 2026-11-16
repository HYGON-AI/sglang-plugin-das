# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
from pathlib import Path

import torch


_CLASSES = {
    "KimiK3DecoderLayer", "KimiK3DeltaAttention", "KimiK3MLAAttention",
    "KimiK3MoE", "KimiK3MLP", "ParallelLMHead",
}


def _check(value, label, *, logits=False):
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _check(item, f"{label}[{index}]", logits=logits)
        return
    if not torch.is_tensor(value) or not value.is_floating_point() or not value.numel():
        return
    bad = torch.isnan(value) | torch.isposinf(value) if logits else ~torch.isfinite(value)
    if not bool(bad.any().item()):
        return
    record = {
        "event": "kimi31_first_nonfinite", "pid": os.getpid(),
        "rank": (torch.distributed.get_rank() if torch.distributed.is_initialized()
                 else os.environ.get("RANK")), "label": label,
        "shape": list(value.shape), "dtype": str(value.dtype),
        "nan": int(torch.isnan(value).sum().item()),
        "posinf": int(torch.isposinf(value).sum().item()),
        "neginf": int(torch.isneginf(value).sum().item()),
    }
    if value.ndim:
        rows = bad.reshape(value.shape[0], -1).any(dim=-1)
        record["bad_rows"] = rows.nonzero().flatten()[:16].cpu().tolist()
    print(json.dumps(record), flush=True)
    raise RuntimeError(f"Kimi31 non-finite activation at {label}")


def _install_model(model):
    if getattr(model, "_kimi31_finite_diagnostics", False):
        return
    handles = []
    for name, module in model.named_modules():
        if type(module).__name__ not in _CLASSES:
            continue

        def before(current, args, kwargs, label=name):
            hidden = kwargs.get("hidden_states")
            try:
                _check(hidden if hidden is not None else args, f"{label}.input")
            except RuntimeError:
                _dump(current, args, kwargs, None, f"{label}.input")
                raise

        def after(current, args, kwargs, output, label=name):
            try:
                _check(output, f"{label}.output")
            except RuntimeError:
                _dump(current, args, kwargs, output, f"{label}.output")
                raise

        handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(module.register_forward_hook(after, with_kwargs=True))
    model._kimi31_finite_diagnostics = True
    model._kimi31_finite_diagnostic_handles = handles
    print(json.dumps({"event": "kimi31_finite_diagnostics_installed", "pid": os.getpid(),
                      "modules": len(handles) // 2}), flush=True)


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return {"type": type(value).__name__}


def _dump(module, args, kwargs, output, label):
    directory = os.environ.get("HCU_KIMI31_DIAG_DIR")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{os.getpid()}-{re.sub('[^A-Za-z0-9._-]', '_', label)}.pt"
    torch.save({"label": label, "module_type": type(module).__name__,
                "args": _cpu(args), "kwargs": _cpu(kwargs), "output": _cpu(output),
                "state_dict": _cpu(module.state_dict()) if module is not None else {},
                "input_snapshot_timing": "post-forward inputs may have been modified in place"}, target)
    print(json.dumps({"event": "kimi31_diagnostic_dump", "path": str(target)}), flush=True)


def _load_model(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    _install_model(self.model)
    return result


def _sample(original, self, logits_output, *args, **kwargs):
    try:
        _check(logits_output.next_token_logits, "sampler.input_logits", logits=True)
    except RuntimeError:
        _dump(None, (logits_output.next_token_logits,), {}, None, "sampler.input_logits")
        raise
    return original(self, logits_output, *args, **kwargs)


def _multinomial(original, probabilities, *args, **kwargs):
    try:
        _check(probabilities, "sampler.multinomial.probabilities")
        if probabilities.numel() and (
            bool((probabilities < 0).any().item())
            or bool((probabilities.sum(dim=-1) <= 0).any().item())
        ):
            print(json.dumps({"event": "kimi31_invalid_probability", "pid": os.getpid(),
                              "shape": list(probabilities.shape)}), flush=True)
            raise RuntimeError("Kimi31 invalid multinomial probability distribution")
    except RuntimeError:
        _dump(None, (probabilities,), {}, None, "sampler.multinomial.probabilities")
        raise
    return original(probabilities, *args, **kwargs)


def register_hooks():
    if os.environ.get("HCU_KIMI31_FINITE_DIAG") != "1":
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.load_model",
                          _load_model, HookType.AROUND)
    HookRegistry.register("sglang.srt.layers.sampler.Sampler.forward", _sample, HookType.AROUND)
    HookRegistry.register("torch.multinomial", _multinomial, HookType.AROUND)
