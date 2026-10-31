#!/usr/bin/env python3
"""One-shot extractor: legacy deepseek_v2_impl.py -> methods/*.py.

The monolithic impl file was removed in P3. To regenerate methods/, restore
deepseek_v2_impl.py from git history into sglang_hcu_plugin/models/deepseek/.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "sglang_hcu_plugin/models/deepseek/deepseek_v2_impl.py"
OUT = ROOT / "sglang_hcu_plugin/models/deepseek/deepseek_v2/methods"

CONST_NAMES = [
    "_is_hcu",
    "_is_hip",
    "_is_cuda",
    "_is_npu",
    "_is_cpu",
    "_is_cpu_amx_available",
    "_device_sm",
    "_is_gfx95_supported",
    "_is_sbo_enabled",
    "_use_aiter",
    "_use_aiter_gfx95",
    "_use_lightop_moe_sum_mul_add",
    "_use_fused_silu_mul_quant",
    "_use_opt_cat_decode",
    "_use_fused_mla_cat",
    "_use_fused_rmsnorm_rope",
    "_use_fused_rms_quant",
    "_rms_quant_path",
    "FORWARD_ABSORB_CORE_ATTENTION_BACKENDS",
    "dtype_mapping",
    "fused_rms_norm_rope_contiguous",
]

EXTRA_SPECS: dict[str, list[tuple[str, str, str] | tuple[str, str]]] = {
    "mlp.py": [("DeepseekV2MLP", "forward")],
    "moe.py": [
        ("DeepseekV2MoE", n)
        for n in (
            "__init__",
            "forward",
            "forward_normal",
            "forward_deepep",
            "forward_normal_dual_stream",
            "_forward_shared_experts",
        )
    ],
    "decoder.py": [
        ("DeepseekV2DecoderLayer", "__init__"),
        ("DeepseekV2DecoderLayer", "forward"),
    ],
    "attention.py": [
        ("DeepseekV2AttentionMLA", n)
        for n in (
            "__init__",
            "prepare_qkv_latent",
            "forward_absorb_prepare",
            "forward_absorb_core",
            "_set_mla_kv_buffer",
            "_get_mla_kv_buffer",
            "_get_mla_kv_buffer_from_fp8",
            "_concat_and_cast_mha_k",
            "_chunked_prefix_attn_mha",
            "forward_normal_chunked_kv_prepare",
            "forward_normal_chunked_kv_core",
            "forward_normal_one_shot_prepare",
            "forward_normal_one_shot_core",
            "forward_absorb_fused_mla_rope_prepare",
            "forward_absorb_fused_mla_rope_core",
            "forward_absorb_fused_mla_rope_cpu_prepare",
            "forward_absorb_fused_mla_rope_cpu_core",
        )
    ],
}

FILE_HEADER = '''\
"""HCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_v2 import constants as C
from sglang_hcu_plugin.models.deepseek.deepseek_v2.methods._runtime import *  # noqa: F403

'''


def transform_source(src: str) -> str:
    for name in CONST_NAMES:
        src = re.sub(rf"\b{name}\b", f"C.{name}", src)
    src = re.sub(r"^\s*global C\._use_fused_rmsnorm_rope\s*\n", "", src, flags=re.M)
    return src


def extract_method(source: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    seg = ast.get_source_segment(source, item)
                    if not seg:
                        raise RuntimeError(f"no source for {class_name}.{method_name}")
                    return transform_source(seg)
    raise RuntimeError(f"missing {class_name}.{method_name}")


def main() -> None:
    text = IMPL.read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    for filename, specs in EXTRA_SPECS.items():
        parts = [FILE_HEADER]
        seen: set[str] = set()
        for spec in specs:
            if len(spec) == 2:
                cls, method = spec
                export = method
            else:
                cls, method, export = spec
            if export in seen:
                continue
            seen.add(export)
            body = extract_method(text, cls, method)
            if export != method:
                body = body.replace(f"def {method}(", f"def {export}(", 1)
            parts.append(body)
            parts.append("\n\n")
        (OUT / filename).write_text("".join(parts), encoding="utf-8")
        print(f"wrote {filename} ({len(seen)} methods)")

    init = '''\
"""Extracted HCU fork method bodies for DeepSeek V2 plugin hooks."""

from sglang_hcu_plugin.models.deepseek.deepseek_v2.methods import (
    attention,
    decoder,
    mlp,
    moe,
)

__all__ = ["attention", "decoder", "mlp", "moe"]
'''
    (OUT / "__init__.py").write_text(init, encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
