#!/usr/bin/env python3
"""Extract HCU fork methods from hcu-sglang-now deepseek_v4.py -> deepseek_v4/methods/."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT.parent
    / "hcu-sglang-now/hcu-sglang/python/sglang/srt/models/deepseek_v4.py"
)
OUT = ROOT / "sglang_hcu_plugin/models/deepseek/deepseek_v4/methods"

CONST_NAMES = [
    "_is_hcu",
    "_use_dpskv4_lightop_rmsnorm",
    "_use_fused_qnorm_rope_kv_rope_quant",
    "_use_aiter_tilelang_mhc",
    "op",
]

EXTRA_SPECS: dict[str, list[tuple[str, str] | tuple[str, str, str]]] = {
    "mqa.py": [
        ("MQALayer", n)
        for n in ("__init__", "_compute_q_b", "_forward_prepare", "forward")
    ],
    "decoder.py": [
        ("DeepseekV4DecoderLayer", n) for n in ("hc_pre", "hc_post", "forward")
    ],
    "model.py": [
        ("DeepseekV4Model", "__init__"),
        ("DeepseekV4Model", "forward"),
    ],
    "causal_lm.py": [
        ("DeepseekV4ForCausalLM", n)
        for n in ("__init__", "forward", "load_weights", "post_load_weights")
    ],
}

FILE_HEADER = '''\
"""HCU DeepSeek V4 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_v4 import constants as C
from sglang_hcu_plugin.models.deepseek.deepseek_v4.methods._runtime import *  # noqa: F403

'''


def transform_source(src: str) -> str:
    for name in CONST_NAMES:
        src = re.sub(rf"\b{name}\b", f"C.{name}", src)
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
    text = SOURCE.read_text(encoding="utf-8")
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
"""Extracted HCU fork method bodies for DeepSeek V4 plugin hooks."""

from sglang_hcu_plugin.models.deepseek.deepseek_v4.methods import (
    causal_lm,
    decoder,
    model,
    mqa,
)

__all__ = ["causal_lm", "decoder", "model", "mqa"]
'''
    (OUT / "__init__.py").write_text(init, encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
