# sglang-dcu-plugin

DCU adaptations for SGLang **model** and **layer** modules via general plugin hooks.

## Install

From the plugin repository root:

```bash
pip install -e . --no-deps
```

## Enable

```bash
export SGLANG_PLUGINS=dcu
sglang serve --model-path <model> --log-level info
```

See `docs/setup.md` and `docs/hook_targets.md`.
