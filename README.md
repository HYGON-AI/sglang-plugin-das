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

## Example plugin

Hook framework walkthrough (not required for DCU serving):

```bash
pip install -e examples/demo_general_plugin --no-deps
export SGLANG_PLUGINS=demo_general
```

See `examples/demo_general_plugin/README.md`.

When both packages are installed, always set `SGLANG_PLUGINS` explicitly (`dcu` for production).
