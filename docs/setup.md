# Setup

## 1. Install SGLang

Install from your local SGLang source tree (editable install recommended):

```bash
pip install -e <sglang-source>/python --no-deps
```

## 2. Install this plugin

From the plugin repository root:

```bash
pip install -e . --no-deps
```

## 3. Enable

```bash
export SGLANG_PLUGINS=dcu
sglang serve --model-path <model>
```

> **Note:** If `examples/demo_general_plugin` is also installed, set `SGLANG_PLUGINS=dcu`
> explicitly. When unset, SGLang loads **all** installed general plugins.
