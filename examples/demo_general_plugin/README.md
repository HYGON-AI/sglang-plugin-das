# Demo General Plugin

SGLang 通用插件示例，演示如何通过 Hook 在不修改主仓库代码的情况下扩展行为。

本示例位于 `dcu-sglang-plugin` 仓库；生产 DCU 适配请使用主包 `sglang-dcu-plugin`（`SGLANG_PLUGINS=dcu`）。

## 功能

| Hook 类型 | 目标 | 行为 |
|-----------|------|------|
| AROUND | `ServerArgs.__post_init__` | 打印 model_path、tp_size、attention_backend |
| REPLACE | `ServerArgs.check_server_args` | 替换校验逻辑，打印 model_path 后委托原函数 |
| AROUND | `Scheduler.__init__` | 统计 Scheduler 初始化耗时 |
| AFTER | `Scheduler.get_next_batch_to_run` | 每 10000 次调度打印一次计数 |

## 安装

在 `dcu-sglang-plugin` 仓库根目录下：

```bash
pip install -e examples/demo_general_plugin --no-deps
```

## 启用


> **Note:** 若同时安装了 `sglang-dcu-plugin`，请用 `SGLANG_PLUGINS=demo_general` 只加载本示例，
> 生产环境请用 `SGLANG_PLUGINS=dcu`，避免示例 Hook 影响服务启动。

```bash
# 只加载本插件（推荐，避免加载其他插件）
export SGLANG_PLUGINS=demo_general

# 启动 SGLang 服务
sglang serve --model-path <model> --log-level info
```

日志中应出现 `[demo_general]` 前缀的输出。

## 验证 entry point 已注册

```bash
python3 -c "
from importlib.metadata import entry_points
eps = entry_points(group='sglang.srt.plugins')
print([(e.name, e.value) for e in eps])
"
```

## 编写自己的插件

1. 复制本目录，修改 `pyproject.toml` 中的包名和 entry point 名称
2. 在 `register()` 中通过 `HookRegistry.register()` 注册 Hook
3. 或使用 `@plugin_hook` 装饰器
4. `pip install -e .` 后设置 `SGLANG_PLUGINS=<your_plugin_name>`

更多 Hook 目标见 SGLang 主仓库 `docs/platforms/plugin.md`。
