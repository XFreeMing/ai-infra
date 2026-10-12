# 实验定义

最后更新：2026-10-12

每个实验围绕一个可检验的问题，允许多次运行。实验定义与实际运行记录分开保存，避免用最新参数覆盖旧结果。

```sh
python main.py new experiment prefix-cache-boundary
```

生成 `experiments/prefix-cache-boundary/README.md` 和 `experiment.json`。前者说明假设、源码依据、变量控制、资源预算、正确性检查及停止条件；后者保存机器可读取的执行配置。生成后仍是草案，`command` 默认为空，需要填写后再运行。

## 执行配置

下面的配置示例需要先创建对应的 `run.py`，并完成目标源码的安装与环境检查：

```json
{
  "id": "prefix-cache-boundary",
  "command": ["{python}", "experiments/prefix-cache-boundary/run.py"],
  "timeout_seconds": 3600,
  "required_sources": ["sources/vllm"]
}
```

| 字段 | 含义 |
| --- | --- |
| `id` | 与实验目录 slug 一致 |
| `command` | 非空字符串数组，首项为程序，后续为参数；独立的 `{python}` 参数展开为当前 CLI 的 Python 路径；不使用 shell 拼接 |
| `timeout_seconds` | 正数，超时后终止运行；先通过试跑估算 |
| `required_sources` | 本次运行依赖的 submodule 路径；未依赖上游源码时为空数组 |

`experiment.json` 必须是主仓库跟踪的普通文件。提交全部实验定义，并在干净工作区执行：

```sh
uv run --offline python main.py run prefix-cache-boundary
```

执行工作目录为主仓库根目录。可用 `--output-dir <path>` 改变产物根目录；目录必须位于仓库外或已被 Git 忽略，每次运行仍创建唯一子目录。CLI 检查所需 submodule 及其嵌套 submodule 已初始化、commit 与各自父仓库 gitlink 一致且工作区干净，但不会安装框架、确认实际导入路径、自动连到 PGX 或代替质量检查。

执行器通过 `AI_INFRA_RUN_DIR` 提供本次运行目录的绝对路径，通过 `AI_INFRA_RUN_ID` 提供唯一编号。实验脚本将 CSV、JSON 或 trace 写入该目录，避免多次运行混在一起。例如，Python 脚本可用 `Path(os.environ["AI_INFRA_RUN_DIR"])` 获取目录（需导入 `os` 和 `pathlib.Path`）。配置中的命令应保持前台运行，由执行器管理生命周期。

运行快照不支持用 `skip-worktree` 或 `assume-unchanged` 隐藏已跟踪文件的修改；主仓库及所需源码中存在这些标记时会拒绝运行。使用完整、干净的实验 clone，并把实际执行的修改提交后再运行。

运行目录保存环境与版本等执行信息的 `manifest.json`，以及 `stdout.log`、`stderr.log`。CLI 输出运行目录路径，并传递实验退出码；超时返回 124，Ctrl+C 中断返回 130，SIGTERM 终止返回 143。将经过检查的必要证据整理进入 `reports/`，原始运行目录默认不提交。

## 流程自检

```sh
uv run --offline python main.py run smoke
```

内置 `smoke` 仅验证命令执行与证据保存流程，无需 GPU 或初始化源码；它不测试框架兼容性或推理性能。

实验代码、配置和分析脚本放在本实验目录；上游源码放在 `sources/` 的 submodule 中。先在小输入上确认加载的源码、输出质量和采集方式，再开始 GB10 长任务。

运行时固定主仓库与 submodule 提交，保留原始结果。大型日志、性能 trace、模型权重和缓存放在 Git 外；轻量数据和报告按唯一运行编号放在 `reports/<experiment>/<run-id>/`。

采用[实验模板](../templates/experiment.md)，按[双机流程](../docs/workflow.md)交接，再按[成果标准](../reports/README.md)复盘。初始化仓库不包含已完成的 GPU 性能实验。
