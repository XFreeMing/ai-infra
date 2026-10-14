# AI Infra 工作台 Codemap

最后更新：2026-10-13

入口：[main.py](../../main.py)、[lab_tools/cli.py](../../lab_tools/cli.py)、[README.md](../../README.md)

## 架构

```text
main.py
  └─ lab_tools/cli.py
      ├─ doctor            → 环境 / Git / submodule 状态
      ├─ init              → prepare.py / native_prepare.py → artifacts/init/<run-id>/
      ├─ new study         → templates/reading.md → research/studies/<slug>/
      ├─ new experiment    → templates/experiment.md → experiments/<slug>/
      └─ run               → experiment.json → 子进程 → artifacts/runs/<run-id>/

sources/*（Git submodule） → 研究与实验的版本依据
artifacts/（本机产物）     → 人工审阅与整理 → reports/（轻量证据与报告）

uv run ds4 / llama → serve_ds4.py / serve_llama.py（顶部配置）
                   → server_launcher.py → prepared CUDA 二进制（前台服务）
uv run sglang / vllm / freetoken → serve_*.py（顶部配置）
                     → native_launcher.py → 独立 CUDA venv（前台服务）
手动 ModelScope 下载 → scripts/download_deepseek_v41_flash_nvfp4.py → /data/models/
```

## 关键区域

| 区域 | 作用 | 输入 / 输出 | 依赖 |
| --- | --- | --- | --- |
| [main.py](../../main.py) | CLI 入口 | 命令参数 | `lab_tools.cli` |
| [lab_tools/cli.py](../../lab_tools/cli.py) | 环境检查、记录创建、实验执行 | 实验配置、JSON / Markdown / 日志 | Python 标准库、Git |
| [lab_tools/prepare.py](../../lab_tools/prepare.py) | ds4 P0 模型校验、CUDA 探针、独立构建 | R0_MODEL/R0_INPUTS、本地模型 / manifest、二进制与日志 | 标准库、Git、CUDA、make、CMake |
| [服务入口](../../lab_tools/server_launcher.py) | 解析准备好的服务二进制，设置 CUDA 环境并在前台启动 | 顶部模型/参数、prepared manifest / launch.json、服务进程 | 标准库、Git、ds4-server、llama-server |
| [原生框架准备](../../lab_tools/native_prepare.py) | 固定源码、隔离 venv、CUDA 扩展构建与实际 import probe | gitlink、构建参数 / 日志、freeze、扩展 hash | uv、Git、CUDA、各框架独立依赖 |
| [原生服务入口](../../lab_tools/native_launcher.py) | 本地分片、CUDA、容量与来源检查后前台启动 | 顶部配置、模型目录 / launch.json | 标准库、框架独立 venv |
| [FreeToken 入口](../../lab_tools/serve_freetoken.py) | 本地 Qwen BF16 配置、ft serve、6666 端口 | 独立 kernel cache、Qwen 模型清单 / 前台服务 | native_prepare/native_launcher、freetoken_model 的量化准入检查 |
| [手动模型下载](../../scripts/download_deepseek_v41_flash_nvfp4.py) | 用户显式下载指定 ModelScope revision 并核对全部 SHA256 | NATIVE_MODEL / 本地权重及下载回执 | ModelScope SDK（只由此脚本安装使用） |
| [templates/](../../templates/reading.md) | 统一记录结构 | 研究、人物、实验、报告 | 无 |
| [research/](../../research/README.md) | 研究知识与下一步入口 | 一手来源、源码证据、假设 | 来源与源码永久链接 |
| [experiments/](../../experiments/README.md) | 可执行实验定义 | README、JSON、实验脚本 | 按实验单独配置环境 |
| [tiny-inference](../../experiments/tiny-inference/README.md) | 小模型准备、运行记录与 LLDB 练习 | 固定模型、构建记录、运行结果 | Python 标准库、clang、curl、llama2.c submodule |
| [sources/](../../sources/README.md) | 固定上游源码 | vLLM、SGLang、llama.cpp、ds4 | Git submodule |
| [reports/](../../reports/README.md) | 可公开验证的成果 | 轻量原始数据、报告 | 运行清单与分析脚本 |

## 数据流

Mac 使用模板形成研究卡和实验定义，提交后在 PGX 的独立 clone 检出相同主仓库与 submodule 版本。执行器记录一次运行，实验脚本产生领域数据。结果经独立分支回流 Mac，整理成能够追溯原始证据的报告。

工作台本身没有数据库、网络服务或后台调度器，没有第三方 Python 运行依赖。各上游框架拥有独立依赖与构建要求，其版本以 submodule gitlink 为准。

## 相关文档

- [双机工作流](../workflow.md)
- [AI 工作约定](../../AGENTS.md)
- [实验配置](../../experiments/README.md)
- [成果标准](../../reports/README.md)
