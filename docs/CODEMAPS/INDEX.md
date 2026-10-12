# AI Infra 工作台 Codemap

最后更新：2026-10-13

入口：[main.py](../../main.py)、[ai_infra/cli.py](../../ai_infra/cli.py)、[README.md](../../README.md)

## 架构

```text
main.py
  └─ ai_infra/cli.py
      ├─ doctor            → 环境 / Git / submodule 状态
      ├─ new study         → templates/reading.md → research/studies/<slug>/
      ├─ new experiment    → templates/experiment.md → experiments/<slug>/
      └─ run               → experiment.json → 子进程 → artifacts/runs/<run-id>/

sources/*（Git submodule） → 研究与实验的版本依据
artifacts/（本机产物）     → 人工审阅与整理 → reports/（轻量证据与报告）
```

## 关键区域

| 区域 | 作用 | 输入 / 输出 | 依赖 |
| --- | --- | --- | --- |
| [main.py](../../main.py) | CLI 入口 | 命令参数 | `ai_infra.cli` |
| [ai_infra/cli.py](../../ai_infra/cli.py) | 环境检查、记录创建、实验执行 | 实验配置、JSON / Markdown / 日志 | Python 标准库、Git |
| [templates/](../../templates/reading.md) | 统一记录结构 | 研究、人物、实验、报告 | 无 |
| [research/](../../research/README.md) | 研究知识与下一步入口 | 一手来源、源码证据、假设 | 来源与源码永久链接 |
| [experiments/](../../experiments/README.md) | 可执行实验定义 | README、JSON、实验脚本 | 按实验单独配置环境 |
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
