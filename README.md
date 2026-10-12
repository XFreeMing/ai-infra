# AI Infra 研究工作台

最后更新：2026-10-12

在 Mac 上用 AI 阅读一手资讯、技术作者讨论和固定版本源码，形成可以验证的实验假设；在 PGX GB10 上运行独立版本的实验，再回到 Mac 分析结果。Git 保存研究过程、实验定义和轻量证据，Git submodule 固定上游源码。

适合每周投入 5–8 小时持续积累。首版提供研究模板、实验模板、环境检查和本地执行记录；GB10 兼容性与 GPU 性能实验尚未验证。

```text
官方资讯 / 作者讨论 → 源码研究 → 可检验假设 → 固定 Git 提交
                          ↑                      ↓
                    Mac 分析报告 ← 实验证据 ← PGX 独立运行
                          └─ GB10 运行期间，Mac 继续下一项研究
```

## 快速开始

需要 Git、Python 3.12+。可以使用已有 Python，也可以使用 uv 管理的环境；工作台本身没有第三方 Python 运行依赖。

```sh
git clone git@github.com:XFreeMing/ai-infra.git
cd ai-infra
uv run --offline python main.py doctor
uv run --offline python main.py run smoke
```

`uv --offline` 需要本机已有满足版本要求的 Python；不用 uv 时，将 `uv run --offline python` 替换为 `python3`。`doctor` 输出环境和源码状态的 JSON；`smoke` 只检查执行与证据保存流程，不需要 GPU 或初始化上游源码，也不产生性能结论。

`run` 要求主仓库干净且实验配置已经提交，结果默认写到被 Git 忽略的 `artifacts/runs/<run-id>/`，包含 `manifest.json`、`stdout.log` 与 `stderr.log`。每次运行使用新目录；内置 smoke 还会写入 `result.json`，用于检查实验脚本能否把数据保存到正确的运行目录。

需要阅读源码时，按需下载：

```sh
git submodule update --init --recursive sources/vllm
```

也可以选择 `sources/sglang`、`sources/llama.cpp`，或省略路径初始化全部。源码版本由主仓库提交固定。

## 开始一次研究

```sh
uv run --offline python main.py new study prefix-cache-boundary
uv run --offline python main.py new experiment prefix-cache-boundary
```

1. 填写研究卡中的原始链接、人物、源码永久链接及理解，区分原作者主张、推断和验证。
2. 在实验卡中明确假设、对照、输入、指标、质量检查和资源预算。
3. 编辑 `experiment.json` 中的执行参数，把自己的脚本和分析逻辑放入实验目录。
4. 提交并推送实验定义，在 PGX 独立 clone 中检出明确的主仓库 SHA 与 submodule，再试跑与执行。
5. 将轻量原始数据和报告通过结果分支带回 Mac，注明结论边界。

完整命令与注意事项见[双机工作流](docs/workflow.md)。第一项里程碑是完成一次“源码阅读 → 小规模试跑 → 对照实验 → 可复现报告”。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | AI 阅读、修改和证据记录约定 |
| [sources/](sources/README.md) | vLLM、SGLang、llama.cpp 的 submodule 与官方入口 |
| [research/](research/README.md) | 研究卡、英文精读、人物档案 |
| [experiments/](experiments/README.md) | 假设、执行配置、实验和分析脚本 |
| [reports/](reports/README.md) | 原始小型数据、实验报告、专题成果 |
| [templates/](templates/reading.md) | 阅读、人物、实验、报告模板 |
| [docs/workflow.md](docs/workflow.md) | Mac / PGX 交接与结果回流 |
| [main.py](main.py) | 创建记录、环境检查、实验执行入口 |
| `artifacts/` | 被 Git 忽略的本机运行产物 |

架构索引见 [docs/CODEMAPS/INDEX.md](docs/CODEMAPS/INDEX.md)。资讯采集、人物检索和周报目前采用文件与模板；完成两轮实际研究后再决定需要哪些自动化。

## 可公开验证的成果

每份成果保留“原始讨论 → 源码版本 → 假设 → 环境与配置 → 原始数据 → 图表 → 结论”的对应关系。没有观察到提升、只能在部分负载复现，以及公开数据重分析都可以记录，但需要准确标明证据类型。

模型权重、缓存与大体积 trace 放在 Git 外；报告记录其版本、位置、校验值和获取方法。发布标准见 [reports/README.md](reports/README.md)。
