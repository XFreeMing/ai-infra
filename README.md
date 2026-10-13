# AI Infra 研究工作台

最后更新：2026-10-13

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

也可以选择 `sources/sglang`、`sources/llama.cpp`、`sources/ds4`，或省略路径初始化全部。源码版本由主仓库提交固定。

## 当前研究

当前动手入口：[在 Mac 上调试一个小模型](experiments/tiny-inference/README.md)。TinyStories 15M 约 61 MB，搭配 llama2.c 的单个 C 推理文件，已在本机 CPU 上运行并验证 LLDB 断点。每次 10–20 分钟，从 token 与生成循环开始；[实测记录](reports/tiny-inference/README.md)保留来源与原始输出。

从零开始可以先读[源码学习路线](research/learning/README.md)与[第 1 课：一次请求如何变成模型输出](research/learning/01-request-to-tokens.md)。它们以已有后端经验为起点，每次解释一个流程、阅读少量代码，并用小练习检查理解。

[ds4 / DwarfStar 源码调研](research/studies/ds4-inference/README.md)已经整理实现机制、作者性能与质量证据、与 llama.cpp / vLLM / SGLang 的比较边界，以及按每周 5–8 小时推进的[调研与实验计划](research/studies/ds4-inference/PLAN.md)。本轮完成静态源码和一手证据研究，尚未运行模型或 GPU 性能实验。

[R0 兼容性核查](research/studies/ds4-inference/R0.md)已锁定同 GGUF 候选，读取实际头部并核对模型、量化和模板；同文件具备静态比较条件，PGX 的加载、token 对齐与短生成仍待验证。

2026-10-13 用户反馈：另一台机器已开始 PGX 验证；本机继续基础学习与源码分析。运行结果尚未回传，本仓库不据此更新为验证通过。

在 PGX 准备 P0（R0–R2）的前置环境：

```sh
uv run init
```

优先复用 `/data/models` 中完整的 Flash 0731 Q2 GGUF，校验冻结的大小、含对齐 padding 的 GGUF header 与全文件 SHA256；不下载另一份主权重，不选择 `.incomplete` 或其他模型替代同文件比较。然后初始化固定版本的 ds4/llama.cpp、准备 R2 的 DSpark support、小型 CUDA kernel 探针，并串行构建两个 ds4 版本、llama.cpp 与质量 scorer。只需 Python 标准库；`uv` 用 setuptools 安装本项目的命令入口。

每次准备生成独立的 `artifacts/init/<run-id>/`，保存输入原始字节、环境、完整命令、构建日志、二进制 hash 与 `ldd` 路径。源子模块保持固定提交，旧版 ds4 在单独 clone 构建。模型权重不复制进构建目录；缺少的 support 下载到被 Git 忽略的 `artifacts/models/<HF-revision>/`。

```sh
uv run init --check-only                  # 只盘点；退出码 2，表示还未准备完
uv run init --models-dir /data/models --cuda-home /usr/local/cuda --jobs 4
uv run --offline init --offline           # uv 与 init 分别禁止联网
uv run --offline python main.py init      # 原入口仍可用
uv run init --resume artifacts/init/<run-id>  # 恢复未完成的准备，保留历史日志
```

`--build-timeout` 默认 1800 秒，全部构建共享预算；`--download-timeout` 默认 1800 秒，support 下载支持续传；Git 初始化每个子模块预算 900 秒。已有本地模型仍需完整读取约 81 GiB 以计算 SHA256。缺少编译器、toolkit、设备权限或依赖时保存 `blockers` 并返回 2；不会自动修改系统驱动、安装系统包、清空 swap 或停止其他进程。首次 `uv` 安装需要已有 setuptools 缓存或网络。

`prepared`/退出码 0 仅表示前置准备完成。实际 token 对齐、两个引擎加载同一模型、32-token 短生成、质量检查及 R2 的 rewind 触发仍需按 [R0.md](research/studies/ds4-inference/R0.md) 执行；CUDA 小探针不等于模型兼容性或性能验证。长实验先提交工作台改动，在独立且干净的主仓库 clone 中重新准备。

固定版本的 [`score_llama.cpp:187`](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/score_llama.cpp#L187) 使用旧 API `use_mmap=true`；固定 llama.cpp 已改为 [`load_mode`](https://github.com/ggml-org/llama.cpp/blob/972b5fcc19670e597f8de9bf5cacfb85752a5fff/include/llama.h#L329)。`init` 对 scorer 的固定源码 hash 做检查，在产物目录生成仅替换为 `LLAMA_LOAD_MODE_MMAP` 的副本，用锁定的 llama.cpp 头文件和库编译；manifest 保存原始/适配源码与 patch 的 hash。评分和分词代码保持原样，质量正确性仍需 P7 验证。

`--resume` 只接受 `blocked`/`interrupted` 目录；复用前检查源码提交、clean 状态和二进制 hash，完整模型仍重新校验。已就绪的准备目录不允许恢复，避免改动可能用于运行的产物。未完成的引擎构建保留原目录，需新建准备；scorer 或下载失败可直接恢复。

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
| [sources/](sources/README.md) | vLLM、SGLang、llama.cpp、ds4、llama2.c 的 submodule 与官方入口 |
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
