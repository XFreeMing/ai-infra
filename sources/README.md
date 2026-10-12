# 上游源码与一手入口

最后更新：2026-10-13

所有供阅读和实验使用的第三方源码通过 Git submodule 引用。主仓库保存其提交版本，第三方项目的代码与许可证由各自仓库维护。

| 项目 | 源码目录 | 官方入口 | 优先阅读 |
| --- | --- | --- | --- |
| vLLM | `sources/vllm` | [仓库](https://github.com/vllm-project/vllm)、[文档](https://docs.vllm.ai/) | 发布、调度与缓存相关 PR、性能回退讨论 |
| SGLang | `sources/sglang` | [仓库](https://github.com/sgl-project/sglang)、[文档](https://docs.sglang.io/) | 缓存、推理执行、设计与测试讨论 |
| llama.cpp | `sources/llama.cpp` | [仓库](https://github.com/ggml-org/llama.cpp)、[基准工具](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) | 本地推理、量化、硬件后端与测量方法 |
| ds4 / DwarfStar | `sources/ds4` | [仓库](https://github.com/antirez/ds4)、[已固定版本的 Spark 指南](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/DGX_SPARK.md) | 少数模型的专用推理、混合低比特量化、投机解码、会话复用与 SSD 专家缓存 |
| CUDA / NVIDIA 工具 | 按需加入具体开源组件 | [CUDA 文档](https://docs.nvidia.com/cuda/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/) | 与实验环境对应的发布说明、构建与性能分析文档 |

CUDA 文档入口不代表整个 CUDA Toolkit 的源码可作为一个 submodule 获取。需要阅读具体开源组件时，单独确定其官方仓库与许可证。

首个真实研究专题：[ds4 的实现差异、性能与效果证据、横向对照及计划](../research/studies/ds4-inference/README.md)。ds4 初始固定为 `fc80bd695da76ee14bcb86820bf900a5cdfda806`，本机额外获取了最近 20 个祖先提交用于分析真实变更；未切换源码版本。

## 初始化与版本约定

初次 clone 可以只拉主仓库，再选择所需框架：

```sh
git submodule update --init --recursive sources/vllm
git submodule status --recursive
```

全量获取：

```sh
git submodule update --init --recursive
```

这会检出主仓库记录的提交。不要将更新到上游最新分支混入实验初始化步骤。升级由研究任务明确发起，并提交新的引用。

初始化时本机的源码采用浅克隆节省时间与磁盘。若 `git log` 缺少历史，在需要追溯作者或提交演变时，先检查并按需补全：

```sh
git -C sources/vllm rev-parse --is-shallow-repository
git -C sources/vllm fetch --unshallow
```

仅在第一个命令返回 `true` 时执行 `--unshallow`。它下载历史，不需要切换正在研究的源码提交。

记录资料时保存原始链接、作者、发布时间、读取日期、适用版本和当时的讨论状态。源码引用使用完整 commit 的永久链接；需要依赖文档细节时，优先选择固定版本文档并注明版本。

修改 submodule 中的代码，需要先提交并推送到可访问的子仓库；仅提交主仓库无法同步未提交的子仓库改动。详见[双机工作流](../docs/workflow.md)。
