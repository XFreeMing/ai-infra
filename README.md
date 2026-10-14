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

本项目的 Python 工具集中在 [`lab_tools/`](lab_tools/)，负责环境准备、实验记录和服务启动。

uv 默认 PyPI index 已配置为清华 TUNA：`https://pypi.tuna.tsinghua.edu.cn/simple`，设置位于 [pyproject.toml](pyproject.toml)。`init` 在独立上游 clone 中执行的依赖安装也显式传入此默认源，启动计划和日志会显示 `--default-index`；需要换源时修改项目配置中的 default index。手动 ModelScope 下载脚本的独立 uv 环境也配置了同一镜像。PyTorch CUDA 包仍通过 `--torch-backend cu130` 选择官方 CUDA 专用源，其大型 wheel 不保证由这个 PyPI 镜像加速。此配置影响 Python 包安装，模型权重仍由原来的手动下载脚本处理。[TUNA 官方说明](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)、[uv 脚本 index 配置](https://docs.astral.sh/uv/guides/scripts/#using-alternative-package-indexes)

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

`init` 默认显示每步的用途、完整命令、工作目录、超时设置和日志绝对路径。下载/编译的 stdout 与 stderr 实时显示，同时保存完整原始日志；Python 子进程启用无缓冲输出。步骤运行期间每 20 秒显示耗时、距上次新增输出的时间和「不限时」；设置有限超时时则显示剩余预算。结束时显示状态、耗时和退出码。例如 `freetoken-build-deps` 会明确说明正在安装 PyTorch/CUDA 构建依赖，不下载模型权重，并显示上游工具报告的 `Downloading` / `Downloaded` 信息。下载百分比与速度以工具实际输出为准。

```sh
uv run init --check-only                  # 只盘点；退出码 2，表示还未准备完
uv run init --models-dir /data/models --cuda-home /usr/local/cuda --jobs 4
uv run --offline init --offline           # uv 与 init 分别禁止联网
uv run --offline python main.py init      # 原入口仍可用
uv run init --resume artifacts/init/<run-id>  # 恢复未完成的准备，保留历史日志
```

初始化的依赖安装、源码获取、构建与 DSpark support 下载默认不限时，持续等待完成；可随时用 Ctrl+C 中断并保留日志。`--build-timeout 0` / `--download-timeout 0` 也表示不限时；需要有限预算时可显式设置秒数，例如 `--build-timeout 7200`。不限时不会忽略安装/编译错误，失败仍保存 `blockers` 并返回 2。已有本地模型仍需完整读取约 81 GiB 以计算 SHA256；support 下载支持续传。首次 `uv` 安装需要已有 setuptools 缓存或网络。不会自动修改系统驱动、安装系统包、清空 swap 或停止其他进程。

`prepared`/退出码 0 仅表示前置准备完成。实际 token 对齐、两个引擎加载同一模型、32-token 短生成、质量检查及 R2 的 rewind 触发仍需按 [R0.md](research/studies/ds4-inference/R0.md) 执行；CUDA 小探针不等于模型兼容性或性能验证。长实验先提交工作台改动，在独立且干净的主仓库 clone 中重新准备。

固定版本的 [`score_llama.cpp:187`](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/score_llama.cpp#L187) 使用旧 API `use_mmap=true`；固定 llama.cpp 已改为 [`load_mode`](https://github.com/ggml-org/llama.cpp/blob/972b5fcc19670e597f8de9bf5cacfb85752a5fff/include/llama.h#L329)。`init` 对 scorer 的固定源码 hash 做检查，在产物目录生成仅替换为 `LLAMA_LOAD_MODE_MMAP` 的副本，用锁定的 llama.cpp 头文件和库编译；manifest 保存原始/适配源码与 patch 的 hash。评分和分词代码保持原样，质量正确性仍需 P7 验证。

ds4/llama P0 的 `--resume` 只接受 `blocked`/`interrupted` 目录；复用前检查源码提交、clean 状态和二进制 hash，完整模型仍重新校验。已就绪的准备目录不允许恢复，避免改动可能用于运行的产物。P0 未完成的引擎构建保留原目录，需新建准备；scorer 或下载失败可直接恢复。SGLang/vLLM/FreeToken 的源码构建可在原目录续跑，命令见下方原生框架说明。

## 启动 ds4 / llama.cpp 服务

完成 `uv run init` 后，在项目目录启动其中一个服务：

```sh
uv run ds4                 # 0.0.0.0:1111，显式 --cuda
# Ctrl+C 退出 ds4，再启动 llama：
uv run llama               # 0.0.0.0:2222，--gpu-layers all --fit off
```

两者默认加载同一份约 81 GiB 的 Flash 0731 Q2，单 GB10 上按顺序启动，避免同时驻留两份大模型。命令在前台运行，Ctrl+C 直接交给服务进程；加载与请求日志显示在终端。`artifacts/servers/<launch-id>/launch.json` 保存启动请求、实际 argv、GPU 环境变量、二进制 hash、模型路径/stat 和配置文件 hash；这份记录不代表服务 ready 或推理已验证。

修改脚本顶部变量即可调整后续配置：

| 文件 | 常用变量 | 初始值 |
| --- | --- | --- |
| [serve_ds4.py](lab_tools/serve_ds4.py) | `MODEL_PATH`、`CONTEXT_SIZE`、`HOST`、`PORT`、`PREFILL_CHUNK`、`MAX_OUTPUT_TOKENS` | context 4096，端口 1111，prefill chunk 4096，默认输出上限 512 |
| [serve_llama.py](lab_tools/serve_llama.py) | `MODEL_PATH`、`CONTEXT_SIZE`、`HOST`、`PORT`、`BATCH_SIZE`、`UBATCH_SIZE`、`MMPROJ_PATH` | context 4096，端口 2222，batch 4096，ubatch 2048，单 slot，Jinja 开启 |

`SERVER_BINARY=None` 默认选取与 R0 引擎提交匹配的最新 `prepared` 构建并校验二进制 hash；`PREPARATION_DIR` 可固定某次构建目录，`SERVER_BINARY` 可指定自己的绝对路径。换模型时修改 `MODEL_PATH`；本次文本 GGUF 不需要 `mmproj`，视觉模型则在 llama 脚本设置匹配的 `MMPROJ_PATH`，ds4 使用 `VISION_PATH`。

当前是 NVIDIA GB10，两个入口设置 `CUDA_VISIBLE_DEVICES=0`。参考命令中的 `HIP_VISIBLE_DEVICES` 对应 ROCm。模型 metadata 声明 context 上限 1,048,576；本次按 [P0 准入计划](research/studies/ds4-inference/R0.md) 从 `4096` 开始，加载、32-token 短生成与内存检查通过后再提高。llama 禁用自动 fit，保留显式 context/offload 请求；实际 GPU 层分配仍以启动日志为准。

```sh
uv run ds4 --dry-run        # 显示命令，检查模型/二进制路径，不加载模型
uv run llama --dry-run
uv run ds4 --server-help    # 核验配置参数并查看上游帮助，不加载模型
uv run llama --server-help
uv run llama -- --ctx-size 8192  # 临时追加上游参数；模型/context 建议长期改顶部变量
curl http://127.0.0.1:1111/v1/models
curl http://127.0.0.1:2222/v1/models
```

llama 初始 `REASONING="off"`。ds4 此版本通过请求选择 non-thinking，P0 请求使用 `"model":"deepseek-chat"` 或 `"thinking":{"type":"disabled"}`，并显式设置 `"temperature":0,"max_tokens":32`。两端模板/token 对齐与输出质量仍需 R0 验证。参数核对依据：固定 ds4 的 [server 参数解析](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_server.c#L15663)、固定 llama.cpp 的 [GPU offload 参数](https://github.com/ggml-org/llama.cpp/blob/972b5fcc19670e597f8de9bf5cacfb85752a5fff/common/arg.cpp#L2802) 与 [Jinja/reasoning 参数](https://github.com/ggml-org/llama.cpp/blob/972b5fcc19670e597f8de9bf5cacfb85752a5fff/common/arg.cpp#L3691)。

## SGLang / vLLM：初始化、手动下载与启动

原生框架默认模型为指定的 [nv-community/DeepSeek-V4.1-Flash-NVFP4](https://www.modelscope.cn/models/nv-community/DeepSeek-V4.1-Flash-NVFP4)，本地目录 `/data/models/DeepSeek-V4.1-Flash-NVFP4`。固定 revision `962b2d0e8b06e475b77ab32e694b5df51f7949fb`，85 个文件的大小/SHA256 见 [NATIVE_MODEL.json](research/studies/ds4-inference/NATIVE_MODEL.json)。这是 V4.1 原生混合精度 safetensors，routed experts 使用 NVFP4；不同于 ds4/llama 的 V4 Flash 0731 Q2 GGUF，不能直接作为同权重 P0 对照。

48 个权重分片合计 **527,293,384,576 bytes，约 491.1 GiB**，单台 GB10 的统一内存无法容纳完整权重。启动入口会在加载前阻止这种容量配置；减少 context 或在同一统一内存里 CPU offload 不能补足容量。模型卡报告的测试配置为四张 GB300；该上游结果不是本机结果。继续使用本仓库指定模型，但运行需要满足容量和 kernel 条件。[静态核查与来源](research/studies/ds4-inference/NATIVE_SERVING.md)

```sh
# 只审阅计划，不安装、构建或下载：
uv run init --engines sglang vllm --dry-run

# 以后手动执行框架初始化；不会下载模型：
uv run init --engines sglang vllm --jobs 4
# 单独准备也可：--engines sglang 或 --engines vllm
# 失败后复用源码和 venv，并保留旧 manifest 与日志：
uv run init --engines sglang --resume artifacts/init/<sglang-run-id> --jobs 4
uv run init --engines vllm --resume artifacts/init/<vllm-run-id> --jobs 4
# --engines all 包含 ds4/llama P0、SGLang、vLLM 和 FreeToken。

# 唯一主动下载新模型的入口；由用户手动运行：
uv run scripts/download_deepseek_v41_flash_nvfp4.py
# 可改 --local-dir；仅核对已有下载：追加 --verify-only
# 下载前只查看配置（不安装 ModelScope SDK）：
uv run --offline python scripts/download_deepseek_v41_flash_nvfp4.py --dry-run

# 预览启动命令；没有模型或框架时也显示缺项：
uv run sglang 3333 --dry-run
uv run vllm 5555 --dry-run
# 容量/kernel 门槛满足后再启动；端口参数可省略：
uv run sglang 3333
uv run vllm 5555
```

模型、context、TP、GPU 可见列表和批处理参数位于 [serve_sglang.py](lab_tools/serve_sglang.py) / [serve_vllm.py](lab_tools/serve_vllm.py) 顶部。默认 host `0.0.0.0`、context `4096`、TP=1、单并发、`CUDA_VISIBLE_DEVICES=0`；模型配置上限为 1,048,576。`QUANTIZATION=None` 让框架读取混合精度配置，避免错误地将全部层当普通 FP8 或 NVFP4。两端 reasoning/tokenizer/parser 名称按各自源码和模型卡设置。临时追加上游参数用 `--`，例如 `uv run vllm -- --max-model-len 8192`。

`init` 保持默认 ds4/llama P0 行为；选择原生框架后，各自从固定 gitlink clone 到新的 `artifacts/init/<run-id>/<engine>/source`，在各自的 `venv` 安装依赖并构建 CUDA 扩展。SGLang 固定源码要求 torch 2.14.1，vLLM 要求 torch 2.13.0，两套环境分别安装 CUDA 13.0 PyTorch wheel，用本机 CUDA toolkit 与 sm_121a 构建。SGLang 从当前源码的 `python/sglang/kernels/aot` 构建 sglang-kernel，再安装 editable Python package；HTTP 服务配置关闭可选 Rust 扩展。vLLM 使用 `requirements/build/cuda.txt` 及 editable 源码构建，关闭预编译扩展选择。构建可能需要下载依赖和子模块；`--offline` 禁止这些网络来源，缺项保存为 blocked。

准备流程保存预算、源码/子模块 commit、安装日志、freeze、实际 import path、torch CUDA 版本、CUDA tensor smoke 和扩展 SHA256；框架 `prepared` 不要求模型已下载，也不代表模型推理通过。启动入口选择匹配当前 gitlink 的环境，校验扩展身份、本地 config/index/分片，并再次检查 CUDA/import path。设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，不通过在线 repo ID 隐式下载权重。启动记录仍写入 `artifacts/servers/<launch-id>/launch.json`。`PYTHON_EXECUTABLE` 可指定自己的 **venv/bin/python 路径**，`PREPARATION_DIR` 可固定准备目录；`--server-help` 只查看已安装上游帮助。

初版交付只准备脚本并执行工具测试/命令预览。2026-10-14 已开始实际初始化失败恢复，排查日志和已执行检查见 [初始化恢复记录](research/studies/ds4-inference/NATIVE_SERVING.md)；完整准备结果以 run 的 manifest 为准。此模型未下载，也未执行其推理，V4.1 NVFP4 在 sm_121 上的完整 kernel 支持仍待验证。

原生初始化在同一 workspace 内串行执行，等待期间每 20 秒显示详情。uv 自身的缓存锁每次等待 3600 秒，只有明确的缓存锁超时才重试；默认总预算不限时，显式 `--build-timeout` 仍生效。`UV_LOCK_TIMEOUT=0` 在 uv 中表示立即超时，不能用来关闭超时。`--resume` 只接受相同引擎、CUDA_HOME 和源码提交的 `blocked`/`interrupted` 原生目录，拒绝修改已就绪目录。vLLM 会先检查 `elfutils/libdwfl.h`；系统缺少头文件时，将与已安装 libdw/libelf runtime 同版本的 Debian 开发包下载、解压到准备目录，记录 SHA256 并进行 C++ 头文件检查，无需 sudo。其 setuptools/CMake 编译目录保存在 `vllm/build-cache`，失败后续跑可复用对象文件。

## FreeToken：固定子模块与 CUDA 服务入口

[FreeToken](https://github.com/FlashML-org/FreeToken) 已加入 `sources/freetoken`，固定 commit `95ac86f21567bc244c7a9d91972838fbf049b4ed`。初步源码阅读和兼容性边界见 [研究记录](research/studies/freetoken-inference/README.md)。它提供 MoE expert offload、CPU/GPU 协同执行、缓存和 OpenAI/Anthropic HTTP 服务；在 GB10 上，原生 BF16 MoE 的 auto 策略选择 GPU fused resident 路径。

按用户选择，默认复用 **`/data/models/Qwen3.6-35B-A3B-BF16`**：完整的 26 个 safetensors 分片，约 67.0 GiB，BF16，无额外量化。config/index/tokenizer 的 SHA256 与分片大小保存在 [FreeToken 模型清单](research/studies/freetoken-inference/NATIVE_MODEL.json)。原始下载的远端 revision 未知，本轮没有读取全部权重核对 SHA256，也没有加载模型。

```sh
uv run init --engines freetoken --dry-run       # 查看计划，不安装、构建或下载
uv run init --engines freetoken --check-only    # 盘点本地模型；退出 2 表示尚未构建
uv run init --engines freetoken --jobs 4
# 可合并：uv run init --engines freetoken sglang vllm

uv run freetoken --dry-run
uv run freetoken 6666 --dry-run
uv run freetoken --server-help                 # 已安装环境中的 ft serve --help
uv run freetoken                               # 默认 0.0.0.0:6666，前台运行
```

[serve_freetoken.py](lab_tools/serve_freetoken.py) 顶部设置模型目录、CUDA 可见设备、host/port、context、dtype、MoE 策略和 kernel cache。默认 context `4096`（本地模型上限 262144），prefill chunk 4096，单并发，输出预算 512，memory ratio 0.80，text-only、reasoning off；需要图像输入时自行调整 `TEXT_MODEL_ONLY`。临时参数放在 `--` 后，例如 `uv run freetoken -- --max-seq-len-override 8192`。

`init --engines freetoken` 不下载模型。它从固定子模块 clone 到独立准备目录，创建专用 venv，安装 torch 2.14.1/CUDA 13.0 wheel 与上游 `[accel]` 依赖，再从源码构建 `_pinned_tensor`、`_cpu_moe`、`_row_store`。调用上游 `compile_and_package_kernels`，将其默认 AOT spec 集构建到准备目录的独立 kernel cache，设置 `TVM_FFI_CUDA_ARCH_LIST=12.1`；源码保持原样，不使用 x86_64 nightly wheel。每个 spec 顺序构建，内部编译并行度由 `--jobs` 控制。最后校验实际 import path、CUDA Tensor、FreeToken 自有 GPU store kernel 输出，以及扩展/cache 文件的 SHA256。

上游安装文档的正式要求仍写 Linux x86_64，ARM64 依赖和源构建必须实际通过；失败保存为 blocked，不能把脚本存在当作 GB10 兼容证明。默认 AOT 集不覆盖全部模型 kernel；Triton、FlashInfer 以及未覆盖的 FreeToken shape 可能在首次运行时 JIT。准备环境的 `CUDA_HOME`、kernel cache、Torch/Triton cache 路径传给启动进程；`KERNEL_CACHE_DIR` 可在脚本顶部改为自己的目录。模型加载、32-token smoke、完整内存预算和输出质量仍待执行。

此前选择的 `nv-community/DeepSeek-V4.1-Flash-NVFP4` 继续用于 SGLang/vLLM。FreeToken 此固定版本的 V4.1 reader 仅支持 MXFP4，且没有该模型的 `nvfp4_expert_spec`，因此此入口在加载前拒绝 V4.1 NVFP4。当前 GGUF 注册仅有 Gemma4，不能直接复用 ds4 的 Q2 GGUF 或现有 Qwen GGUF。这里与 ds4 P0 使用不同模型，后续测量必须分别记录。本轮只添加子模块、入口和工具测试，未安装 FreeToken、构建 CUDA 扩展或执行模型推理。

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
