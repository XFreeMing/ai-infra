# FreeToken：固定源码阅读与 GB10 准备入口

日期：2026-10-13。固定版本：`95ac86f21567bc244c7a9d91972838fbf049b4ed`，子模块目录 `sources/freetoken`。本次只添加源码与工作台脚本，未安装框架、构建 CUDA kernel 或执行模型推理。

## 先理解执行路径

上游声明：FreeToken 是面向本地 MoE 模型的引擎，使用 GPU expert cache、CPU/GPU 协同执行、双缓冲 prefill 和 FTW 快速加载格式；提供 OpenAI、Responses 与 Anthropic HTTP API。这里没有复述或验证其速度宣传。[固定 README](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/README.md)、[CLI 文档](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/docs/cli.md)

静态观测：入口 [`freetoken.cli.main` / `_run_serve`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/cli.py) 调用 server，再由 scheduler/engine 管理执行。`models/register.py` 按 architecture 注册 config、weight loader 与多模态 processor；`moe/expert_pieces.py` 选择专家 reader，`expert_banks.py` 将专家打包为 host banks，`offload_cache.py` 管理 GPU expert cache。普通 safetensors 可以直接加载，FTW 转换是可选步骤，不在 init 自动执行。

GB10 的关键边界：[`_is_unified_memory_gpu` / `_fused_resident_ok` / `_adjust_config`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/engine/engine.py#L1156) 检测统一内存，对具备 resident 路径的模型将 auto MoE 策略解析为 fused，以避免同一 DRAM 的 host/GPU 副本与来回拷贝。GGUF 和 DeepSeek-V4 专用专家格式有例外，不能强制套用普通 NVFP4/BF16 resident 路径。[对应模型文档](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/docs/models.md#unified-memory-gpus-gb10--dgx-spark)

## 本次模型选择

用户选择复用本地 `/data/models/hugging-Qwen3.6-35B-A3B`。本机文件元数据观测：architecture 为 `Qwen3_5MoeForConditionalGeneration`，text_config.dtype 为 `bfloat16`，context 上限 262144；26 个 safetensors 分片均存在，合计 71,903,776,776 bytes（约 67.0 GiB）。该 architecture 在固定 [`_MODEL_REGISTRY`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/models/register.py#L182) 有注册。

[NATIVE_MODEL.json](NATIVE_MODEL.json) 保存 config/index/tokenizer 的完整 SHA256 与分片大小；权重全文件 SHA256 未校验，原始下载 revision 未知，不能宣称与某个远端 revision bit-identical。分片齐全不是加载或数值正确性证明。默认 text-only，不加载 vision tower；与 ds4 Flash 0731 Q2 的结果不能作为同权重引擎排名。

静态兼容性结论：此版本不能直接使用此前选定的 `nv-community/DeepSeek-V4.1-Flash-NVFP4`。[`deepseek_v41.weight.iter_expert_pieces`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/models/deepseek_v41/weight.py#L203) 只接收 `QuantKind.MXFP4`；NVFP4 会返回 None。随后公共 [`moe.expert_pieces.iter_expert_pieces`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/moe/expert_pieces.py#L40) 寻找该 family 的 `nvfp4_expert_spec`，但 V4.1 module 没有此函数。模型 architecture 有注册，不等于每种专家量化 export 都有 reader。工作台在 CUDA/模型加载前阻止这一配置。

GGUF 同样不能按扩展名泛化：[`GGUF_ARCH_TO_REGISTRY`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/models/gguf/config.py#L20) 仅包含 Gemma4。当前 ds4 DeepSeek Q2 GGUF 和 Qwen GGUF 不能直接送给此 loader。

V4.1 容量边界另有区别：上游 [weight loader](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/models/deepseek_v41/weight.py) 与 [模型文档](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/docs/models.md#notes) 将 Engram 表交给磁盘 row store，因此原始 checkpoint 文件字节不都常驻内存。本次不将 SGLang/vLLM 的完整文件容量门槛直接套到这条路径，也不据此宣称巨大模型装得下。

## 初始化与 CUDA 证据设计

`uv run init --engines freetoken --dry-run` 显示计划。真正执行时，先 clone 固定 SHA 到独立 `artifacts/init/<run-id>/freetoken/source`，再建独立 venv。新子模块首次提交主仓库前，通过 index 的 gitlink（`:sources/freetoken`）读取固定 SHA；manifest 同时记录 parent commit、实际 source commit、固定引用位置与工作台代码 hash。长期实验仍需要干净、已提交的主仓库快照。

构建依据：[`pyproject.toml`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/pyproject.toml) 要求 torch 2.14.x，`sglang-kernel==0.4.9` 要求 torch 2.14.1，FlashInfer `[cu13]` 固定 0.6.18.post1；transformers 为 5.16.x，不能共用 SGLang/vLLM 环境。准备脚本选 torch 2.14.1 CUDA 13.0 wheel，安装源码 `[accel]`。

[`setup.py`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/setup.py) 构建 `_pinned_tensor`、`_cpu_moe` 与 Linux `_row_store`。前两者是 host C++ extension，链接 cudart；它们不是纯 GPU MoE kernel。实际 GPU kernel 来自 CUDA/TVM-FFI、Triton 与加速依赖。[`_toolchain.check_nvcc_matches_torch`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/kernel/_toolchain.py) 检查 CUDA major；本机 toolkit 13.4 与 torch cu130 的 minor 差异仍须实测。

工作台调用 [`compile_and_package_kernels`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/kernel/aot.py#L238) 构建默认 spec 集；设置 `TVM_FFI_CUDA_ARCH_LIST=12.1`，将 cache/build 放在 clone 外部的本次准备目录，避免修改上游 tracked 元数据或采用 x86_64 nightly 包。它依照上游 [`load_aot` / `load_jit`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/kernel/utils.py) 的 override 接口加载缓存。默认 AOT spec 集不覆盖全部 Triton、FlashInfer 和所有模型 shape，后续 JIT 仍可能发生。

环境 `prepared` 需要实际 import path 位于固定 clone、torch CUDA 可访问、compute capability 12.1、CUDA tensor sum 正确，以及 FreeToken `store_cache` 真正执行 CUDA 并正确写入指定 cache 行。该小探针禁用 FreeToken runtime JIT，强制使用刚构建的 cache；记录全部 extension/cache 文件的 SHA256。它不加载任何模型，不作为 GB10 推理或性能结果。

上游 [安装文档](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/docs/install.md) 仍标 Linux x86_64；[`cpu_moe_ext.cpp`](https://github.com/FlashML-org/FreeToken/blob/95ac86f21567bc244c7a9d91972838fbf049b4ed/python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp#L45) 有 x86 ISA guards 与 aarch64 路径，但这不足以证明整个 ARM64 依赖、C++、JIT 和模型链路可用。失败保留在 manifest/log，不修改上游源码绕过失败。

## 后续验证

启动配置在 [serve_freetoken.py](../../../ai_infra/serve_freetoken.py) 顶部：CUDA_VISIBLE_DEVICES=0，host 0.0.0.0，port 6666，context 4096，prefill chunk 4096，单并发，512-token 默认输出预算，memory ratio 0.80。`uv run freetoken` 在准备好的专用 venv 调用 `python -m freetoken.cli serve`，使用本地完整模型和离线 HF 模式，保存启动请求与 provenance。不要并发驻留其他大模型。

已理解：FreeToken 将专家数据位置、计算后端与统一内存条件分开处理；checkpoint architecture 支持和具体量化 reader 支持是不同准入条件。本次默认原生 Qwen BF16，可以复用本地文件，但不同于 ds4 P0 模型。

未解决：ARM64 wheel/编译链路、全部 GPU kernel、模型加载、内存峰值、32-token 输出质量与性能尚未执行；本地模型远端 revision 与全文件权重 hash 未确定。

下一步：审阅 init dry-run；按用户要求，环境初始化默认不限时，Ctrl+C 可中断，任一步失败停止并保留日志。需要有限预算时显式指定 `--build-timeout`。通过 CUDA/source provenance 后，在独立干净 checkout 上串行执行预算明确的单请求 32-token smoke，再决定是否扩展测量。
