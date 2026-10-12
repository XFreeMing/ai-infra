# ds4 横向对照：先确定可比性，再比较性能与效果

研究日期：2026-10-13。状态：已完成固定版本源码阅读，尚未执行推理或 GB10 实验。

本文回答“哪些对象可以比较、比较什么、有哪些门槛”。ds4 的具体实现机制另见同目录的主体研究记录。

## 1. 版本与证据范围

| 项目 | 本次阅读的完整 commit |
| --- | --- |
| ds4 | `fc80bd695da76ee14bcb86820bf900a5cdfda806` |
| llama.cpp | `972b5fcc19670e597f8de9bf5cacfb85752a5fff` |
| vLLM | `2534a96fcf165202f5358335f8b0705175159fd0` |
| SGLang | `f006fb49ecd93322802ddebff80821f8e6296893` |

- **代码事实**：可以定位到上述版本的文件与符号；只证明实现或限制存在。
- **上游主张**：来自上游说明、注释或配方中的验证记录；不是本项目实测结果。
- **研究判断**：据当前证据安排比较优先级；不代表已经证明谁更快或质量更高。
- **待验证**：本机加载、GB10 kernel、内存占用、输出与速度都需要执行实验。

本次没有安装框架、下载模型或运行 GPU 测试。后续实验必须另外记录实际导入路径和构建来源。

### 面向本项目的优势与代价

下表是根据后文源码与支持条件作出的研究判断；“优势”指值得验证的工程能力，不是已测出的速度排名。

| 项目 | 值得学习和验证的优势 | 对当前研究的代价/限制 | 最合适的角色 |
| --- | --- | --- | --- |
| ds4 | 已知模型的专用执行、GB10/Metal 路径、低 bit 专家、SSD 与会话恢复可以沿一条完整路径研究 | 模型布局与功能组合受限；量化、采样策略和复杂状态会影响比较；当前源码体量不小 | 主研究对象，逐项验证机制 |
| llama.cpp | GGUF 转换/执行、CUDA/Metal、V4 模型与服务缓存都可直接对照 | 同文件能否加载与正确执行仍需检查；通用功能丰富不保证此模型此后端更快 | 首个同权重基线和双机学习参照 |
| vLLM | token budget 调度、分块 prefill、prefix cache 和原生 V4 路径适合研究服务吞吐与排队 | 单 GB10 的容量/kernel 条件需核实；GGUF 依赖额外实验性插件，比较维护成本更高 | 调度学习与通过准入后的服务对照 |
| SGLang | RadixCache、overlap 调度、V4 状态管理与 SSD expert-pack 提供不同实现切口 | 原生 Spark 配方为双机；expert-pack 的量化、并行与 CUDA graph 限制不能忽略 | 缓存/调度学习及后续 SSD 专题 |

## 2. 第一层：同模型、同权重的引擎对比

研究判断：优先比较 **ds4 与 llama.cpp**。两者使用相同模型文件的可能性最值得先验证。
同一个 ds4 GGUF 能否在本次固定的 llama.cpp 中完整加载并正确生成，**尚未验证**。

| 对象 | 已确认的代码事实 | 对齐条件 | 本阶段安排 |
| --- | --- | --- | --- |
| ds4 | 本研究的起点；实现范围由主体研究记录说明 | 固定 GGUF、配置、模板及编译版本 | 建立单请求基线 |
| llama.cpp | `DeepseekV4Model` 注册 V4 GGUF；`llama_model_deepseek4` 实现模型及 MTP graph | 必须实际加载同一 GGUF；核对 tensor 类型、模型变体、tokenizer、模板 | 首个直接对手，先做正确性 smoke |
| vLLM | 注册 V4、MTP、DSpark；原生量化配置区分 MXFP4/FP8 experts | 外置 GGUF 插件尚未纳入本项目版本锁；未证明可运行相同文件 | 暂不进入同权重速度排名 |
| SGLang | 有 V4 模型和专用 GGUF expert-pack 路径 | DeepSeek expert-pack 的 experts 只接受 MXFP4；非专家类型也受限 | 满足格式与运行门槛后再考虑 |

llama.cpp 的 converter 使用 `MODEL_ARCH.DEEPSEEK4`，并声明 `supports_mtp_export = True`。[证据 2]
运行端包含 `build_hc_sinkhorn`（L316）、`build_hca_compressed_kv_from_state`（L470）、`build_csa_lid_attention`（L707）等符号。[证据 3]
这些事实说明模型机制已经实现，不说明任何新旧 GGUF、任意量化类型和所有后端组合都能工作。

同权重比较先固定以下条件：

1. 模型必须是同一变体与 revision；不混用 Flash、Flash-Base、Flash-0731、V4.1。
2. 保存 GGUF 的 SHA256、分片列表与 tensor dtype 清单；文件名中的“4bit”不足以证明权重相同。
3. 对齐 tokenizer、chat template、BOS/EOS、thinking/reasoning 模式及 stop 条件。
4. 对齐输入 token 序列、输出预算、采样参数、context/rope 参数。
5. 先关闭 speculative 和跨请求 prefix cache；成功后再分别开启一个优化。
6. 若引擎必须改变权重、KV 精度或 attention 设置，将其作为单独配置记录。

## 3. 第二层：同一台 GB10 上可部署的完整方案

这一层回答“在我的设备上，哪个配置更有用”。权重或存储方式不同，需要同时报告速度、质量和资源。
它不能直接回答“哪个引擎的实现更快”，因为量化、offload 和 checkpoint 差异可能主导结果。

| 方案 | 潜在研究价值 | 明确门槛或限制 | 当前结论 |
| --- | --- | --- | --- |
| ds4 + 已选 GGUF | 建立单用户学习、问答与代码研究的基准 | 实际模型体积、工作内存和当前 backend | 待加载及 GB10 smoke |
| llama.cpp + 同一 GGUF | 尽量保持权重一致，比较内核、缓存和 offload | 目标 GGUF/量化的完整加载；GB10 构建；模型与 KV 的总占用 | 优先验证 |
| vLLM + 原生 V4 checkpoint | 学习服务调度与原生量化路径 | checkpoint 容量、SM12x attention/MoE kernels、依赖版本 | 先做可部署性清单 |
| vLLM + GGUF 插件 | 如支持相同 GGUF，可扩大同权重对照范围 | 插件 commit、依赖、模型映射、量化算子及功能组合都未锁定/验证 | 独立候选，不预设可用 |
| SGLang + 原生 V4 checkpoint | 学习缓存、调度和 speculative 组合 | 本版本官方 Spark 配方采用两台机器；不能直接照搬到单机 | 当前不作为单 GB10 基线 |
| SGLang + SSD expert-pack | 研究专家流式读取、缓存命中、I/O 与内存的取舍 | MXFP4 experts、受限非专家 dtype、SSD 空间/带宽、CUDA kernels | 第二阶段专题候选 |

### SGLang expert-pack 的具体边界

代码事实：`handle_expert_pack` 接受 raw GGUF 文件，准备配置及 pack；要求 TP、DP、attention-DP、EP 都为 1。[证据 9]
该路径关闭 CUDA graph 和 shared-experts fusion；不能沿用原生服务路径的优化配置来描述它。
`ExpertPackStore` 拒绝非 MXFP4 expert dtype/quant scheme，并检查 identity transform 与 block size。[证据 8]
`deepseek4_nonexpert_weights_iterator` 接受 Q8_0、BF16、F32、I32；其他非专家类型会报错。[证据 7]
因此，“SGLang 支持 GGUF”不等于“可直接加载 ds4 使用的任意 Q2_K/IQ2 混合 GGUF”。
该路径目前要求 CUDA；单卡限制也不等于已经在本项目的 GB10 上验证。

上游主张：官方 V4 Spark 配方使用 `2×GB10`、`TP=2`、专用 preview image，并说明所列官方 checkpoint 无法放入单台 128GB 机器。[证据 10]
这个限制只针对配方中的 checkpoint 与部署形态，不能外推为“SGLang 的所有 V4 配置都不能单机”。
不要把上游双机吞吐、DSpark 接受率或质量数字当成用户单机的预期结果。

### vLLM 的具体边界

代码事实：`DeepseekV4FP8Config` 区分 Flash 的 MXFP4 experts 与 Flash-Base 的 FP8 experts。[证据 4]
官方说明将 GGUF 功能移至 `vllm-gguf-plugin`，并标为实验性、尚未优化。[证据 5]
本次只锁定 vLLM 主仓库，未锁定该插件；不能据模型注册推断同一 GGUF 可运行。
SM12x sparse attention 要求相应 FlashInfer sparse MLA API，并限制 KV cache dtype。[证据 6]
这些检查只是部分兼容性门槛；还要验证 ARM64 依赖、MoE kernel、内存、启动及正确生成。

## 4. 第三层：架构学习，不做跨硬件速度排名

| 学习问题 | llama.cpp 阅读入口 | vLLM 阅读入口 | SGLang 阅读入口 |
| --- | --- | --- | --- |
| 单请求与批处理如何协调 | `tools/server/server-context.cpp`：slot 初始化 L1391、连续批处理 L3478 | `vllm/v1/core/sched/scheduler.py`：`Scheduler` L84、token budget L611 | `python/sglang/srt/managers/scheduler.py`：`Scheduler` L442、overlap L495 |
| prefill 如何分块 | server batch 与 ubatch 配置/执行路径 | 同 scheduler 的 chunked-prefill 分支 L1161–1169 | `init_chunked_prefill` L1313 |
| prefix cache 如何复用 | `server-context.cpp`：prompt cache L1496、命中统计 L3805 | scheduler 的 `enable_prefix_caching` L332 | `mem_cache/radix_cache.py`：`RadixCache` L281、`match_prefix` L364 |
| V4 状态如何保存 | `src/llama-model.cpp` L2994–3030：V4 专用 cache 与 MTP cache | `vllm/models/deepseek_v4/` 的 compressor、sparse MLA | `mem_cache/deepseek_v4_memory_pool.py`、`deepseek_v4_compress_state.py` |
| 投机解码有哪些状态与代价 | `common/speculative.cpp`：MTP L1390；server 支持多种 speculative 类型 | `model_executor/models/registry.py`：DSpark L643、MTP L676 | `models/deepseek_v4_dspark.py`：draft 模型 L731 |
| 跨设备如何组织 | `ggml/src/CMakeLists.txt` L601/L604：CUDA/Metal | V4 的 NVIDIA/AMD/CPU/XPU 实现分支 | 原生模型、SM120 attention 与 expert-pack CUDA 路径 |

表中行号均对应第一节固定 commit；这些入口用来追踪实现，不能视为全部功能组合已经验证。
llama.cpp 是 Mac 与 PGX 共用研究对象的自然起点，但 Metal 后端存在不保证用户 Mac 有足够内存运行目标模型。
不把 SGLang 的其他 MLX 路径或 vLLM 的其他平台扩展，当成本研究 V4 GGUF 已支持 Metal 的证据。
术语需要分清：**DSpark 是 speculative draft 方法；DGX Spark 是 GB10 设备。**

## 5. 首批实验的准入与指标

先完成一次 32-token smoke：检查加载日志、tensor/权重来源、实际 backend、输出、结束原因、峰值内存。
有条件时检查 logits 或 greedy token 差异；浮点实现差异可能影响 token，不预设必须逐 token 完全相同。
只有能够解释正确性差异的配置，才进入性能测量。

候选矩阵：ds4/llama.cpp × 输入 1K/8K/32K × 输出 256 × 并发 1；每格重复三次。
这是计划，不是已确认可运行的参数；实际支持、运行预算或内存不足时先缩小矩阵。
分别报告冷启动、无历史缓存的 prompt、热 prefix、多轮追加上下文与稳态 decode，避免混在一起。
性能指标：启动时间、TTFT、prefill tok/s、decode tok/s、TPOT 分布、总请求时间、峰值统一内存。
三次重复用于初筛变动；少量请求的 p95 不具有稳健的尾延迟代表性，保留原始时间序列。
涉及 SSD 时额外记录读取量、I/O 等待、expert cache 命中率和缓存大小；涉及 speculative 时记录接受率与验证开销。
效果指标：短文本一致性、可验证问答/代码题、工具调用格式、长上下文检索、截断与错误率。
小样本效果测试用于发现明显回退，不能据此声称模型综合能力提升。
改变权重量化、KV 精度或 speculative 策略时，必须单列质量与资源变化，不能只给 tok/s。

## 6. 关键永久链接

1. [ds4 固定研究版本](https://github.com/antirez/ds4/tree/fc80bd695da76ee14bcb86820bf900a5cdfda806)
2. [llama.cpp：V4 GGUF converter、MTP 与模板](https://github.com/ggml-org/llama.cpp/blob/972b5fcc19670e597f8de9bf5cacfb85752a5fff/conversion/deepseek.py#L525-L571)
3. [llama.cpp：V4 模型、MTP graph 与 attention 构建](https://github.com/ggml-org/llama.cpp/blob/972b5fcc19670e597f8de9bf5cacfb85752a5fff/src/models/deepseek4.cpp#L19-L824)
4. [vLLM：V4 原生 FP8/MXFP4 配置](https://github.com/vllm-project/vllm/blob/2534a96fcf165202f5358335f8b0705175159fd0/vllm/models/deepseek_v4/quant_config.py#L29-L46)
5. [vLLM：GGUF 外置插件说明](https://github.com/vllm-project/vllm/blob/2534a96fcf165202f5358335f8b0705175159fd0/docs/features/quantization/gguf.md#L1-L12)
6. [vLLM：SM12 attention 兼容性门槛](https://github.com/vllm-project/vllm/blob/2534a96fcf165202f5358335f8b0705175159fd0/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py#L140-L176)
7. [SGLang：V4 非专家 GGUF 类型限制](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/python/sglang/srt/model_loader/expert_pack_loader.py#L73-L138)
8. [SGLang：expert-pack 的 MXFP4 限制](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/python/sglang/srt/layers/moe/expert_pack.py#L294-L306)
9. [SGLang：expert-pack 参数与 CUDA graph 限制](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/python/sglang/srt/arg_groups/expert_pack_hook.py#L32-L204)
10. [SGLang：V4 双 GB10 配方及容量说明](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx#L3655-L3685)
11. [llama.cpp：server 连续批处理入口](https://github.com/ggml-org/llama.cpp/blob/972b5fcc19670e597f8de9bf5cacfb85752a5fff/tools/server/server-context.cpp#L3478)

## 7. 本次理解与下一步

已理解：四个项目应按同权重引擎、同硬件方案、架构学习分层；当前最合适的首个直接对手是 llama.cpp。
未解决：目标 GGUF 的确切来源与 tensor 类型、llama.cpp 实际加载、单 GB10 内存余量、插件及 kernel 版本、正确性差异。
下一步：完成目标模型文件清单与 GB10 环境盘点，先让 ds4 和 llama.cpp 各完成同输入的 32-token smoke，再冻结首批性能矩阵。
