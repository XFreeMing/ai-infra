# ds4 / DwarfStar：源码证据地图

阅读日期：2026-10-13。固定版本：`fc80bd695da76ee14bcb86820bf900a5cdfda806`。
上游：[antirez/ds4 固定版本](https://github.com/antirez/ds4/tree/fc80bd695da76ee14bcb86820bf900a5cdfda806)。
状态：以下为**静态已读代码事实**与**待测推断**；本轮未构建、未下载权重、未执行推理，未产生 Mac 或 GB10 性能结果。

## 阅读边界与方法

先读了上游 [AGENT.md](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/AGENT.md)，保持上游只读。
其中“V4 Flash specific”及 `ds4.c` 文件头描述比当前实现窄；当前代码已有 DeepSeek V4/V4.1、GLM DSA、Qwen4 EXP 等 model family。
因此不能把“只支持 V4 Flash”当成当前事实，也不能因为支持几个 family 就称为通用 GGUF 执行器。
当前 `ds4.c` 约 8.7 万行、CUDA 约 3.6 万行、Metal 约 5.1 万行；“small/readable”是作者目标，阅读宜沿具体问题跳转。
下文说“专用”表示固定模型布局与硬件路径的工程取舍；没有据此宣称某项技术由 ds4 首创。

## 先分清：模型机制与引擎优化

| 层次 | 本版本看到的内容 | 公平比较时要固定什么 |
| --- | --- | --- |
| 模型架构 | V4 Flash 的 SWA、ratio-4 / ratio-128 压缩注意力、indexer、MoE、HC | 同一模型 revision；不能拿这些直接证明 ds4 胜过其他引擎 |
| 权重表示 | 各 tensor 可采用不同量化类型；低 bit routed experts、imatrix | 相同权重文件/校验值或明确区分量化配方 |
| 执行实现 | 专用 graph driver、融合 kernel、aligned repack、不同 batch dispatch | 相同模型与配置下做 engine 或路径消融 |
| 内存/I/O | mmap、GPU residency、SSD expert cache、异步预取 | 可用 RAM、cache 容量、冷热状态、SSD 与系统负载 |
| 请求与状态 | live KV、磁盘 checkpoint、slot routing、推测解码 | 同一 prompt、token 历史、采样、并发与缓存条件 |

`SWA` 是 sliding-window attention；不要把 `HC`（Hyper-Connection）与 `HCA`（压缩注意力命名）混为一谈。
代码主要用 compression ratio 表达 V4 压缩路径，并不靠 `CSA/HCA` 字符串命名。
若要把 ratio-4/128 正式映射为论文 CSA/HCA 术语，还需对照相同模型版本的一手模型说明。
`MLA` 符号在这里也用于 GLM 路径；不能仅凭同一文件有 MLA 字样就把全部模型统称为同一种注意力实现。

## 十条可定位的源码事实

### 1. 支持范围扩大，但仍按已知模型布局精确校验

**事实**：`ds4_model_family` / `ds4_variant` 枚举已有多个家族；`DS4_SHAPE_FLASH` 固定 V4 Flash 为 43 层、4096 embedding、256 experts 中使用 6 个、1 个 shared expert、128 SWA、4 HC streams。
`ds4_select_shape_from_metadata()` 对 DeepSeek4 的 Flash/Pro 布局逐项匹配；`validate_compress_ratio_metadata()` 要求层压缩配置与预期一致，否则退出。
这是缩小语义范围、尽早拒绝错误输入的实现方式；代价是模型扩展需要专门适配，不能任意换 GGUF。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`ds4.c:525–638` [family、variant、Flash shape](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L525-L638)；`ds4.c:6377–6475` [shape 与压缩元数据验证](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L6377-L6475)。

### 2. 原生压缩注意力与 HC 是模型语义，不能归功于执行引擎

**事实**：`ds4_expected_layer_compress_ratio()` 对 V4 Flash 前两层返回 0，此后偶数层 4、奇数层 128；V4.1 是另一套规则。
`layer_attention_raw_swa_one()` 维护 raw KV，非零 ratio 生成 compressed KV，ratio-4 还维护 indexer compressed KV 并选择可见行。
`layer_attention_mixed_one()` 对 raw + compressed rows 计算 attention；未选中的 compressed rows 被 mask。
`hc_split_sinkhorn_one()` 则计算 HC pre/post gates 与经多轮行列归一化的 combine matrix。
这里的 CPU 实现有助于读懂语义，只读它并不构成 GPU 正确性验证。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`ds4.c:1366–1382` [压缩规则](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L1366-L1382)；`ds4.c:15206–15315` [单 token 注意力](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L15206-L15315)；`ds4.c:14835–14896` [mixed attention](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L14835-L14896)；`ds4.c:11862–11940` [HC Sinkhorn](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L11862-L11940)。

### 3. 专用 graph driver 与 Metal 无复制权重映射减少通用层开销

**事实**：`metal_graph_eval_token_raw_swa()` 在 begin/end commands 中调用 `metal_graph_encode_token_raw_swa()`，然后读取 logits；driver 明确管理模型步骤和状态。
`ds4_gpu_add_model_view_range()` 用 `newBufferWithBytesNoCopy` 包装 mmap 页，并处理 GPU buffer 范围/大小限制。
`metal_graph_*` 名称沿袭 Metal，但当前也被 CUDA 共用；看到函数名不能断定执行硬件。
“whole-model graph”也不能直接翻译成“全模型单 kernel”或“所有模式都只提交一次”；streaming、同步和后端会改变命令边界。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`ds4.c:33986–34038` [graph token 执行](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L33986-L34038)；`ds4_metal.m:2172–2303` [mmap no-copy views](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_metal.m#L2172-L2303)。

### 4. 低 bit 不是一个统一开关；tensor 配方、校准和 kernel 都有区别

**事实**：`policy_type()` 可分别指定 routed gate/down/up、embedding、output、shared expert、attention 的类型。
`f32_to_type()` 使用给定 imatrix；若目标格式要求 imatrix 而未提供，就用每列权重平方和构造 synthetic importance。
CUDA `routed_moe_launch()` 对 IQ2_XXS gate/up + Q2_K down 的 aligned artifacts 分开选择短 row vector 路径和批量 fused 路径。
因此“Q2”标签不足以证明两个文件等价；synthetic importance 也不能写成执行了 activation calibration。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`gguf-tools/deepseek4-quantize.c:1291–1316` [量化策略](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/deepseek4-quantize.c#L1291-L1316)；`:1355–1397` [importance fallback](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/deepseek4-quantize.c#L1355-L1397)；`ds4_cuda.cu:26117–26278` [aligned MoE dispatch](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu#L26117-L26278)。

### 5. GB10 有实际专用判断；repack 与权重驻留是主要研究切口

**事实**：`ds4_gpu_init_multi()` 将 integrated 且 compute capability 12.1 的首个设备标为 Spark，`ds4_gpu_device_is_spark()` 还要求仅一个 GPU。
`cuda_build_f16_pair_chunks()` 只在 Spark、非 SSD、单 GPU 情况为满足具体尺寸的 compressor F16 矩阵构建分块副本。
`cuda_build_derived_artifacts()` 在 integrated CUDA 上构建 IQ2/Q2K/Q8 aligned artifacts；`ds4_gpu_set_model_map()` 在 artifacts 能替代 expert residency 时保留原 mmap 为 unpinned。
推断：这可能减少 decode 开销或重复驻留，但也带来启动构建时间与额外布局存储；收益需连同启动成本测量。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`ds4_cuda.cu:316–318` [单 GPU 判断](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu#L316-L318)；`:2768–2774` [Spark 识别](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu#L2768-L2774)；`:4766–4794` [F16 pair 限制](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu#L4766-L4794)；`:4848–4945` [derived artifacts](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu#L4848-L4945)；`:4007–4043` [避免完整模型复制/驻留](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu#L4007-L4043)。

### 6. SSD decode 用有限 expert cache，命中保护与替换是显式的

**事实**：`cuda_stream_selected_cache_begin_load()` 将本批选中 expert 去重、映射到 cache slots，先保护所有命中，再淘汰未受保护且 `used` 最旧的 slot。
容量依据预算与可用内存调整，预留 8 GiB headroom；分配失败会缩小容量重试；该 CUDA SSD 路径要求单 GPU。
`ds4_ssd_auto_cache_plan()` 由目标内存、non-routed bytes、每个 expert bytes 和最大数量计算 cache 容量。
`ds4_ssd.c` 主要是参数/预算/内存锁辅助；真实读盘、搬运和替换在 backend 与 graph driver。
推断：同样 RAM 下，实际 expert locality 与缓存容量可能比标称 SSD 顺序带宽更能解释 decode 波动。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`ds4_cuda.cu:28979–29146` [expert cache](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu#L28979-L29146)；`ds4_ssd.c:108–140` [auto cache plan](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_ssd.c#L108-L140)。

### 7. 异步预取有路径和容量条件，不能笼统称为预测下一个 expert

**事实**：`ds4_gpu_stream_expert_cache_prefetch()` 接收 current/next layer table，要求 SSD、单 GPU、有 cache budget、对应布局兼容，并要求 slot 容量至少容纳 current + next 的全部 experts。
生产调用点在 V4.1 的 `ds41_graph_prefill_sweep()`，且 `total_count >= 2048` 才尝试；不能把它当作所有模型所有 decode 的默认行为。
它为下一层未命中的 experts 保留 slots，按文件顺序整理读取；`cuda_stream_prefetch_read()` 用后台线程、两个 8 MiB staging buffers、异步 H2D copy 重叠 I/O。
`ds4_gpu_stream_expert_cache_prefetch_finish()` 只在真实 router 命中后提升预取条目的热度，避免仅仅被预取就挤掉常用项。
另有 decode selected-load/shared-expert overlap 分支，受 quant、TP、profile、router 等条件控制；这与整层 look-ahead 是不同机制。
须测量命中率、读入但未使用字节、等待时间、prefill/decode 分项和输出漂移；不要只记录“预取已开启”。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`ds4_cuda.cu:29199–29392` [预取线程/发布/条件](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu#L29199-L29392)；`ds4.c:42468–42484` [V4.1 prefill 调用门槛](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L42468-L42484)；`ds4.c:26672–26740` [selected/shared overlap 条件](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L26672-L26740)。

### 8. 长会话优化覆盖 live slot、磁盘 payload 与 agent 恢复

**事实**：server `job_slot_score()` 按真正可复用的 token 长度选择 slot；没有可复用内容时优先空 slot，再按闲置时间或重建代价选择淘汰对象。
`ds4_kvstore_try_load_text()` 按文本前缀寻找磁盘记录，检查 header、模型、文本 hash/字节前缀，载入 payload；随后保留 payload 原始 token 历史，只 tokenize 新后缀。
agent `agent_kv_load_path()` / `agent_kv_save_path()` 另含量化与 transcript 检查、payload 存取、临时文件后 rename；部分 TP 恢复必须从文本重建两端状态。
这强调用户实际连续使用时的 TTFT；它与 paged KV allocator、跨请求 radix tree 不是可直接互换的名词。
实验仍需固定模型文件 hash、模板和 sampling，并验证恢复后 logits/输出；本地图缓存 header 不是实验 provenance 的替代品。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`ds4_server.c:14819–14850` [slot routing](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_server.c#L14819-L14850)；`ds4_kvstore.c:1196–1322` [磁盘前缀恢复](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_kvstore.c#L1196-L1322)；`ds4_agent.c:4896–4986` [agent 恢复](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_agent.c#L4896-L4986)；`:5001–5108` [agent 保存](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_agent.c#L5001-L5108)。

### 9. 服务端可接多请求，不代表所有组合都有原生 row batch

**事实**：`ds41_cuda_row_batch_supported()` 和 `ds41_sessions_batch_supported()` 的联合条件比平台文档的一句话精确；`DS4_TP_BATCH_MAX_ROWS` 为 8。
下面是此版本 V4.1 CUDA native row batch 的静态条件，不是测速承诺。

| 组合 | 此路径的条件/结论 |
| --- | --- |
| 单机 CUDA + SSD V4.1 | 2–8 rows；所有层 routed gate/up 为 IQ2_XXS、down 为 Q2_K |
| CUDA network TP=2 + resident V4.1 | 5–8 rows；同样要求上述 Q2 配方 |
| 单机 CUDA + resident V4.1 | 不满足该 native row batch placement 条件 |
| CUDA network TP=2 + SSD V4.1 | 不满足该 native row batch placement 条件 |
| quality / imatrix / image / support model / invalid checkpoint | 不进入该支持分支；对应配置是否整体可用仍须看其他验证 |
| 关闭 `DS4_CUDA_SESSION_BATCH_MOE` 或设置 `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6` | 不进入该支持分支 |

还要求共享 engine、graph 与 checkpoint position 对齐、workspace 的 prefill capacity 足够；Metal 的 V4.1 TP 门槛是 3 rows，且此支持分支拒绝 streaming。
CUDA V4 的 `ds4_sessions_eval_batch_cuda()` 可顺序 encode 多 session 后同步，多 GPU placement 可走 pipeline interleave；这与 V4.1 row batch 不同。
因此需要按模型、后端、resident/SSD、slots、quant、speculation 分列，避免“CUDA 全部 ordered”或“全部 continuous batching”的绝对说法。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`ds4.c:42733–42761` [CUDA row placement/quant](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L42733-L42761)；`:79550–79573` [sessions 条件与 TP 门槛](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L79550-L79573)；`:85496–85582` [CUDA batch/fallback](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L85496-L85582)；`ds4_tp.h:34` [最大 rows](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_tp.h#L34)。

### 10. DSpark 推测解码与 DGX Spark 硬件必须分开记录

**事实**：`ds4_session_eval_speculative()` 的接口契约说明：positive-temperature DSpark 默认提交经 greedy 验证的 draft tokens；`dspark_exact_sampling` 才选择精确 stochastic p/q acceptance。
所以不能假定“开启推测解码，所有采样分布天然不变”。具体速度/质量比较应独立列 no-spec、greedy verification、exact sampling。
DGX Spark 是硬件选择；DSpark 是可选推测执行方案。是否可以组合还需检查模型与 backend 支持。

证据：`fc80bd695da76ee14bcb86820bf900a5cdfda806`，`ds4.h:571–581`，`ds4_session_eval_speculative` [采样契约](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.h#L571-L581)。

## 从源码提出的对照问题

以下均为**尚未执行的假设检验**，需要先由 [调研计划](PLAN.md) 做容量、模型与预算筛选。

| 研究问题 | 最小对照 | 性能证据 | 正确性/限制证据 |
| --- | --- | --- | --- |
| 专用执行是否省下固定开销？ | 相同兼容 GGUF、单请求、同 prompt 的 ds4 / llama.cpp | 冷启动、TTFT、prefill、decode、内存 | tokenizer/template/token 数；先确认两边真实支持该布局 |
| aligned repack 是否值得？ | GB10 resident 固定权重，默认 / 禁用 derived weights | 构建时间、首轮与稳态 tok/s、峰值内存 | logits、top-k、固定题集；调试开关可改变多条路径需记录 |
| SSD cache 多大才有效？ | 固定模型与负载，两个可容纳预算；冷/暖分开 | miss/read bytes、I/O wait、p50/p95 token latency | 输出一致性、内存余量、热缓存造成的虚假增益 |
| 预取是否隐藏 I/O？ | 满足条件后，默认 / `DS4_CUDA_DISABLE_SSD_PREFETCH` | prefill、decode、预取 wait/bytes、SSD 占用 | 保持相同 cache 容量；未进入分支的运行不算对照 |
| KV 复用是否缩短真实多轮等待？ | 同会话冷重建 / live reuse / 进程重启后 disk restore | 各轮 TTFT、reused/prefilled tokens、save/load 时间 | tool message 与模板边界、恢复后输出；保存也有成本 |
| native batch 对单机 V4.1 SSD 有何收益？ | 满足 Q2 条件后 slots=1/2/4/8 | aggregate 与 per-user tok/s、TTFT、尾延迟 | 每个 session 独立 token/KV 正确性；不同用户公平性 |
| 低 bit/推测解码影响什么？ | 同题集、同基础模型，分开改变 quant 或采样方案 | 时间、内存、accepted drafts | reference logprob/KL、正确率、tool-call 成功率；避免同时改两个因素 |

暂不据此推断 ds4 普遍快于 vLLM/SGLang：后两者的优势可能在大 batch、多用户调度、不同量化或硬件，需用可运行且公平的配置验证。
Mac 不必持有完整权重也可完成源码与实验设计；GB10 先做环境检查和有限 smoke，再决定长跑。

## 本次理解与交接

**已理解**：ds4 的研究价值在少数模型的端到端适配、低 bit MoE 执行、内存/I/O 管理、会话状态复用之间的组合；各模块的主入口已经定位。
**未解决**：目标 GGUF 在其他引擎的实际兼容性、Mac/GB10 可用容量、各分支实际触发情况、长期多轮质量与性能退化；都不能靠本次静态阅读裁定。
**下次具体步骤**：先核查 GB10 的模型/内存/磁盘条件；然后沿 `routed_moe_launch()` 或 `ds4_session_sync()` 选择一个问题，记录真实分支日志，准备只有一项变化的对照。
如以当前 HEAD 的 DSpark response-boundary 修复为切口，应继续比较父提交并设计多轮/tool-call 用例；该变更分析见本研究其他记录。

English recap: DwarfStar combines model-specific execution, low-bit expert kernels, explicit memory management, and reusable session state. These are testable engineering choices, not measured advantages in this study.
