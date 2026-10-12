# ds4 性能与效果：公开证据审计

研究日期：2026-10-13。阅读版本：[`antirez/ds4@fc80bd695da76ee14bcb86820bf900a5cdfda806`](https://github.com/antirez/ds4/tree/fc80bd695da76ee14bcb86820bf900a5cdfda806)（提交日期 2026-10-07）。
本轮只阅读源码、文档和上游保存的结果；**未构建、未下载模型、未运行推理，没有本项目 Mac 或 PGX GB10 实测数据**。
下文“作者报告”表示上游测量，“静态代码”表示读取实现，“待验证”表示研究假设。文档版本不等于测量版本。
“未载”表示引用段落未提供相应信息，不用当前 commit 或提交日期补作测量 provenance。

## 1. 单 GB10 的普通解码基线：有参照，缺少完整测量身份

**作者报告**：[单 Spark QA 表](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/QA_BEFORE_RELEASES.md#L1329-L1351)。
硬件为单 DGX Spark，模型为 Flash 0731 Q2、权重 resident；Promessi Sposi 输入，分配 context 9216，128 teacher-forced decode tokens，三次中位数。
温度不适用于 teacher-forced continuation；具体 GGUF checksum、测量 commit、日期、时钟和功耗配置未载。

| Context frontier | Prefill t/s | Decode t/s |
| --- | ---: | ---: |
| 2048 | 823.49 | 19.25 |
| 4096 | 899.89 | 16.22 |
| 8192 | 931.17 | 15.97 |

用途：估计第一轮实验预算和发现明显回退。不能写成“当前 fc80bd6 在本机达到这些速度”，也不能据此宣称领先其他引擎。

## 2. 上下文扫点：新增区间 prefill 与完整 prompt prefill 不同

**作者报告**：[历史 Flash Q2 表](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/PERFORMANCE.md#L35-L51)，原始 [GB10 CSV](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/speed-bench/gb10.csv) 与 [M5 Max CSV](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/speed-bench/m5_max.csv)。
两台机器均标 128 GB；模型只标 Flash Q2，具体 checkpoint/checksum 未载。Promessi Sposi、每步新增 2048 tokens、每个 frontier 生成 128 tokens。
该文档描述固定 greedy、排除 EOS 的 probe；分配容量、测量日期、运行 commit、重复次数未载，不能与上节的 teacher-forced 表混作同一次测量。

| 硬件 | Frontier | Prefill t/s | Generation t/s |
| --- | ---: | ---: | ---: |
| M5 Max | 2048 | 790.18 | 39.35 |
| M5 Max | 65536 | 398.50 | 27.64 |
| DGX Spark | 2048 | 825.76 | 18.05 |
| DGX Spark | 65536 | 822.98 | 13.84 |

后续行测的是 **continued prefill 的新增区间**；不能称“64K 完整 prompt prefill 822.98 t/s”。两种硬件还同时改变 backend，不能单独归因于 Metal/CUDA 算法。
文档明确这是 retained baseline，未逐次重测后来的 commit。[计时说明](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/PERFORMANCE.md#L12-L33)

## 3. DSpark：负载相关加速，默认非零温度会改变采样分布

**作者报告**：[single-Spark DSpark 对照](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/QA_BEFORE_RELEASES.md#L1338-L1351)。
单 Spark、匹配的 Flash 0731 Q2 target/drafter、分配 context 4096、prefill chunk 512、256 output tokens。
QA 段落未载日期与重复次数；配套 [DGX Spark 指南](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/DGX_SPARK.md#L151-L178) 补充为 2026-09-06、三次中位数。测量 commit 与具体权重 checksum 仍未载。
文档所指 sampling 为 nothink、top_p=.95、min_p=.05、seed=12345，提示词是完整 C hash-table 实现与不可预测散文。[设置与 prompts](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/QA_BEFORE_RELEASES.md#L1415-L1422)

| 工作负载 | 温度 | Ordinary t/s | DSpark t/s | 比值（由原表计算） |
| --- | ---: | ---: | ---: | ---: |
| C hash table | 0 | 19.72 | 31.41 | 1.59× |
| C hash table | 1 | 19.53 | 29.98 | 1.54× |
| 不可预测散文 | 1 | 19.53 | 18.81 | 0.96× |

**静态文档与源码**：T>0 默认 opportunistic。普通边界 token 按请求采样，匹配 target greedy 路径的 draft 直接接受，故比普通采样更确定。
`--mtp-exact-sampling` 的设计目标是保留 target sampling distribution：按 target 概率接受 proposal，拒绝时从剩余分布采样。
因此温度 1 的速度表不能称“同分布无损提速 54%”；低接受率散文还可能变慢。[sampling 语义](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/SPECULATIVE_DECODING.md#L86-L110)
[ds4.c 的两个分支](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L86648-L86700) 分别调用 speculative argmax 与 stochastic verifier，支持上述解释；没有因此完成 exact 模式的独立统计验证。
批量 verifier 的 floating-point reduction 顺序可能变化，长 greedy continuation 不保证 byte-identical。`--quality` / `--dspark-strict` 禁用 DeepSeek speculative acceptance，不能当作 exact sampling 同义词。
DSpark support 约增加 5.6 GiB weights 加运行状态、最多提议 5 future tokens，要求 target resident，不支持 external DSpark + SSD streaming，不加速 prefill。
单 Spark Q2 默认 confidence=.4，exact 默认 .8；模式对比还应记录阈值和调度。[支持范围与默认值](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/SPECULATIVE_DECODING.md#L5-L49)

## 4. “模型能力”“推理保真”“任务效果”不能混成一个分数

**静态文档**：`MODEL_CARD.md` 是官方 DeepSeek model card 摘录；其中 GPQA、LiveCodeBench、SWE 等数据属于模型与 reasoning mode，不是 DS4 引擎测评。
[文件定位](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/MODEL_CARD.md#L1-L7) 与 [模型分数表](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/MODEL_CARD.md#L101-L118)
该摘录给出官方 temperature=1/top_p=1，而 DS4 默认增加 min_p=.05；横比应显式对齐，不能各用默认值。[默认采样差异](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/MODEL_CARD.md#L215-L227)
效果评估应分别报告：模型 checkpoint/reasoning、NLL/logprob/数值状态保真、独立 coding/tool/retrieval 任务通过率。输出更像 API 不等于答题更正确。

## 5. 质量工具可复用，但小型回归集不是 leaderboard

**静态文档**：[quality-testing](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/README.md#L1-L50) 用固定官方 continuation 的 target-token NLL，避免只看一次 sampled answer。
公开 API 只提供有限 top-N alternatives；scorer 检查 token bytes 对齐，失败时跳过 API 概率比较，仍计文本 NLL。LCP 衡量精确前缀一致，早期一个 mismatch 会截断后续计分。
已有独立 `score_llama`，采用 **同 GGUF、同 manifest、token-identical DS4 renderer**，首轮横向保真审计可以复用。[控制 scorer](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/README.md#L270-L281)
上游说明旧 Flash API 已重定向到 V4.1；0731 固定 fixtures 必须保留，不能新调用 API 后冒充旧 checkpoint oracle。[checkpoint 警示](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/README.md#L52-L56)
`ds4-eval` 明确是 curated integration tests：hard 50 题，core 92 题；并非官方完整 benchmark distribution 或排行榜成绩。[EVAL_DATA](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/EVAL_DATA.md#L1-L17)、[core 构成](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/EVAL_DATA.md#L50-L54)
可以据此发现同配置回归，不能把测试通过率宣传为 DS4 对 vLLM/SGLang/llama.cpp 的总体效果排名。

## 6. 量化结果：局部有改善，不等于通用任务提升

**作者报告**：[Flash Q4 imatrix 比较](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/imatrix/README.md#L148-L175)。
模型为 Flash Q4，100 条官方 continuation；硬件、精确 checkpoint、测量日期/commit、温度、context、重复次数和置信区间未载。
old Q4 NLL .177357819 → imatrix .173895148（相对 -1.95%），case wins 54/46。**只能称参考续写 NLL 改善，不能称通用任务能力 +1.95%。**
[混合量化实验](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/mixed/README.md#L11-L21) 把 IQ2 base 的最后六层 37–42 routed experts 换为 Q4，约 91 GB，相比 full Q4 约 153 GB。
其硬件/日期/commit/temperature/context/重复次数未载；作者称 output agreement 更接近 Q4，且明确不是 benchmark score。适合形成消融假设，不能写“效果等同 Q4”。
较完整的另一模型方法实例：2026-09-15，DGX Spark GB10 CUDA sm_121，source `3cae02295f003defca8bdf9db8043dde8f846c57`，Qwen3.8-Flash-Next Q2/Q4，无 MTP/vision。
100 short cases，context4096；12 long cases，context32768。teacher-forced scoring 的温度不适用；fast/quality 各一组保存评分，并以配对 case bootstrap 估计差异。
short default NLL Q2/Q4=.352980/.290297，long=.154199/.124611。[结果说明](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/README.md#L375-L409)、[含模型/source hashes 的原始结果](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/data/qwen38-flash-alibaba-100/cuda-spark-20260915/results.json#L1-L43)
作者报告 fast-vs-quality 的 paired intervals 包含零；这不证明所有 context/任务等价。Qwen hosted endpoint 精度未公开，且不承诺与开放 Next 权重完全相同。[参考限制](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/README.md#L160-L164)

## 7. 指标、评分器和 tokenizer 本身也要审计

**作者报告**：[V4.1 router 比较](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/README.md#L59-L95)。
M5 Max，V4.1 IQ2_XXS/Q2_K imatrix，48 GiB expert cache，context34816，`aafc65b444f` 对 PR #1044 shader changes，112 prompts 单个完整 pair。
fixtures 于2026-09-19收集，T=1、nothink、max64、top_p默认1；实际本地测量日期未载，NLL为teacher-forced评分。
NLL .396840→.395353，paired Δ=-.001488，95% CI [-.003328,+.000199]；未见明确回退，也未预设 equivalence margin，**不能声明证明等价**。
该组有一题 tokenizer 将 continuation 编为65 tokens，而API/官方tokenizer为64；保留 reproducer 并排除API概率比较，仍计text NLL。[token 对齐问题](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/README.md#L42-L48)
**静态历史审计**：2026-10-06 的 [`d6b516a`](https://github.com/antirez/ds4/commit/d6b516abb4a286a423fdb4071a19f86a3d9ab9c0) 修正 `score_official.c::api_top1_update`。
此前将 alternatives 第一项当 top1；修正后按最大 logprob 选择，并处理并列与不可映射 token。历史 API top-token agreement 若未用新 grader 重算，不能作为最终核准基准。
这不表示所有旧 top1 数字都错，也不能据此否定历史 NLL。复测应统一 grader 版本，把“运行时变化”与“评分算法变化”分开。

## 8. 容量与 SSD 路径：要比较完整部署代价

**作者文档**：V4.1 calibrated Q2 main weights约151.8 GiB、Q4约294.15 GiB，另188.8 GiB Engram disk-only；单128GB Spark不能全驻留其Q2。[容量说明](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/QA_BEFORE_RELEASES.md#L1688-L1709)
它与上文可 resident 的 Flash0731 Q2 是不同 checkpoint/容量问题，不能混在“Q2”一栏当相同模型。
**作者报告**：[V4.1 CUDA SSD 工作负载](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/QA_BEFORE_RELEASES.md#L1815-L1864)。
单Spark，V4.1 Q2、disk-only Engram、无spec，64 GiB expert cache hint、context64K、Promessi Sposi、32 teacher-forced outputs；single-run，测量日期/commit未载。
fresh2K prefill84.90 t/s，fresh32K384.45 t/s，32K后追加3241 tokens为87.77 t/s。初始化大小、append区间与kernel分支变化明显，不能只挑最佳单点。
另auto-cache约80GiB，在32K后约3.2K `/read` prompt记录约100 t/s prefill、9.3 t/s decode（256 outputs）；该段重复次数未载。
cache hint可能被admission调小；测量需保留实际cache、冷/热状态、SSD吞吐、memory/swap。容量方案对比与纯runtime对比应分表。

## 9. 现有工具的计时口径与复用条件

**静态代码**：`ds4_bench.c` 的 `prefill_sec` 只围绕 `ds4_session_sync`，按 `frontier-previous` 计新增tokens；snapshot/replay在计时窗口外。
`gen_first_ms` 是 prefill 完成后的第一个 decode cycle，不是 HTTP TTFT。[循环与计时](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_bench.c#L817-L839)、[decode 输出字段](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_bench.c#L977-L1030)
Metal microbench已有warmup、ABBA/BAAB顺序、同engine/fresh sessions、完整vocabulary logits bit identity，是单优化消融可以复用的基础。[A/B harness](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/speed-bench/README.md#L30-L96)
`serve_concurrency_bench.py` 把queueing/render/prefix reuse/streaming算入HTTP测量，TPOT使用usage tokens；其ITL实际计 **SSE text chunk间隔**，有buffering/speculation时不能称逐token ITL。
[HTTP指标实现](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/speed-bench/serve_concurrency_bench.py#L68-L133)
同脚本按随机nonce与Python `hash(nonce)`选择语料offset；跨引擎实验要先冻结相同request清单，并另外保留输出文本，不能各自随机跑一次当成相同负载。
[输入构造](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/speed-bench/serve_concurrency_bench.py#L160-L173)
上游QA建议warmup后3次中位数、同GGUF/prompt/settings、保留acceptance和生成文本；重复循环可能虚增投机吞吐，需teacher-forced对照分离内容变化与kernel成本。
[性能QA规范](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/QA_BEFORE_RELEASES.md#L1293-L1311)

## 10. 对调研计划的直接建议（待验证）

1. 先做模型/架构/量化支持、GPU构建来源、tokenizer/template可比性门槛，再填跨引擎速度表；失败和不支持也是结果。
2. 第一轮以单GB10、同一可容纳GGUF、普通decode对比ds4与llama.cpp；分别测fresh/append、cold/warm与engine/HTTP。
3. DSpark拆ordinary、opportunistic、exact三组，代码与非模板化散文都测，保留acceptance、重复、代码测试结果和time-to-valid-answer。
4. default-vs-quality独立评估，固定新的grader与匹配checkpoint fixtures；量化Q2/mixed/Q4另设实验轴，避免把数值模式、量化、sampling混合。
5. 长期会话复现追加prompt、工具结果、KV恢复、prefix cache，先并发1/2，再决定是否需要4/8；单请求速度不能替代多会话aggregate throughput。

已理解：ds4保留了可复用的性能与保真工具；速度收益强烈依赖负载、权重驻留和采样语义，现有材料不足以给跨框架总排名。
未解决：本机GB10可用模型/空间与实际backend、其他引擎对相同checkpoint/GGUF的支持、exact sampling统计行为、旧结果是否以修正grader重算。
下一步：按调研计划先完成可比性与来源核验，冻结输入/权重/构建配置，再运行预算受限的普通decode基线；本记录中的历史表保持为作者报告。
