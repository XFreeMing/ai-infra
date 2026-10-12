# ds4 / DwarfStar：从真实源码到可验证的横向对照

研究日期：2026-10-13（Asia/Shanghai）

状态：首轮源码与一手资料调研完成；模型加载、GPU 性能和任务效果均未实测。

## 要回答的问题

在一台 PGX GB10 上，ds4 为少数模型做了哪些专用设计？哪些速度或容量收益来自引擎，哪些来自量化、模型结构、采样策略或工作负载变化？与 llama.cpp、vLLM、SGLang 比较时，哪些条件真的可比？

Mac 负责阅读、实验设计和结果分析；PGX 的长实验另行安排。本轮没有下载模型、安装推理依赖、修改上游源码或运行推理。文中的数字均注明为上游报告，不能当作本机成绩。

## 阅读顺序与产物

| 文档 | 解决什么问题 |
| --- | --- |
| 本页 | 先理解项目定位、关键差异及一个具体变更 |
| [SOURCE_MAP.md](SOURCE_MAP.md) | 机制落在哪些函数，模型结构与工程实现如何区分 |
| [EVIDENCE.md](EVIDENCE.md) | 速度与效果数字来自什么条件，能支持什么结论 |
| [COMPARISON.md](COMPARISON.md) | 四个框架的能力、代价及实际可比较性 |
| [PLAN.md](PLAN.md) | 下一步读什么、对比什么、如何验收与控制资源 |
| [antirez 人物记录](../../people/antirez.md) | 从作者的实际提交理解他在解决的问题 |

## 版本基线

| 项目 | 本次固定提交 |
| --- | --- |
| ds4 | `fc80bd695da76ee14bcb86820bf900a5cdfda806` |
| llama.cpp | `972b5fcc19670e597f8de9bf5cacfb85752a5fff` |
| vLLM | `2534a96fcf165202f5358335f8b0705175159fd0` |
| SGLang | `f006fb49ecd93322802ddebff80821f8e6296893` |

ds4 HEAD 的上游提交日期为 2026-10-07；读取日期为 2026-10-13。资料中的 9 月、10 月初性能记录是历史测量，不是本次固定提交的重测结果。其他框架维持项目已有版本，没有为本轮调研追随上游更新。

## 首轮结论

### 1. ds4 是专用推理系统，效果首先取决于所运行的模型与推理条件

当前名称为 DwarfStar，支持 DeepSeek V4/V4.1、GLM 5.x、Qwen3.8 的若干指定配置。它把模型加载、提示词渲染、工具调用、KV 状态、服务与本地 agent 放在一起设计，要求使用项目支持的具体 GGUF 布局。不能仅凭文件扩展名判断兼容。[固定版本 README](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/README.md#L6-L35)

优势是可以针对已知模型和硬件减少通用适配成本，连同实际交互路径一起优化。代价是模型、布局和后端组合受限，升级或更换模型需要逐项验证。支持某个模型家族不等于支持其所有 checkpoint、量化与视觉配置。

“让答案更好”要拆成量化损失、数值正确性、模板/工具协议正确性、采样行为及任务成功率；不能把更换模型后的能力提升归因于推理引擎。

### 2. 压缩注意力和 MoE 是模型机制，专用执行与状态管理才是引擎实现

DeepSeek 的注意力压缩、滑动窗口、专家路由和 hyper-connections 不是 ds4 发明的。ds4 的研究价值在于：针对这些形状与状态构建执行路径，如何管理缓存、缓冲区、量化计算和会话边界。详细区分与函数定位见 [源码地图](SOURCE_MAP.md)。

作者将“小型、可读”作为设计目标，但当前 `ds4.c` 已有 86,979 行，CUDA 文件 36,275 行，Metal 文件 50,528 行（本次对固定文件的静态行数统计，含注释和空行）。因此适合按一个问题追踪路径，不宜把“小型”理解成可以快速通读全仓库。

### 3. Q2 是混合精度配方，需要分开比较容量与质量

Flash 0731 的 Q2 配方将 routed experts 的 gate/up 压成 IQ2_XXS、down 压成 Q2_K；其余部分还有 Q8、F16、F32。它不是全模型统一 2 bit，也不能把同样标有“Q2”的两份文件当作相同权重。公开指南将约 81 GiB 的 Flash Q2 作为 96/128 GB 机器的起点，实际还要容纳 KV 与运行缓冲。[模型与量化说明](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/MODELS.md#L17-L42)

因此第一条直接引擎对照优先使用同一个 GGUF 文件并记录 SHA256。更换配方、精度或校准集后，另列为“部署方案的性能—容量—质量取舍”。

### 4. Spark 是明确优化目标，但收益仍依赖模型、精度和调用路径

`make cuda-spark` 的架构分支面向 GB10 的 `sm_121a`，启用 Blackwell 相关实现。原生 MXFP4 的批量专家计算与 Q2 路径不是同一实验条件；不能因为有 FP4 指令就推断所有低比特解码都获得相同收益。[Spark 指南](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/DGX_SPARK.md#L5-L20)、[构建分支](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/Makefile#L39-L49)

### 5. SSD streaming 解决容量问题，必须同时看 I/O 与延迟

它缓存 routed experts，读取未命中的专家，并尝试把加载与计算重叠。模型其余权重、激活和 KV 仍占用内存。更多专家缓存只有在剩余工作集也能容纳时才可能更快。[SSD 说明](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/SSD_STREAMING.md#L5-L69)

V4.1 Q2 的文件约 341 GiB，其中约 189 GiB 为磁盘 Engram 表，主权重约 152 GiB；这是另一种容量与 SSD 问题。第一轮先研究 Flash 0731 Q2，把 V4.1、PRO 和双机留到后续阶段。[V4.1 约束](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/MODELS.md#L46-L80)

### 6. 投机解码需要同时比较速度和采样语义

DSpark 是草稿/验证方法；DGX Spark 是 GB10 硬件，两者不能混称。启用投机后，非零温度的默认 opportunistic 策略不保证与普通目标采样同分布；exact-sampling 路径才以保留目标分布为目标。温度为零、随机 exact、随机 opportunistic 应分别比较。[采样策略](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/SPECULATIVE_DECODING.md)

作者的单 Spark Flash Q2 记录中，C 代码提示、温度 0 的生成速度从 19.72 提升到 31.41 token/s；不可预测散文、温度 1 则从 19.53 降到 18.81 token/s。条件是 2026-09-06、256 个生成 token、4K 分配上下文、512-token prefill chunk、三次运行中位数。这是工作负载相关的作者测量，不是 ds4 与其他引擎的对比，也不是当前提交的保证。[原表与条件](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/DGX_SPARK.md#L151-L178)

### 7. 多轮状态与服务策略可能比单次 token/s 更影响日常使用

ds4 有 live KV、磁盘会话、工具调用精确重放与回复边界处理。这些机制适合研究“代码阅读 → 调工具 → 继续回答”的真实任务。并发能力则必须按模型、后端、驻留/流式方式和会话数分析；文档中的宽泛 fallback 表已不能完整描述某些 V4.1 路径，源码裁决见 [SOURCE_MAP.md](SOURCE_MAP.md)。

## 一个已经读到具体变更的案例：回复边界为什么会重新 prefill

研究提交：[fc80bd6](https://github.com/antirez/ds4/commit/fc80bd695da76ee14bcb86820bf900a5cdfda806)，对照父提交：`d6b516abb4a286a423fdb4071a19f86a3d9ab9c0`。

投机解码一次验证多个 token。如果回复在已接受块的中间结束，或工具语法使采样模式切换，保留的 token 边界可能与内部状态推进位置不同。仅缩短文字不能保证缓存和下一 token 的 logits 正确。

该提交保留最近 verifier frontier 及 logits，在边界仍属于有效快照时，恢复对应前缀并只重放有界的少量 token；快照失效或回退位置过旧时，仍保留重建路径。这是一个带适用条件的状态恢复优化。

```text
已验证 token 块 → 回复/工具边界截断 → 状态需要回退
                                  ├─ 快照有效：恢复前缀 + 有界重放
                                  └─ 快照不可用：原有失效/重建路径
```

源码阅读链：

1. [`server_generation_rewind`](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_server.c#L12583-L12608)：服务器处理边界与重建判断。
2. [`ds4_session_restore_speculative_prefix` / `dspark_rewind_recent`](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L86725-L86821)：检查快照有效范围、恢复与重放。
3. [服务器 trace](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_server.c#L14241-L14264)：记录回退位置与 `rewind_ms`。
4. [双轮 HTTP 与停止回归](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/tests/test_server_speculative_stop.py)需要已启动的真实服务；[rewind host-mock 回归](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/tests/test_dspark_rewind.py)抽取真实 helper，但使用模拟后端状态。后者不能证明 GPU 数值/KV 正确性，前者也不直接检查内部状态。本轮均未执行。

**可检验假设：**在能触发该边界恢复的多轮请求上，当前版本应减少重复 prefill 或回退耗时；普通无边界解码可能没有可见改善。必须检查下一轮输出、logits/状态一致性和旧快照 fallback，不能只量测第一轮平均 token/s。后续对照见计划 R2。

## “效果”的四个独立问题

| 层次 | 要看什么 | 不能替代什么 |
| --- | --- | --- |
| 数值正确性 | 同输入 token 的 logits、概率、top-token、状态恢复 | 短 prompt 一致不证明长会话正确 |
| 量化保真 | 固定 teacher-forced continuation 的概率损失、top-token 一致率 | 更低 NLL 不自动等于所有任务更强 |
| 采样与协议 | exact/opportunistic、thinking、stop、模板、工具参数 | 同温度/seed 不自动等于同分布或逐 token 相同 |
| 任务效果 | 代码测试、工具调用完成、检索正确率、失败率、成功任务耗时 | 项目自测通过率不能直接变成公开排行榜分数 |

质量评分器本身也会变化。父提交 [d6b516a](https://github.com/antirez/ds4/commit/d6b516abb4a286a423fdb4071a19f86a3d9ab9c0) 修复了 API top-choice 评分对候选列表顺序的依赖。历史 top1 数字应核对评分器版本与原始数据，必要时用固定的修正评分器重算；这不意味着所有旧数字都错误。

## 本次结束位置

- 已理解：项目定位、量化与采样的比较边界、Spark/SSD 约束、会话恢复的一个真实变更，以及对比框架的兼容性门槛。
- 仍未知：用户 PGX 的实际软件环境和空闲资源、具体 GGUF 在两套引擎上的兼容性、固定版本性能、任务质量差异、Mac 的模型可运行内存预算。
- 下一步：按 [PLAN.md](PLAN.md) 完成 R0 兼容性与条件清单；先锁定 Flash 0731 同一 GGUF，再决定是否进入 PGX 试跑。
- 每次后续阅读或运行都追加证据与判断变化，不把本页的待验证假设改写成未经测量的事实。
