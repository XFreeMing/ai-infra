# 从 token 编号到向量计算和 RMSNorm

学习日期：2026-10-13。通过 Mac 上的 TinyStories 15M 和 llama2.c，把后端开发熟悉的数组、索引、文件布局连接到模型推理。本文记录本次问答与已核对的数据，理解程度仍需要逐步调试和复述来确认。

后续学习见 [2026-10-16 的 QKV 个人复盘](03-qkv-personal-review.md)：根据新的追问，先补齐矩阵乘法、数据身份与 attention 的完整去向，再继续本篇的 RMSNorm 调试。

源码基准为 llama2.c 提交 `350e04fe35433e6d2941dce5a1f53308f87058eb` 的 [run.c](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c)。模型与 tokenizer 的下载来源、版本、SHA256 见 [model.lock.json](../../experiments/tiny-inference/model.lock.json)，运行入口见 [小模型动手课](../../experiments/tiny-inference/README.md)。以下数值观察来自本机文件读取，不是 PGX GB10 的性能结果。

## dim 是向量有多少个数

问题：`dim = 288` 的“宽度”是什么？

`dim` 是 dimension 的缩写。在这里，它表示模型主要计算通路中，一个 token 的向量有 288 个元素。可以先把它理解成 `float x[288]`。这是元素个数，不是数学上表示向量大小的“模长”。

当前模型的配置如下。这些值规定了权重的形状，不能像命令行参数一样随意修改。

| 配置 | 当前值 | 含义 |
| --- | --- | --- |
| `dim` | 288 | 每层输入和输出的主要向量维度 |
| `hidden_dim` | 768 | FFN 内部中间向量的维度 |
| `n_layers` | 6 | Transformer 层数 |
| `n_heads` | 6 | Query 的 attention head 数量 |
| `n_kv_heads` | 6 | Key 和 Value 的 head 数量 |
| `vocab_size` | 32000 | 词表中的 token 数量 |
| `seq_len` | 256 | 此实现按模型配置分配的最大序列容量 |

所以“宽度 288、深度 6”表示一个 288 元素的向量依次经过 6 层。层内部可以暂时变成其他维度，例如 FFN 的 768；回到主通路时仍然是 288。

## 字符串如何对应 token 编号

问题：怎么知道 `time` 对应哪个编号？

需要区分两张表：tokenizer 的词表负责“字符串与编号”，模型的 Embedding 表负责“编号与向量”。

本次读取 `sources/llama2.c/tokenizer.bin` 得到以下内容。空格是字符串的一部分：

| token ID | 词表字符串 |
| --- | --- |
| 263 | `" a"` |
| 931 | `" time"` |
| 2230 | `"time"` |
| 2501 | `" upon"` |
| 9038 | `" Once"` |

**此前把 931 简写成 `time` 不够准确：931 对应带前导空格的 `" time"`，不带空格的 `"time"` 是 2230。** 一个词也不保证对应一个 token，可能拆成多个。

[`build_tokenizer`](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L385) 顺序读取文件中的字符串和分数，放入 `vocab[i]` 与 `vocab_scores[i]`。第 `i` 个条目的编号就是 `i`。为方便查找而排序的 `sorted_vocab` 保留原始 ID，排序位置不会成为新的 ID。

[`encode`](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L452) 先构造初始 token 序列，再按词表分数选择可合并的相邻片段。这里的词表分数用于分词，不是模型预测下一个 token 的 logits。

## token 编号如何找到 288 个数

问题：编号 931 对应的一行数据从哪里找到？

本机权重文件位于项目的 `models/tiny-inference/stories15M.bin`，约 61 MB，属于不提交到 Git 的本地模型文件。它包含模型配置和权重；tokenizer 则是前述单独的文件。

Embedding 表的逻辑形状是 `[32000, 288]`：每个 token ID 对应一行训练得到的数值。C 实现把它平铺成一段连续内存。[`forward`](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L231) 中的查表操作是：

```c
float* content_row = w->token_embedding_table + token * dim;
memcpy(x, content_row, dim * sizeof(*x));
```

当 `token = 931`，行首偏移是 `931 × 288 = 268128` 个 float。复制接下来的 288 个数，就是当前 token 的初始 `x`。编号只负责索引，不会通过某个公式把整数 931 变成这些数。

[`read_checkpoint` 与 `memory_map_weights`](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L111) 负责读取配置、映射权重文件，再将权重指针指向各段数据。

## bin 与 GGUF 和 safetensors 的关系

问题：换一种模型格式，词表在哪里？

`.bin` 只是通用扩展名，没有统一的内部规范。本例的模型文件和 tokenizer 文件采用 llama2.c 自己约定的布局。GGUF 和 safetensors 则有各自明确的格式。

| 格式或使用方式 | 模型权重 | tokenizer 通常在哪里 |
| --- | --- | --- |
| 当前 llama2.c | `stories15M.bin` 中的配置和连续权重 | 独立的 `tokenizer.bin` |
| 常见文本模型 GGUF | 文件中的命名 tensor 与元数据 | 通常放在同一个 GGUF 的 tokenizer 元数据中 |
| Hugging Face safetensors 模型目录 | 一个或多个 `.safetensors` 文件；模型配置通常在 `config.json` | 通常是 `tokenizer.json`、`tokenizer_config.json`，或 `tokenizer.model`、`vocab.json` 和 `merges.txt` 等配套文件 |

GGUF 的相关字段包括 `tokenizer.ggml.tokens`、`tokenizer.ggml.scores`、`tokenizer.ggml.merges` 和 `tokenizer.ggml.model`，具体字段取决于 tokenizer 类型。字段核对版本为 llama.cpp `071b553fb1d3a1b76c0b12ab9a1546714cafe4e1` 的 [constants.py](https://github.com/ggml-org/llama.cpp/blob/071b553fb1d3a1b76c0b12ab9a1546714cafe4e1/gguf-py/gguf/constants.py#L338)。并非每个 GGUF 都必须包含完整词表。

safetensors 主要规定 tensor 的保存方式，不能只拿到权重就假定已经拿到 tokenizer。词表之外，还需要正确的切分、合并、特殊 token 等规则。无论文件格式如何变化，token ID 与 Embedding 行号都必须保持一致。参考 [GGUF 介绍](https://huggingface.co/docs/hub/gguf)、[safetensors 文档](https://huggingface.co/docs/safetensors/index) 和 [Transformers tokenizer 文档](https://huggingface.co/docs/transformers/main_classes/tokenizer)。

## 多个 token 怎样经过每一层

问题：一个 token 是一行，多个 token 是多行，那多行怎样过层？

逻辑上，T 个 token 查表后可以表示成 `[T, 288]`。但当前 llama2.c 的实际执行顺序是：第一个 token 走完 6 层，再让第二个 token 走完 6 层，以此类推。当前工作向量 `x` 只保存一个 token 的 288 个数；历史信息通过各层 KV cache 留下。

一些引擎可以把 prompt 的多个位置按矩阵批量处理，逐层完成 prefill，同时通过因果遮罩限制每个位置只能利用自身和更早的位置。这个批量视角有助于理解，但不能当作当前 `run.c` 的执行顺序。

“经过一层”可以初步理解为：根据权重和上下文，计算出新的 288 个数。可能每个数都会变化；不能假定某个固定下标对应一个明确的人类概念。Attention 引入上下文信息，FFN 对当前位置的特征做进一步变换，残差连接将计算结果加回主通路。

## Q K V 从哪里来以及每层缓存什么

问题：第一层的 Q、K、V 也是 288 个数吗？KV 只缓存第六层之后的数据吗？

当前模型中 Q、K、V 都是 288 维，因为 `n_heads = n_kv_heads = 6`，每个 head 的维度为 `288 / 6 = 48`。在采用 GQA 的其他配置里，K、V 的总维度可以比 Q 小。

第一层的计算入口是：

```text
Embedding 查表得到 x[288]
        ↓ RMSNorm
      xb[288]
        ├─ Wq × xb → Q[288]
        ├─ Wk × xb → K[288]
        └─ Wv × xb → V[288]
```

这里三组矩阵是训练得到的权重，当前形状各为 `[288, 288]`。每层都有自己的矩阵。矩阵乘法的一个输出元素是 `q[i] = Σ Wq[i,j] × xb[j]`，因此它通常综合了输入的全部维度。后续层接收上一层的输出，而不是重新查 Embedding 表。

KV cache **每层都有**，按层和位置保存 K、V，并非只保存最后一层的输出。当前两个缓存各自的逻辑形状是 `[6, 256, 288]`。计算当前位置时，Q 会与同一层已经缓存的历史 K 配合，再利用 V 汇总信息。Q 不作为历史 KV cache 保存。

代码先将投影得到的 K、V 写入对应缓存位置，再对 Q 和 K 做 RoPE 位置变换；V 不做这一步。因此缓存中最终供 attention 使用的 K 已包含该位置变换。见 [`forward` 的投影、RoPE 与 attention](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L250)。

## RMSNorm 做了什么

问题：既然 x 已经从表里查出来，为什么还需要归一化？

RMS 是 Root Mean Square，即均方根。先平方、求平均，再开平方，用一个数描述整组数值的尺度。当前 [`rmsnorm`](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L182) 的计算是：

```text
r = sqrt(mean(x[j]²) + 0.00001)
output[j] = weight[j] × x[j] / r
```

`weight[j]` 是训练得到的逐元素缩放参数，通常记为 γ。它不是把输出限制在某个固定区间，也不是把元素变成总和为 1 的概率。RMSNorm 不做减均值；输出元素个数仍为 288。在当前调用中，结果写到 `xb`，原来的 `x` 留给残差连接。

忽略 epsilon 和 γ，用二维例子看：

```text
x = [3, 4]
RMS = sqrt((9 + 16) / 2) ≈ 3.5355
x / RMS ≈ [0.8485, 1.1314]

x = [30, 40]
RMS ≈ 35.3553
x / RMS ≈ [0.8485, 1.1314]
```

正比例放大输入时，RMS 也按同样比例放大，所以除法消除了这个整体倍数。实际实现含有 epsilon，两者只是近似相同；再乘上同一组 γ 后仍近似相同，但数值不再必然是上面的两个数。

“放大 10 倍”是解释性质的假设例子，不表示推理中有一条随机乘 10 的操作。模型训练时就在这些归一化操作参与下学习权重，查出来的 Embedding 并不要求天然具有单位 RMS。理论背景见 Zhang 与 Sennrich 的 [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。

### 从本地权重读到的数值

本次用 Python 标准库按文件头后 Embedding 的布局读取 FP32 数值，再计算 RMS。下面是文件观察值，计算使用 Python 浮点累加；不是 C 内核逐位一致性的验证。

| token ID | 原始向量前四项，近似值 | 288 项的 RMS，不含 epsilon |
| --- | --- | --- |
| 1 | `[-0.017096, -0.013531, 0.006738, -0.004548]` | 0.02342070 |
| 931 | `[0.004835, -0.024510, 0.015095, 0.000862]` | 0.02766505 |

这些行的 RMS 确实不是 1。以 token 1 为例，加上 epsilon 后的除数约为 `0.02363322`，除法后的前四项约为 `[-0.723405, -0.572559, 0.285119, -0.192434]`；这还没有乘 γ，不能直接当作最终的 `xb`。

## 为什么不在模型出厂时全部归一化

用户提出的想法：既然权重固定，可以预先算好，省掉推理时的 RMSNorm 吗？

**源码推导：当前架构中，第一层 attention 之前的 RMSNorm 可以考虑预先计算。** 它的输入是固定 Embedding 行，γ 也是固定权重，位置变换发生在后面的 Q、K 投影之后。

但同一层 attention 已经会根据上下文产生不同结果，残差相加之后，FFN 前的 RMSNorm 输入就变成运行时数据。更高层及最终输出前的归一化也需要根据实际输入计算，无法仅根据 token ID 提前存好。

还不能直接用归一化后的表覆盖原始 Embedding 表：原始 `x` 要用于残差相加。覆盖它会改变模型计算。候选方案是保留原表，另存第一层归一化结果，计算 attention 时读取这张新表。

**待验证假设，尚未实现或测量：** 额外缓存可以省掉每个 token 的一次 RMSNorm，但新 FP32 表需要 `32000 × 288 × 4 = 36,864,000` 字节，约 35.16 MiB；是否加速取决于访存、缓存和初始化成本，不能仅凭少算一个函数得出结论。

如果后续开展实验，应固定模型、源码、输入和编译配置，先保留 epsilon、γ 与原始残差路径，对比 logits 数值误差及生成结果，再分别测量初始化成本、推理延迟和额外内存。该实验目前只作为学习产生的候选研究问题。

## 已讲清的内容与下一步

本次已讲解并核对：dim 是元素个数；词表负责文本与 ID，Embedding 负责 ID 与向量；当前实现逐 token 通过所有层；KV 按层缓存；Q、K、V 来自各层矩阵乘法；RMSNorm 根据当前向量计算尺度，再乘训练得到的 γ。是否掌握，以下次能否独立追踪数值为准。

尚未展开：attention 怎样用 Q 与 K 算出权重、V 怎样参与汇总、RoPE 如何编码位置、FFN 怎样变换特征。预先缓存第一层归一化的收益与正确性也尚未验证。

**下次只读 `rmsnorm` 一个函数。** 在第一层 attention 前停住，观察 `x[0..3]`、`size = 288` 和 `weight[0..3]`；进入函数观察 `ss` 从平方和变成均值、加 epsilon、再变成 RMS 的倒数；返回后核对 `xb[0] = weight[0] × x[0] × ss`，并确认原始 `x[0]` 未被修改。先解释这一条数值链，再进入 `matmul`。
