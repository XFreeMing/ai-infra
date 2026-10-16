# 在 Mac 上调试一个小模型

用 TinyStories 15M 和 llama2.c 在 CPU 上执行真实推理，每次只观察一个环节。模型文件 60,816,028 字节，约 61 MB；源码主文件 `run.c` 为 973 行。无需 Python 机器学习库、Metal 或 CUDA。模型适合简单英文故事续写，不能用来判断通用聊天、中文能力或 ds4 的性能。

## 准备与运行

在项目根目录执行，需要 Python 3.12+、Git、curl 和 clang；调试另需 LLDB。当前 Mac 已准备完成。新机器先执行：

```sh
git submodule update --init sources/llama2.c
uv run --offline python experiments/tiny-inference/lab.py prepare
```

准备工具按 [model.lock.json](model.lock.json) 下载固定版本模型并核验大小和 SHA256，随后生成普通版与调试版。权重在 `models/tiny-inference/`，二进制与构建记录在 `artifacts/tiny-inference/`，均不进入 Git。已存在且校验通过的模型会复用。

日常只需这一条命令：

```sh
uv run --offline python experiments/tiny-inference/lab.py run
```

默认输入 `Once upon a time`，temperature 为 0（每次取最高分），seed 为 1，总 forward 步数上限为 32，单次进程超时为 30 秒。`--debug-build` 使用无优化且带符号的调试版；平时用默认普通版。

每次留下新的运行目录，包含 `result.json`、`model.stdout.log` 和 `model.stderr.log`，记录实际程序、校验值、构建参数、模型、输入、机器、主仓库版本及未提交状态。直接学习模式允许主仓库有改动，并如实记录；正式执行器的干净工作区要求仍然保留。在干净、已提交的独立 checkout 中，可用 `uv run --offline python main.py run tiny-inference` 纳入统一执行器。

本引擎的 `--steps` 对应上游 `-n`，**包含输入处理的步数**。32 步不等于生成 32 个新 token。输出包含输入的回显，内置 `achieved tok/s` 混合输入处理与生成，不能当成纯 decode 基准。小练习只看实际耗时是否适合交互，不据此比较 ds4、llama.cpp 或 GB10。

## 在 VS Code 中按 F5 调试

用 VS Code 打开整个 `ai-infra` 项目目录，使用 CodeLLDB 扩展（`vadimcn.vscode-lldb`，当前 Mac 已安装）。[launch.json](../../.vscode/launch.json) 提供三个入口：

| 配置 | 停在哪里 | 适合做什么 |
| --- | --- | --- |
| `TinyStories：QKV 从哪里来` | `run.c:260`，`pos == 1 && l == 0` | 观察 `Once` 在第 0 层由同一输入算出 Q/K/V |
| `TinyStories：第一个新 token` | `run.c:753`，最后一个输入 token 计算完毕 | 从第一节开始，观察输入与生成的分界 |
| `TinyStories：从 main 开始` | `main` 函数入口 | 逐步看参数、模型加载和初始化 |

在左侧“运行和调试”选择所学小节对应的配置，按 **F5**。启动前 [tasks.json](../../.vscode/tasks.json) 自动执行 `lab.py prepare`，校验固定源码和模型，重新编译带符号的调试版；模型已存在时无需再次下载。准备任务需要 `uv` 在 VS Code 的 PATH 中，新机器也需要先初始化 llama2.c 子模块。

VS Code 的准备任务使用 `uv run --offline --no-sync`，复用已有 Python 环境执行标准库脚本，避免调试前触发主项目的包构建。当前 Mac 已有 `.venv`；新机器应先完成项目 Python 环境初始化。

使用“第一个新 token”配置时，停下后在 Variables 中查看 `pos`、`token`、`num_prompt_tokens`，也可在 Watch 中添加 `transformer->config`。按 **F10** 两次执行完 `sample`，再添加 `next`、`logits[next]`、`tokenizer->vocab[next]`；不要在采样执行前读取尚未赋值的 `next`。**F11** 进入函数，**Shift+F11** 返回调用者，**F5** 继续。Mac 若把功能键用于系统控制，需要同时按 Fn。

使用“QKV 从哪里来”配置时，先结束旧调试会话，再选择新配置启动；若旧的手动断点提前命中，继续到 `run.c:260`。此时 `token=9038`（Once）、`pos=1`、`l=0`、`dim=kv_dim=288`、`head_size=48`。`s->xb` 已经过 RMSNorm，是三次矩阵乘法共用的输入；`w->wq/wk/wv` 是三组训练好的权重。连续 F10 三次，到 265 行，才完成当前 Q/K/V 的计算。260 行的 `s->q` 可能残留前一 token 的计算，当前 K/V 位置也尚未写入，不能提前解释为当前结果。

深入观察 Q 的计算时，在 260 行按 F11 进入 `matmul`，再 F10 到 225 行的 `val += w[i * n + j] * x[j]`。第一次到达这里时 `i=j=0`、`val=0`；内部 `x` 是外部的 `s->xb`，内部 `w` 是该层的 Q 权重数组。执行一次乘加，观察 `val` 改变。内层循环累加 288 项后，227 行才写入 `xout[0]`，即 `s->q[0]`；外层循环重复这一过程，得到全部 288 个 Q 分量。K/V 使用同样的输入，通过各自训练矩阵执行相同形式的计算。

本节源码为 llama2.c `350e04fe35433e6d2941dce5a1f53308f87058eb`。Q 暂存于 `s->q`，K/V 通过 `s->k/s->v` 直接写到缓存中当前层、当前位置。265–279 行随后用 RoPE 变换 Q/K；比较 attention 分数时，应使用变换完成后的 Q 与缓存 K。下一段观察第 0 个头：当前 Q 与开始标记、Once 两个位置的 K 分别计算分数，softmax 后作为权重汇总两个 V。入口是 [固定源码](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L260-L317)；更换 layer/pos 时需重新确认指针与循环变量的作用域。

VS Code 直接调试 C 程序，输出在集成终端，不经过 `lab.py run` 的 30 秒限制，也不会自动保存它的运行记录。结束练习可按 **Shift+F5**；需要实验日志时仍使用上面的 `lab.py run`。配置中的断点对应固定源码版本，升级 llama2.c 时需要重新核对行号。配置字段依据 [CodeLLDB 官方说明](https://github.com/vadimcn/codelldb/blob/master/MANUAL.md#starting-a-new-debug-session)。

## 第一节 10 分钟

先运行一次，看到故事继续。然后只读固定源码的 [`generate` 中这一段](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L747-L769)：

```c
float* logits = forward(transformer, token, pos);
if (pos < num_prompt_tokens - 1) {
    next = prompt_tokens[pos + 1];
} else {
    next = sample(sampler, logits);
}
```

`token` 是当前输入片段的编号；`pos` 是它在序列里的位置，从 0 开始；`forward` 根据已有上下文计算下一片段的候选分数 `logits`。输入还没读完时，下一片段来自你提供的文字；读到输入末尾时，才用 `sample` 选出模型自己的下一片段。

在项目根目录打开调试器：

```sh
lldb -s experiments/tiny-inference/first-token.lldb -- \
  artifacts/tiny-inference/run-debug \
  models/tiny-inference/stories15M.bin \
  -z sources/llama2.c/tokenizer.bin \
  -t 0 -s 1 -n 8 -i 'Once upon a time'
```

脚本会停在 `run.c:753`：最后一个输入 token 的 `forward` 已返回，第一次选择新 token 尚未发生。默认输入对应：

| 观察值 | 本机实测 | 先这样理解 |
| --- | --- | --- |
| `num_prompt_tokens` | 5 | 4 个词片段加 1 个开头标记；其他句子不一定按词切分 |
| `prompt_tokens` | `[1, 9038, 2501, 263, 931]` | 字符串编码后的整数序列 |
| `pos` | 4 | 当前处理最后一个输入片段 |
| `token` | 931 | 当前片段 `time` 的编号 |
| `vocab_size` | 32000 | 有 32,000 个候选编号，每个对应一个分数 |

此时终端已经显示 `Once upon a time`，这是输入回显，还不是模型生成的内容。脚本只打印前 8 个 logits，它们既不是概率，也不是排名前 8 的候选。

在出现 `(lldb)` 提示符后逐行输入：

```text
next
next
p next
p logits[next]
p tokenizer->vocab[next]
```

第一条 `next` 执行条件判断；第二条执行完 `sample`，停在 `pos++`。本机观察到 `next = 29892`，分数约 `18.35975`，词表中的文本是逗号 `,`。这里 LLDB 命令 `next` 表示执行下一行，而 C 变量 `next` 是选中的 token 编号，两个名字碰巧一样。

用 `continue` 看剩余输出，完成后用 `quit` 退出。若中途停止，可用 `process kill`；这是主动结束教学进程。保持此脚本的默认输入：其中 `parray 5` 专门用于这 5 个输入 token，换短输入时应先检查数量再调整，避免读取数组之外的数据。

本机已验证 LLDB 可以启动该程序。自动化沙盒内启动曾失败，使用本机调试权限后成功；若在受限环境中复现失败，先在 Mac 自己的终端执行上述命令，不需要调整模型。

这节结束只需能解释：**为什么 `pos = 0` 时不能让模型自己挑下一个输入 token，而 `pos = 4` 时可以？**

## 后续每次只加一个观察点

| 小节 | 大约时间 | 动手操作 | 学会什么 |
| --- | --- | --- | --- |
| 输入与生成 | 10 分钟 | 完成本页第一节 | 输入 token 和生成 token 的分界 |
| 一次计算 | 15 分钟 | 断在 `forward`，观察 `config`，再看 embedding 拷贝 | 权重与运行状态有不同用途；向量是浮点数组 |
| 选出结果 | 15 分钟 | 进入 `sample` 与 `sample_argmax` | logits 与 token ID 不同；temperature 为 0 时取最大值 |
| 使用上下文 | 20 分钟 | 在 `forward` 的第 0 层观察 `pos`、`s->k` 与 `s->v` | KV cache 保存之前计算的 Key/Value，并继续参与 attention |
| 最小假设实验 | 20 分钟 | 固定输入，用 8 与 32 steps 分别运行并读日志 | 定义变量、分清总步数和新增输出、保留证据 |

进入 `forward` 后再看局部变量，避免刚到函数入口就读尚未赋值的指针；查看 `logits` 也要停在调用完成后的 753 行，单纯 `finish` 返回可能仍停在赋值尚未结束的 750 行。

## 用数据库查看 token 和初始 embedding

在项目根目录执行，只读取已准备好的固定模型和词表，不会运行或重新编译推理程序：

```sh
uv run --offline --no-sync python experiments/tiny-inference/export_embeddings.py
```

输出为 `artifacts/tiny-inference/token-embeddings.sqlite3`。已存在时保留原文件，可用 `--output artifacts/tiny-inference/token-embeddings-2.sqlite3` 另存。数据库是本机学习产物，不提交 Git；脚本、示例查询随父仓库版本管理。

用 SQLite 数据库工具打开该文件：

- `tokens` 表：32,000 行，每行一个 token。`token_id` 是编号，`token_text` 是词表原文，`token_text_escaped` 用 JSON 引号及转义符显示前导空格、换行；`embedding_000` 到 `embedding_287` 是 288 个初始 embedding 数值。
- `once_embedding` 视图：只展示 `9038`，纵向 288 行。`dim_index = 0` 对应数组第一个元素，`value` 是该元素的值，便于逐行观察。
- `embedding_values` 视图：对其他 token 使用相同的纵向布局。按 `token_id` 筛选即可，无需实际存储第二份权重。
- `metadata` 表：模型配置、SHA256、来源 commit、脚本哈希、环境与导出记录。`tokenizer_score` 是词表分词用的分数，不是生成概率。

[inspect_embeddings.sql](inspect_embeddings.sql) 包含可直接执行的查询。例如：

```sql
SELECT token_id, token_text_escaped FROM tokens WHERE token_id IN (938, 9038);
SELECT * FROM once_embedding;
SELECT dim_index, value FROM embedding_values WHERE token_id = 2501 ORDER BY dim_index;
```

`9038` 的词表原文是 `" Once"`，包含一个前导空格；`938` 是 `" int"`。`decode` 在 BOS 后会移除一个前导空格，所以第一次打印看到的是 `Once`。某些 token 是词的一部分，某些是特殊标记或 `<0xXX>` 字节，不保证每个 token 都能单独显示成完整文字；导出保留原文与字节十六进制，不做有损替换。

这些数来自模型中的 **`token_embedding_table` 固定初始权重**，不是依赖上下文计算的隐藏状态，也不是 Q、K、V。在 `forward` 的 `memcpy` 执行后、第一层计算前，`s->x[i]` 应等于数据库里当前 token 的 `embedding_i`。执行 RMSNorm 或后续层以后，再用该固定表对比运行状态就没有同样含义。

解析依据是固定提交 [`350e04fe35433e6d2941dce5a1f53308f87058eb` 的 run.c](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c)：`Config`、`read_checkpoint`、`memory_map_weights`、`build_tokenizer`、`decode` 与 `forward`。模型头为 7 个小端 int32，随后第一块权重就是按 token ID 顺序存放的 FP32 embedding。导出精确读取 FP32 并存为 SQLite REAL，不人为舍入；此脚本只针对 lock 固定的旧版 `.bin`，不解析 GGUF。

本节理解目标：能把 token ID、词表文本和初始数值数组对应起来。尚未解释：这些数如何在训练中学到、如何变成 Q/K/V。下一步在 `forward` 的 embedding 拷贝后，对比 `9038` 的前三个 `s->x` 数值与数据库。

## 与 ds4 学习连接

llama2.c 在同一个循环中逐个处理输入和输出，没有 ds4 那样独立的批量 prefill API。两者都需要先处理上下文，再不断计算候选分数并选出后续 token；调用形态和性能路径不同。

这个实验加载作者自定义的 FP32 `.bin`，不是 GGUF，也不用于 R0 的同 GGUF 对比。先用它理解计算过程，再回到 [ds4 第 1 课](../../research/learning/01-request-to-tokens.md)，之后在 llama.cpp 中学习 GGUF、Metal 与服务接口。

## 来源与继续入口

- 源码固定为 [`350e04fe35433e6d2941dce5a1f53308f87058eb`](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb)，通过 submodule 管理，无上游源码修改。
- [作者的项目说明](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/README.md)解释其教学目的；[官方模型](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add)提供权重。完整文件身份见 lock 文件。
- [本机运行证据](../../reports/tiny-inference/README.md)区分普通生成、调试观察和测量边界。

已实际验证：模型可运行，断点可查看输入 token、结构参数、logits 和首次选中结果。尚待学习：向量运算、attention 与缓存的计算细节。下一次只读 `forward` 的 embedding 拷贝部分；不要求一次读完整个文件。
