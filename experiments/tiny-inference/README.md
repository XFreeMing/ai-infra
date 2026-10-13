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

用 VS Code 打开整个 `ai-infra` 项目目录，使用 CodeLLDB 扩展（`vadimcn.vscode-lldb`，当前 Mac 已安装）。[launch.json](../../.vscode/launch.json) 提供两个入口：

| 配置 | 停在哪里 | 适合做什么 |
| --- | --- | --- |
| `TinyStories：第一个新 token` | `run.c:753`，最后一个输入 token 计算完毕 | 从第一节开始，观察输入与生成的分界 |
| `TinyStories：从 main 开始` | `main` 函数入口 | 逐步看参数、模型加载和初始化 |

在左侧“运行和调试”选择第一个配置，按 **F5**。启动前 [tasks.json](../../.vscode/tasks.json) 自动执行 `lab.py prepare`，校验固定源码和模型，重新编译带符号的调试版；模型已存在时无需再次下载。准备任务需要 `uv` 在 VS Code 的 PATH 中，新机器也需要先初始化 llama2.c 子模块。

停下后，在 Variables 中查看 `pos`、`token`、`num_prompt_tokens`，也可在 Watch 中添加 `transformer->config`。按 **F10** 两次执行完 `sample`，再添加 `next`、`logits[next]`、`tokenizer->vocab[next]`；不要在采样执行前读取尚未赋值的 `next`。**F11** 进入函数，**Shift+F11** 返回调用者，**F5** 继续。Mac 若把功能键用于系统控制，需要同时按 Fn。

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

## 与 ds4 学习连接

llama2.c 在同一个循环中逐个处理输入和输出，没有 ds4 那样独立的批量 prefill API。两者都需要先处理上下文，再不断计算候选分数并选出后续 token；调用形态和性能路径不同。

这个实验加载作者自定义的 FP32 `.bin`，不是 GGUF，也不用于 R0 的同 GGUF 对比。先用它理解计算过程，再回到 [ds4 第 1 课](../../research/learning/01-request-to-tokens.md)，之后在 llama.cpp 中学习 GGUF、Metal 与服务接口。

## 来源与继续入口

- 源码固定为 [`350e04fe35433e6d2941dce5a1f53308f87058eb`](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb)，通过 submodule 管理，无上游源码修改。
- [作者的项目说明](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/README.md)解释其教学目的；[官方模型](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add)提供权重。完整文件身份见 lock 文件。
- [本机运行证据](../../reports/tiny-inference/README.md)区分普通生成、调试观察和测量边界。

已实际验证：模型可运行，断点可查看输入 token、结构参数、logits 和首次选中结果。尚待学习：向量运算、attention 与缓存的计算细节。下一次只读 `forward` 的 embedding 拷贝部分；不要求一次读完整个文件。
