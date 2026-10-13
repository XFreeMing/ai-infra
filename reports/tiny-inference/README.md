# Mac 小模型推理与调试记录

日期：2026-10-13。目的：确认一个足够小的真实模型可以在本机快速运行、用 LLDB 单步查看，作为 AI Infra 入门实验。

TinyStories 15M 已在本机 CPU 上完成短文本生成；LLDB 已停在首次选择新 token 之前，观察到输入 token、模型结构与候选分数。完整操作见 [动手课](../../experiments/tiny-inference/README.md)。

## 固定条件

- 源码：llama2.c `350e04fe35433e6d2941dce5a1f53308f87058eb`，无源码修改。
- 模型：作者提供的 `stories15M.bin`，60,816,028 字节；版本、下载地址、SHA256 与 tokenizer 身份见 [模型锁](../../experiments/tiny-inference/model.lock.json)。这是 FP32 自定义 checkpoint，不是 GGUF。
- 环境：Mac16,11，arm64，64 GiB 物理内存；Apple clang 21.0.0；使用 CPU，未启用 OpenMP、Metal 或 CUDA。
- 普通版 `-O3`，调试版 `-O0 -g`；完整编译命令与二进制 SHA256 随运行记录保存。
- 输入 `Once upon a time`，temperature 为 0，seed 为 1，总 forward 步数上限为 32。

## 保存的原始证据

代码提交：`e261dddbb7f832f47fa904df336a20b0dfce68de`。下表保存的是准备与初次试跑之后的一组短运行，机器文件缓存可能已热；每个构建仅取这一轮作为操作示例，不估计稳定性能。单次运行最多 30 秒。

| 执行 | 模型进程耗时 | 记录与原始输出 |
| --- | --- | --- |
| 普通版 O3 | 0.186 秒 | [运行记录](20261013T015025Z-ed17764b/result.json)、[输出](20261013T015025Z-ed17764b/model.stdout.log)、[stderr](20261013T015025Z-ed17764b/model.stderr.log) |
| 调试版 O0 | 0.565 秒 | [运行记录](20261013T015025Z-fa1bff78/result.json)、[输出](20261013T015025Z-fa1bff78/model.stdout.log)、[stderr](20261013T015025Z-fa1bff78/model.stderr.log) |
| LLDB 断点 | 不计性能 | [命令与构建身份](debug-first-token/record.json)、[完整调试输出](debug-first-token/lldb.log) |

上面两次生成退出码均为 0，原始 stdout 逐字节相同。LLDB 命令执行成功，读取选中 token 后通过 `process kill` 主动结束被调试进程，日志中的 signal 9 是该操作的结果。

采用直接教学运行模式。当时主仓库中 `llama.cpp`、`sglang`、`vllm` 的工作区引用与父提交不同，记录中已保留；本实验没有使用它们。实际使用的 llama2.c 与固定提交内容一致。原始 JSON 记录本机绝对路径，复现时在自己的项目根目录执行动手课命令，不照抄另一台机器的绝对路径。

## 已观察到的过程

默认输入被编码为 `[1, 9038, 2501, 263, 931]`。在最后一个输入 token 计算结束后，`pos = 4`，候选分数数组长度为 32,000。最高分选择得到 token `29892`，对应逗号 `,`；本机调试观察其分数约 `18.35975`。

普通版与调试版在本轮短输入下输出一致：

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big
```

开头的 `Once upon a time` 是回显的输入，后面才是模型生成内容。32 次 forward 包含 5 次输入处理；本次未触发提前停止，产生 28 个新 token：第 5 次输入计算后就选出首个新 token。输出在步数上限截断，末尾不是完整句子。

## 解释边界

短输出可读、两个构建输出相同，足以验证本次教学路径；不代表通用能力、完整数值等价或质量评测。耗时只用于判断是否适合动手练习。断点观察受到调试器影响，不能用于性能排名。

本实验的进程时间包含程序启动、模型与词表读取、输入处理、生成和输出；不包含下载与编译。Python 运行器另外记录包括前置校验的整体时间。上游自报 tok/s 混合了输入处理和生成，未作为纯 decode 指标使用。本轮没有与 ds4、llama.cpp 或 PGX GB10 做性能比较。

已理解与验证的入口：字符串编码、普通生成循环、第一次选择新 token。未展开：embedding、attention、KV cache 的数值计算。下一步：用同一调试版进入 `forward`，只观察 token 编号如何定位一行 embedding。
