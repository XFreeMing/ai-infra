# AI 300：Ideogram4 2K OOM 与 ROCm attention 验证

实验 ID：`media-rocm-attention`。记录日期：2026-10-16（Asia/Shanghai）。本实验先使用不加载模型的 BF16 SDPA 探针检查后端可用性、短序列正确性和长序列内存，再进行完整 1344×2016 图片验证。探针成功不等于完整模型生成成功。

**2026-10-16 事后更正：** 完整验证期间 09:54:33 已发生 AMDGPU ring timeout/reset 和 Xorg 崩溃；之后第二次运行 10:23 又复现并导致生图进程 SIGABRT。此前只从应用日志得出的“没有 OOM、最终超时”遗漏了系统层故障，不能解释成正常持续计算或稳定性改善。普通 2K/WebUI 启动现已加保护；暂勿按本文旧的 3600 秒建议直接重试。完整证据和恢复步骤见 [GPU 卡死调查](gpu-freeze.md)。以下保留当时的原始实验结果。

**第三次补充及恢复进展：** 停止后台 Qwen 后，10:44 仍发生 GPU 重置，GTT 峰值仅 47.22/96 GiB。hello 已增加 CPU VAE 解码旁路和细分阶段记录。用户随后于 11:14 执行 384×576、GPU 编码/采样加 CPU 解码，完整生成通过：总耗时 174.430 s、生成 40.102 s、CPU 解码 1.146 s、GTT 峰值 29.993 GiB。此后用户于 11:54:10–12:04:05 执行同样 CPU 解码的 1344×2016，run `20261016T035410Z-ideogram4-rjbq82iy` 也完整通过：总耗时 595.098 s、生成 482.235 s、CPU 解码 18.365 s、GTT 峰值 37.868 GiB。对应日志窗口未见 GPU reset/hang、OOM 或 Xorg 崩溃；主标题仍有乱码，不能直接作为文字验收合格的成品。两种尺寸各成功一次，不能证明历史故障根因或连续稳定性，2K 保护保留。运行身份见 [2K 复核](../../artifacts/reports/ai300-media-20261016/latest-2k-audit/summary.json)，输入、图片和性能对比见[产品实测报告](../../artifacts/reports/ai300-media-20261016/deliverables/AI300生图与配音实测报告.html)。

## 运行身份与证据

- 原失败 run：[`20261016T013030Z-ideogram4-cwyx0r5p`](../../artifacts/hello/300/20261016T013030Z-ideogram4-cwyx0r5p/manifest.json)。原父提交为 `da83123838ab4726f718eb26751f37bf427359fc`，该次 manifest 记录工作区有未提交修改。
- 诊断 run：`ideogram4-oom-20261016`。原始探针 JSON、日志和验证上下文位于 [`artifacts/research/ideogram4-oom-20261016/`](../../artifacts/research/ideogram4-oom-20261016/)。这些是机器本地产物，发布报告前需另行提供可访问的证据副本。
- 完整验证使用独立、干净的代码快照 `7d0684101cdf8d1579c301636b53ffc154ea1b85`，目录为 `artifacts/research/ideogram4-oom-20261016/verification-checkout`。父提交、各修改文件 SHA256、环境路径和停止规则见 [`verification-context.json`](../../artifacts/research/ideogram4-oom-20261016/verification-context.json)。运行中不修改该快照或其依赖环境。
- 探针代码：[probe.py](probe.py)，本次 SHA256 为 `67748dff88ec35dcb690f8b42c0ca7307d2100a94e9d5f99b3bac383886d4d32`。
- 实际 Python：`artifacts/init/300/20261015T234635Z-ideogram4-s46lqg0u/venv/bin/python`，版本 3.12.3。实际 Torch 路径为该环境的 `lib/python3.12/site-packages/torch/__init__.py`，版本 `2.11.0+rocm7.2`，Torch 构建提交 `70d99e998b4955e0049d13a98d77ae1b14db1f45`，HIP `7.2.26015`，`torch.version.cuda` 为 null。
- 实际 GPU：AMD Radeon 8060S，`gfx1151`，HIP device 0，可见总容量 96 GiB。整机为 Ryzen AI Max+ 395、128 GB 统一内存；这不是 GB10 实验。
- Ideogram4 固定源码提交：[`990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2)。模型为 `ideogram-ai/ideogram-4-fp8`，ModelScope revision `0697355979847a8e4db198738c1d5ed4453ffa24`，路径 `/home/ops396/models/Ideogram4`；文件身份见原失败 manifest 的 `models`。

## 原失败原因

**实际观察：** 原 run 加载模型成功，耗时 128.071 秒；生成阶段运行 25.391 秒后抛出 `torch.OutOfMemoryError`。完整进程耗时 157.914 秒，退出码 1，并未触发 900 秒超时。异常位置为 Qwen3-VL 文本编码中的 `torch.nn.functional.scaled_dot_product_attention`，尚未完成图片生成。见 [原始日志](../../artifacts/hello/300/20261016T013030Z-ideogram4-cwyx0r5p/00-generate.log) 和 [性能 JSON](../../artifacts/hello/300/20261016T013030Z-ideogram4-cwyx0r5p/performance.json)。

| 测量 | 原失败 run |
|---|---:|
| OOM 当时额外分配请求 | 3.90 GiB |
| OOM 当时 HIP 剩余容量 | 3.83 GiB |
| 启动前整设备 GTT 占用 | 30.148 GiB |
| 整设备 GTT 采样峰值 | 94.567 / 96 GiB |
| 生成阶段 PyTorch allocated / reserved 峰值 | 62.55 / 64.23 GiB |
| 整机可用内存最低值 | 12.732 GiB |
| Swap 使用量 | 开始 5.128 GiB，峰值约 8.000 GiB |

OOM 瞬间日志记录 PyTorch 已分配 58.74 GiB，另有保留但未分配的 4.46 GiB。这与阶段峰值属于不同时间点。`reserved` 包含 `allocated`，不能相加；统一内存上的 RSS、GTT、整机内存也有重叠，不能加总。GTT 是整设备读数，已有约 30 GiB 背景占用不能全部归属于本次生成，现有证据也不能确定它属于哪个后台进程。资源数据每 2 秒采样一次，共 80 次，无采集错误；采样峰值可能遗漏瞬时变化。

**固定源码事实：** [`Ideogram4Pipeline._build_inputs`](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/src/ideogram4/pipeline_ideogram4.py#L343) 将文本 tokens 与图像位置拼接；1344×2016 对应 `(1344/16)×(2016/16)=10584` 个图像 tokens，本次 caption 的 849 个文本 tokens 使总序列长度达到 11433。随后 [`_get_qwen3_vl_embeddings`](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/src/ideogram4/pipeline_ideogram4.py#L414) 在该序列上构造 causal mask 并调用 Qwen decoder。去噪网络也在 [`Ideogram4Attention.forward`](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/src/ideogram4/modeling_ideogram4.py#L130) 使用带 mask 的 SDPA。

**诊断推论与对照证据：** 原日志提示此 AMD GPU 的 Flash / Memory Efficient attention 需要显式启用 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`；原运行环境未配置该变量。下方 env=0/1 探针对照证实，在当前构建和硬件上，未开启时优化后端不可用，开启后带 mask 的 `EFFICIENT_ATTENTION` 可运行。结合长序列与 OOM 位置，原运行的高内存 attention 路径是本次峰值的重要原因；原失败未进行算子 profiler 追踪，因此不把堆栈本身当成某个内核名称的直接测量。

增加超时时间或更换下载源无法解决这次内存分配失败。错误在去噪前的文本编码阶段，单纯减少采样步数也不能消除这里的峰值。

## 官方接口依据

以下链接对应本次读取的 PyTorch 2.11 API 或实际构建提交，不将最新文档状态混入本次结论：

- [PyTorch 2.11：scaled_dot_product_attention](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)：SDPA 的 mask、后端选择与数值差异说明。
- [PyTorch 2.11：sdpa_kernel](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.sdpa_kernel.html)：探针通过该上下文每次只启用一个后端，避免无意回退到 MATH。
- [实际 Torch 构建的 SDPA 选择代码](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)：源码中的 ROCm 分支对应原日志中生成后的 `hip/sdp_utils.cpp` 路径，包含 experimental gate 与后端检查。
- [PyTorch 2.11：HIP semantics](https://docs.pytorch.org/docs/2.11/notes/hip.html)：ROCm 复用 `torch.cuda` 接口；异常出现 “CUDA” 不代表调用了 NVIDIA GPU。

## 探针设计与复现

探针不下载依赖、不加载模型，只分配随机 BF16 Q/K/V，随机种子为 0，`dropout_p=0`，仅 forward。Q/K/V 使用模型常见的非连续转置布局。分别测试无 mask、`is_causal`、广播 key mask、Qwen dense causal/text-key mask、diffusion dense all-true mask。dense mask 保留真实行跨度，不把长度 257 或 11433 补齐到 8 的倍数。

长度不超过 512 时，以 MATH 输出作正确性基线，使用 `rtol=0.02, atol=0.02` 比较并检查有限值。长度超过 512 时，禁止运行 MATH 和大序列正确性基线，以避免重复制造二次方内存分配。每个后端每种 mask 只执行一次，计时包含首次调用初始化/编译开销，不属于稳态吞吐基准。GPU 同步计时不含输入和 mask 的构造、CPU 正确性比较。

从项目根目录使用已有环境执行；下列命令写入新目录，不覆盖本次原始证据。应在没有其他测量运行时串行执行，每个探针最多 120 秒；报错或超时后保留 JSON/日志，先检查原因再继续。

```bash
ATTENTION_PY=artifacts/init/300/20261015T234635Z-ideogram4-s46lqg0u/venv/bin/python

timeout 120s env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 \
  "$ATTENTION_PY" -I experiments/media-rocm-attention/probe.py \
  --sequence-length 257 --heads 32 --head-dim 128 --text-tokens 129 \
  --output artifacts/research/media-rocm-attention-reproduce/qwen-small-disabled.json

timeout 120s env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  "$ATTENTION_PY" -I experiments/media-rocm-attention/probe.py \
  --sequence-length 257 --heads 32 --head-dim 128 --text-tokens 129 \
  --output artifacts/research/media-rocm-attention-reproduce/qwen-small-enabled.json

timeout 120s env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  "$ATTENTION_PY" -I experiments/media-rocm-attention/probe.py \
  --sequence-length 257 --heads 18 --head-dim 256 --text-tokens 129 \
  --output artifacts/research/media-rocm-attention-reproduce/diffusion-small-enabled.json

timeout 120s env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  "$ATTENTION_PY" -I experiments/media-rocm-attention/probe.py \
  --sequence-length 11433 --heads 32 --head-dim 128 --text-tokens 849 \
  --output artifacts/research/media-rocm-attention-reproduce/qwen-2k-enabled.json

timeout 120s env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  "$ATTENTION_PY" -I experiments/media-rocm-attention/probe.py \
  --sequence-length 11433 --heads 18 --head-dim 256 --text-tokens 849 \
  --output artifacts/research/media-rocm-attention-reproduce/diffusion-2k-enabled.json
```

每个 JSON 记录实际 Python、Torch 导入路径、构建版本、GPU 名称、输入 shape/stride、环境变量、mask、警告、异常、计时和内存。顶层 `complete` 表示探针遍历完成，不表示所有后端可用；必须检查各项 `results[].status`。

## 已执行的算子结果

`qwen-small-disabled.json`（env=0，长度 257）中五种 mask 的 MATH 均成功，FLASH_ATTENTION 与 EFFICIENT_ATTENTION 均失败。`qwen-small-enabled.json`（env=1）中 MATH 与 EFFICIENT_ATTENTION 五项均成功；FLASH_ATTENTION 仅无 mask 与 `is_causal` 成功，对非空显式 mask 仍报告不支持。不能把“启用优化 attention”概括成“所有调用都使用 FlashAttention”。

开启后的短序列关键正确性结果如下，均相对于同次 MATH 基线，均为 finite 且 `allclose=true`：

| shape（B,H,L,D）与 mask | 后端 | 最大绝对误差 | 平均绝对误差 |
|---|---|---:|---:|
| (1,32,257,128)，Qwen dense causal/text-key | EFFICIENT_ATTENTION | 0.015625 | 0.000175019741 |
| (1,32,257,128)，diffusion dense all-true | EFFICIENT_ATTENTION | 0.00390625 | 0.000108734603 |
| (1,18,257,256)，Qwen dense causal/text-key | EFFICIENT_ATTENTION | 0.015625 | 0.000179632858 |
| (1,18,257,256)，diffusion dense all-true | EFFICIENT_ATTENTION | 0.00390625 | 0.000109862340 |

原始证据：[Qwen 短序列 env=0](../../artifacts/research/ideogram4-oom-20261016/qwen-small-disabled.json)、[Qwen 短序列 env=1](../../artifacts/research/ideogram4-oom-20261016/qwen-small-enabled.json)、[diffusion 短序列 env=1](../../artifacts/research/ideogram4-oom-20261016/diffusion-small-enabled.json)。

长度 11433 的 `EFFICIENT_ATTENTION` 也完成了两种 dense mask。与模型结构对应的结果为：

| shape（B,H,L,D）与 mask | 同步耗时 | 调用内 allocated 增量峰值 |
|---|---:|---:|
| (1,32,11433,128)，Qwen dense causal/text-key | 2.004060 秒 | 524288000 bytes = 0.48828125 GiB |
| (1,18,11433,256)，diffusion dense all-true | 0.348019 秒 | 524288000 bytes = 0.48828125 GiB |

原始证据：[Qwen 长序列](../../artifacts/research/ideogram4-oom-20261016/qwen-2k-enabled.json)、[diffusion 长序列](../../artifacts/research/ideogram4-oom-20261016/diffusion-2k-enabled.json)。**约 0.4883 GiB 是单个算子相对调用前 allocated 的增量峰值，不能解释为模型总内存，也不包括调用前已存在的 Q/K/V、mask、模型权重。** 长序列未执行 MATH 数值对照；短序列通过不能证明完整模型输出与原路径逐位一致。

## 早期完整 2K GPU 解码验证记录

完整生成 run 为 [`20261016T014220Z-ideogram4-7nuc_7i3`](../../artifacts/hello/300/20261016T014220Z-ideogram4-7nuc_7i3/manifest.json)，原始控制台日志为 [`full-2k-verification.log`](../../artifacts/research/ideogram4-oom-20261016/full-2k-verification.log)。目标保持用户指定 caption、1344×2016、12-step sampler、seed 0；启用实验 ROCm attention 后使用原 prepared 环境，不更换模型或依赖。

**实际结果：达到 1200 秒预算后超时，退出码 124，未生成最终图片。** 本次按预定停止规则结束，未复现原来的 OOM；`gpu_inference_validated=false`，不能视为完整 2K 验证成功。模型加载 133.252 秒，hello 总耗时 1202.135 秒。生成阶段未完成，耗时/吞吐和该阶段 PyTorch allocator 最终峰值均不可用；被终止的 worker 没有完成阶段收尾，因此 runtime JSON 保留最后的 running 状态，最终状态以 manifest/performance 的 `timed_out` 为准。

240 次资源采样中，整卡 GTT 峰值 76.903 GiB，原失败运行为 94.567 GiB；生成阶段 GPU 利用率平均 99.816%，进程 CPU 平均 114.380%（100%=一个逻辑核心），该阶段系统可用内存最低约 30.225 GiB。上述是超时前的观测，不代表完整 pipeline 最终峰值；GPU 温度最高 92℃，SoC 功耗最高 142.044 W。进程结束后只读检查显示 GPU busy 为0，GTT 回落到约29.85 GiB。详见 [performance.json](../../artifacts/hello/300/20261016T014220Z-ideogram4-7nuc_7i3/performance.json) 和 [前后对照 JSON](../../artifacts/research/ideogram4-oom-20261016/comparison.json)。没有最终图像，故没有图像 SHA256 或视觉质量结论。

后续命令中 Ideogram4 hello 默认预算已从900提高到3600秒，IndexTTS2仍为900秒，显式 `--timeout` 优先；这项默认值调整在本次冻结验证之外完成，不改变本次1200秒停止规则，也不证明3600秒必定足够。全部245项工具测试通过。

## 已理解、未解决与下一步

**已理解：** 原失败是 GPU OOM，发生在长序列 Qwen 文本编码；96 GiB GPU 窗口已接近用满。当前 Torch 构建在 gfx1151 上默认不开放实验优化 attention，显式开启后，探针中的 Memory Efficient attention 支持实际长度与 dense mask，短序列结果满足已设数值容差。

**未解决：** 早期 GPU 解码路径发生驱动重置；之后 1344×2016、GPU 编码/采样加 CPU 解码已有一次成功，仍不能证明 GPU 故障根因已排除。早期粗粒度记录无法分离各阶段，新增子阶段可观察解码，但文本编码未独立同步、各去噪步尚无精确 GPU 计时。2K 输出主标题存在乱码，需要后期排版；所有尺寸、mask、软件版本的稳定性及连续生成的稳态耗时仍未验证。

**下一步（两种尺寸 CPU 解码成功后更新）：** 按 [GPU 卡死调查](gpu-freeze.md) 的既有成功配置，在无其他 KFD 计算任务的条件下做有预算的串行重复验证，逐次核查图像和同期内核日志；保留 2K 启动保护，不自动恢复旧 GPU 解码路径。进一步优化前补齐 Qwen 编码及每个去噪步的同步计时。可研究只编码真实文本、跳过最终会归零的图像占位 token，但需先做数值等价对照，不能直接更改模型计算。后续测试保持独立 run、固定代码和显式时间预算。
