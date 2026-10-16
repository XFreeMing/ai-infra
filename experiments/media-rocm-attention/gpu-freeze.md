# AI 300：三次显示器卡死／绿屏与 GPU/CPU 混合恢复验证

日期：2026-10-16，时区 Asia/Shanghai。实验 ID：`media-rocm-attention`，调查 ID：`ideogram4-gpu-freeze-20261016`。这是读取既有运行产物和系统日志的事后调查；本轮没有运行 GPU 推理、安装驱动、停止服务或重启机器。

## 最新实际观察：1344×2016 GPU 采样 + CPU 解码成功

用户于 11:54:10–12:04:05 执行 `uv run 300 hello Ideogram4 --width 1344 --height 2016 --decode-device cpu --timeout 3600 --experimental-gpu`。run `20261016T035410Z-ideogram4-rjbq82iy` 的 manifest 为 `passed`、`gpu_inference_validated=true`，性能记录完整，1344×2016 PNG 的实际 SHA256 与输出记录一致。这是原生目标尺寸的一次完整成功，文本编码和 12 步采样使用 GPU，最后 VAE 解码使用 CPU float32；启动前 KFD 快照没有其他计算进程。

| 指标 | 本次观测 |
| --- | ---: |
| hello 总耗时 / 模型加载 | 595.098 s / 107.591 s |
| 生成（编码、12 步采样、解码） | 482.235 s |
| 解码前 GPU 完成同步 / CPU VAE 解码（含搬运） | 37.270 s / 18.365 s |
| 生成阶段 GPU 利用率 min / max / mean | 0% / 100% / 96.05% |
| 全程整卡 GTT 峰值 / 系统可用内存最低值 | 37.868 GiB / 63.515 GiB |
| Swap 峰值 / GPU edge 温度峰值 | 0 / 91℃ |

11:52–12:06 的内核、Xorg 和系统日志未发现 GPU reset/hang、OOM 或 Xorg 崩溃；有一条 `amdgpu svm_range_deferred_list_work` 占用超过 10000 us 的警告，不等同于 GPU 重置。运行沿用下文固定 prepared 环境、源码和 ModelScope 模型版本，父提交为 `da83123838ab4726f718eb26751f37bf427359fc`，工作区有未提交修改。共 296 次资源采样，间隔 2 秒；GTT、RSS 和 PyTorch allocator 不能相加。

实际图片呈现了绿白与金箔肌理、抽象笔触，但主标题没有正确写出 `Don’t wait for inspiration`，出现乱码及中文描述混入，上部还出现“极小体”字样。底部品牌词和小句可读；可作为视觉草稿或背景，文字需人工排版校对，文件校验通过不等于设计稿验收通过。

证据见 [manifest](../../artifacts/hello/300/20261016T035410Z-ideogram4-rjbq82iy/manifest.json)、[performance.json](../../artifacts/hello/300/20261016T035410Z-ideogram4-rjbq82iy/performance.json)、[独立复核与日志索引](../../artifacts/reports/ai300-media-20261016/latest-2k-audit/summary.json)；面向产品人员的[实测报告](../../artifacts/reports/ai300-media-20261016/deliverables/AI300生图与配音实测报告.html)附输入、成品和各分辨率性能对比。成功只覆盖这一混合配置的一次运行，尚不证明连续稳定或旧 GPU 解码故障已修复，2K/WebUI 启动保护保留。

## 此前实际观察：384×576 GPU 采样 + CPU 解码成功

用户于 11:14:24–11:17:19 执行以下命令，run `20261016T031424Z-ideogram4-w3t23g3p` 的生成和输出检查均返回 0，manifest 为 `passed`、`gpu_inference_validated=true`，性能记录为 `complete`。这次是真实训练权重的 GPU→CPU 完整链路，区别于下文先前的 CPU 小型对照。

```sh
uv run 300 hello Ideogram4 --width 384 --height 576 --decode-device cpu --timeout 300
```

| 指标 | 本次观测 |
| --- | ---: |
| hello 总耗时 | 174.430 s |
| 模型加载 | 128.857 s |
| 生成（编码、12 步采样、解码） | 40.102 s |
| 解码前 GPU 完成同步 | 3.042 s |
| CPU float32 VAE 解码（含搬运） | 1.146 s |
| 生成阶段 GPU 利用率 min / max / mean | 12% / 100% / 94.65% |
| 全程整卡 GTT 峰值 | 29.993 / 96 GiB |
| 生成阶段 PyTorch allocated / reserved 峰值 | 27.287 / 29.172 GiB |
| 全程系统可用内存最低值 / Swap 峰值 | 66.132 GiB / 0 |
| GPU edge 温度采样峰值 | 83℃ |

GPU 完成同步成功，随后 `decoder.effective_device=cpu`，VAE 输入来源为 `cuda`（PyTorch ROCm 同样使用这个设备名）。启动前 KFD 快照没有其他计算进程。共 88 次资源采样，JSONL 与 JSON samples 一致，重算全程及分阶段 108 组 min/max/mean 与保存值一致，测量错误列表为空。采样间隔 2 秒，可能遗漏瞬时峰值；统一内存上的 GTT、RSS 和 PyTorch allocator 不能相加。文本编码子阶段的 0.278 s 没有独立 GPU 同步，不能作为完整编码 GPU 耗时。

输出为 384×576 非纯色 PNG，SHA256 `ac32efd085e6b5557224193495597318ea8fc24b454716badcd2ecee57305739`，已重新核对文件哈希。目视有蓝绿金色肌理，但主英文标题不完整，出现描述性中文字样，部分文字被裁切；自动输出检查只证明文件及基本像素有效，不代表海报文字质量合格。

同期日志复核覆盖当前 boot `d011820652f541909ba1518c17d8660f` 的 11:13–11:19，查询退出码 0、无权限警告，共 753 条系统记录、其中 1 条内核记录。**未发现 GPU reset/hang、ring/MES timeout、Xorg 崩溃或 core dump。** 唯一内核记录为 11:16:37 的 `amdgpu svm_range_deferred_list_work` CPU hog（超过 10000 us、累计 4 次）；不能称为零内核警告，但不等同于 GPU 重置。查询与日志 SHA256 见 [journal-capture.json](../../artifacts/research/ideogram4-gpu-freeze-20261016/small-cpu-decode-success/journal-capture.json)。

运行沿用 prepared 环境 `20261015T234635Z-ideogram4-s46lqg0u` 和下文固定上游版本；父提交为 `da83123838ab4726f718eb26751f37bf427359fc`，用户运行时工作区有未提交修改。模型为 ModelScope `ideogram-ai/ideogram-4-fp8` revision `0697355979847a8e4db198738c1d5ed4453ffa24`；完整提示词、环境、冻结运行文件身份见 [manifest](../../artifacts/hello/300/20261016T031424Z-ideogram4-w3t23g3p/manifest.json)，细分计时见 [performance.json](../../artifacts/hello/300/20261016T031424Z-ideogram4-w3t23g3p/performance.json)，本次复核及输入 SHA256 见 [run-review.json](../../artifacts/research/ideogram4-gpu-freeze-20261016/small-cpu-decode-success/run-review.json)。

**结论范围：** 这套小尺寸配置已有一次成功证据。与第三次失败相比，分辨率和解码设备同时改变，无法单独证明 VAE/MIOpen 是根因，该次小图单独不能证明原生 2K 或连续运行稳定；后续 2K 混合路径的一次成功见本文开头。默认 1344×2016 和提示词保持原设置，2K/WebUI 保护继续保留。

## 第三次故障：停止 Qwen 后仍复现

**新增实际观察：** AIMA 在 10:33:05 完成停止，后台 llama-server 已退出；第三次 `20261016T023416Z-ideogram4-vboyafey` 于 10:34:16 启动，10:36 的只读快照仅有生图 KFD 进程 PID 21631。10:44:20 出现 `comp_1.1.0 timeout`，10:44:22 继续 `gfx_0.0.0 timeout`，随后 MES 无响应、整卡 MODE2 reset，10:44:27 Xorg 和生图均 SIGABRT，10:52:27 机器重新启动。该 run 在保护代码加入前已经启动，运行中的旧进程不会被新入口保护拦截。

这次整卡 GTT 最大值 **47.22 / 96 GiB**，系统可用内存最少 **63.99 GiB**，Swap 全程 **0**，没有内核 OOM kill/oomd 记录。故障前最后采样温度 62℃，全程峰值 93℃；不能由单个峰值推定过热。**单纯停止后台 Qwen 不足以解决，容量耗尽也不符合这次已记录的资源情况。** 第三次说明当前生图路径在没有该后台计算服务时仍能触发故障，不再把并发服务当作必需触发条件。

生成日志再次停在 `MIOpen EvaluateInvokers` / `miopen_convolution` / `unspecified launch failure`。固定源代码 `pipeline_ideogram4.py` 的 `Ideogram4Pipeline.__call__` 在采样结束后调用 `_decode`，后者调用 `autoencoder.decoder`；`autoencoder.py` 包含该路径的 Conv2d。Qwen 编码只运行 language_model，不运行其视觉卷积分支。因此 **VAE 解码是优先隔离的候选阶段**；由于 HIP 异步报错且没有细分阶段日志，不能直接宣称卷积就是首次挂死点。

证据：[第三次摘要及资源统计](../../artifacts/research/ideogram4-gpu-freeze-20261016/third-crash/summary.json)、[内核时间线](../../artifacts/research/ideogram4-gpu-freeze-20261016/third-crash/kernel-window.log)、[原始生图日志](../../artifacts/hello/300/20261016T023416Z-ideogram4-vboyafey/00-generate.log)。AIMA 停止、Xorg 堆栈、输入文件 SHA256 同存第三次目录。

## CPU VAE 旁路：实现与先前的 CPU 对照

仅 `uv run 300 hello Ideogram4` 新增 `--decode-device cpu|gpu`，默认 `cpu`。GPU 继续执行文本编码和采样；进入解码后先同步此前 GPU 工作，再把 latent、归一化参数、autoencoder 移到 CPU float32，使用浅拷贝 pipeline 调用原版 [`Ideogram4Pipeline._decode`](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/src/ideogram4/pipeline_ideogram4.py#L619)。原 pipeline 的 GPU device/dtype/transformer 不变，冻结源码与环境也不变，无需重新 init。该旁路绕开 VAE 的 GPU/MIOpen 卷积，但不证明其余 GPU 工作安全。

如果 GPU 完成同步已经失败，将记录在 `decode_gpu_sync`，不会标成 CPU 解码失败或继续回退。新增 `text_encoding`、`decode_gpu_sync`、`vae_decode` 子阶段，控制台显示开始/结束及解码设备，`runtime-performance.json` 保存 `decoder`、`subphases` 与 `active_subphase`。子阶段耗时是主机调用时间，不重置外层 allocator 峰值；只有显式同步位置证明 GPU 完成，文本编码返回不等于所有异步 GPU 工作完成。

`uv run 300 Ideogram4` 常规/WebUI 仍是原运行器，不使用这个 hello 专用旁路，默认继续受保护。CPU 解码也**不会自动放开 2K 保护**。`--decode-device gpu` 保留旧路径用于隔离对照，不作为当前重试建议。IndexTTS2 不使用该参数。

**实现时的验证范围：** 273 项工具测试通过；[CPU-only 检查脚本](check_cpu_decode.py) 使用实际 PyTorch CPU、固定上游原版 `_decode`、随机小型 Decoder（16×16 输出），对比直接 CPU 解码与适配后解码，像素完全一致，GPU runtime 从未初始化。[结果 JSON](../../artifacts/research/ideogram4-gpu-freeze-20261016/third-crash/cpu-decode-check.json)记录源码提交、实际导入路径和 SHA256。这项 CPU 对照不是训练权重的画质验证、GPU→CPU 实测或完整 2K 推理成功；之后用户执行的小尺寸及 2K 全链路结果见本文开头。

CPU 路径仍消耗系统内存，且 float32 可能较慢、与原 BF16 路径产生数值差异。2K VAE 的 attention 序列长 42336，若使用 dense FP32 score，单份矩阵理论约 6.68 GiB；不能把“移到 CPU”当作零内存成本。本轮未运行真实 GPU 生图或更改驱动。

只查看小尺寸旁路计划：

```sh
uv run 300 hello Ideogram4 --width 384 --height 576 --decode-device cpu --timeout 300 --dry-run
```

## 已确认的故障链

**实际观察：两次均出现 AMDGPU 图形队列超时、重置，随后 Xorg 收到 SIGABRT。** 显示桌面与 AI 计算共用 Radeon 8060S，驱动重置影响整个显示会话，因此会表现为卡死／绿屏。第一次重置后 journal 仍继续记录，不能将显示失效等同于整个内核立即停止运行。

| 北京时间 | 证据 |
| --- | --- |
| 09:42:22 | `20261016T014220Z-ideogram4-7nuc_7i3` 的生成进程开始，PID 2931307 |
| 09:54:33.498 | `ring gfx_0.0.0 timeout`，记录关联 Xorg 提交；随后 ring reset |
| 09:54:33.515 | Xorg SIGABRT；桌面应用随后报无法连接 X display |
| 10:02:22 | 外层 hello 达到 1200 秒预算，记录 `timed_out`；**此时不能解释为模型一直在正常计算** |
| 10:07:49 | 第一次重启后的 boot 开始 |
| 10:11:24 | `20261016T021122Z-ideogram4-zuvsveid` 的生成进程开始，PID 21523 |
| 10:23:03 | 再次 `gfx_0.0.0 timeout`，记录关联 mihomo-party 图形提交 |
| 10:23:05 | MES 不响应 RESET，ring reset 失败；无法驱逐 PID 14780、21523 的 KFD 队列 |
| 10:23:07 | MODE2 全 GPU 重置；MIOpen / HIP 报 `unspecified launch failure`，hello 子进程 SIGABRT（退出码 -6），Xorg 也 SIGABRT |
| 10:25:39 | 第二次重启后的 boot 开始 |

Xorg、mihomo-party 是超时记录中的图形提交者，**这不证明它们引发故障**。GPU 错误可以异步上报，第二次的 `miopen_convolution` 堆栈说明卷积执行／错误处理受影响，不能单凭此栈证明 VAE、MIOpen 或 attention 中某个 kernel 是最初根因。

原始证据：[内核摘要](../../artifacts/research/ideogram4-gpu-freeze-20261016/kernel-crash-summary.json)、[两次运行关联及资源统计](../../artifacts/research/ideogram4-gpu-freeze-20261016/run-correlation.json)、[日志 SHA256](../../artifacts/research/ideogram4-gpu-freeze-20261016/journal-sha256.json)。目录中同时保留两次 boot 的内核日志、相关系统时段、Xorg 堆栈和 AIMA 部署记录。机器本地证据未提交 Git，公开报告需提供可访问副本。

## 与之前 OOM 的区别

此前 `20261016T013030Z-ideogram4-cwyx0r5p` 确实是 Qwen 编码时 OOM。此次开启 memory-efficient attention 后的两次显示故障，证据变成 GPU 队列超时和重置。

| 采样指标 | 第一次显示故障 | 第二次显示故障 |
| --- | ---: | ---: |
| 整卡 GTT 最大值 | 76.90 / 96 GiB | 76.26 / 96 GiB |
| 整机可用内存最小值 | 25.12 GiB | 30.83 GiB |
| Swap 最大用量 | 约 8.00 GiB | 约 0.15 GiB |
| GPU 温度采样最大值 | 92℃ | 90℃ |
| 最终图片数 | 0 | 0 |

两个 boot 的持久化内核日志没有找到 OOM killer、MCE、critical thermal 或 GPU VM/page fault 记录。第二次 GPU reset 后桌面组件出现的 Skia OOM，不能倒推成故障前系统内存耗尽。温度读数也不足以证明过热或排除供电／平台问题。没有保存下来的 devcoredump/pstore 可以进一步定位最先挂起的 kernel。

**纠正此前结论：** “20 分钟内没有 OOM”只描述 Python 异常；该窗口内实际已经发生 GPU reset 和 Xorg 崩溃。99.82% GPU busy 只能说明设备计数器读值，不能证明模型持续推进。增加超时不能修复驱动挂起，Python 超时也不能保证及时打断已经卡住的 GPU。

## 并发负载与兼容性

**实际观察：** 第二次故障的 PID 14780 属于 AIMA 管理的 HIP `llama-server`，模型为 `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`，`ctx-size=262144`、`n-gpu-layers=99`、`batch-size=4096`、`ubatch-size=2048`。下一次启动的 stale PID 部署记录完成了 PID 与模型的关联。当前重启后也发现该服务自动启动并登记为 KFD 计算进程；用户选择自行手动停止它。本次排查没有终止它。

**推论：** 生图、后台 LLM 和桌面共用 GPU，增加了计算调度、内存和显示共存的复杂度。隔离后台模型是必要的对照条件，但尚未证明停止它就一定解决绿屏；约 30 GiB 背景 GTT 也不能无测量全部归给该模型。

调查期间用户自行停止了后台服务。10:36 的只读快照中，KFD 列表只剩新生图进程 PID 21631，对应用户 10:34 启动的 `20261016T023416Z-ideogram4-vboyafey`，尺寸仍为 1344×2016；这不是本轮代理发起的验证。已提示用户先 Ctrl+C，新的入口保护不会干预已启动进程。该运行当时没有最终结果，不能纳入“已恢复”结论。[主机只读快照](../../artifacts/research/ideogram4-gpu-freeze-20261016/host-readonly-snapshot.json)

本机运行身份沿用 [attention 实验](README.md)：Ideogram 源码 `990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`、Torch `2.11.0+rocm7.2`、FP8 权重/BF16 计算、AOTriton 0.11.2、Ubuntu 24.04.4、`7.0.0-38-generic`。当前内核参数还设置 `ttm.pages_limit=25165824 amdgpu.gttsize=98304`；本轮没有更改它们。

**官方说明与范围：** [ROCm 7.2 Ryzen 矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)列有 AI Max+ 395/gfx1151，但验证组合是 Ubuntu 24.04.3、PyTorch 2.9、Python 3.12，精度只正式验证 FP16。当前完整组合超出了这张固定矩阵的验证范围，不等于已证明哪一项不兼容。[7.2 安装说明](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installryz/native_linux/install-pytorch.html)写的是 6.14-1018 OEM 或更新，不能仅凭版本数字断言所有 7.0 内核不支持。该说明本身也建议设置 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`，因此变量名不是故障因果证据；直接关闭它再跑 2K 会重新面临此前的 attention 内存风险。

后续整套环境对照可参考 [ROCm 7.14.1 发布说明](https://rocm.docs.amd.com/en/docs-7.14.1/about/release-notes.html)列出的 Ryzen Ubuntu 24.04.4 + OEM 6.17、PyTorch 2.12 组合。应保留当前启动项和独立 Python 环境，不覆盖已冻结环境；升级本身不是本机稳定性验证。本轮没有升级、降级或修改系统启动设置。

## 本轮防再触发措施

启动入口保留 1344×2016 的用户配置和原提示词，但普通调用不再直接执行这条已两次导致显示故障的路径。Ideogram4 单次超过 512×512 像素面积、WebUI（尺寸可动态扩大），以及检测到其他 KFD 计算进程的运行，会在 GPU worker 启动前拒绝并说明原因。阈值是临时排查策略，**不是已验证的安全分辨率上限**。`--dry-run` 仍可查看完整计划；明确的 `--experimental-gpu` 只用于受控实验重试，写入运行记录，不表示修好驱动。

`hello` 可用 `--width` / `--height` 指定小尺寸。宽高仍必须为 16 的倍数且在 256–2048；例如 384×576 保持 2:3，比 2K 少得多的图像 tokens。IndexTTS2 不受图像尺寸保护规则影响。KFD 检查是一次性只读快照，不会停止服务；未知读取结果不视为空闲，也无法阻止其他服务在检查后抢占 GPU。

工具验证：264 项 unittest 通过，`git diff --check` 通过，默认 2K `--dry-run` 正确记录 `gpu_guard.status=blocked`；`doctor` 也已执行，其沙箱中 `/dev/kfd` 不可见，不能将该结果解释为主机权限回退或 GPU 稳定性验证。测试使用 mock，不发起真实 GPU 推理；日志保存在调查产物目录。

以下保留保护加入时的只读计划命令；后续用户执行的两次成功恢复推理见本文开头：

```sh
uv run 300 hello Ideogram4 --dry-run
uv run 300 hello Ideogram4 --width 384 --height 576 --timeout 300 --dry-run
```

## 已理解、未解决与下一步

已理解：显示失效有完整的 GPU timeout/reset → Xorg abort 证据；第二次同时有后台 Qwen 和生图两个 KFD 任务，停止 Qwen 后第三次仍复现。原来的 OOM 与驱动故障必须分别处理。384×576 和 1344×2016、GPU 编码/采样加 CPU VAE 解码现各有一次完整成功记录。

未解决：最初挂起的 kernel、驱动/固件问题与并发影响尚未分离；CPU 解码的 2K 成功仍不足以称为“已修复绿屏”，旧 GPU 解码路径及连续运行稳定性未验证。不能把单算子通过或工具单元测试当作 GPU 稳定性证明。

下一步（两种尺寸成功后更新）：先在已成功的 384×576、1344×2016、CPU 解码和无其他 KFD 计算任务条件下，做有预算的串行重复验证并检查同期内核日志，同时人工复核文字质量。保留原 2K 保护，不自动恢复旧 GPU 解码路径。后续实验仍应使用独立快照并保留逐次输出、性能和内核证据；若再次出现 GPU 重置，应先做 AMD 验证组合的独立环境对照再决定驱动更换。本次仅复核用户既有成功产物，没有自动发起新一轮推理。
