# AI 300 生图和配音命令

最后更新：2026-10-16。Ideogram4 和 IndexTTS2 的运行环境、模型下载、启动分为三个独立入口，另有 `hello` 快速推理验证。命令不区分大小写，`Indextts2` 与 `IndexTTS2` 都接受。

## 运行环境初始化

```sh
uv run 300 init Ideogram4
uv run 300 init Indextts2
# 也可一次准备两个，按顺序执行
uv run 300 init Ideogram4 Indextts2
```

`init` 只准备固定源码、独立 Python venv、依赖、运行适配和 ROCm/import 小探针，不读取、下载或加载生图/TTS 模型与辅助权重。模型不存在也不影响环境初始化。IndexTTS2 的 NLTK 词典/标注运行资源在此阶段准备，以免启动时隐式联网获取语言资源。

NLTK 3.10.3 的动态下载器默认拒绝代理请求。IndexTTS2 init 改为从官方 `nltk/nltk_data` 的固定 commit [`550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`](https://github.com/nltk/nltk_data/tree/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a) 获取五项语言资源，URL、大小与官方 SHA256 固定在 [nltk_resources.json](../lab_tools/nltk_resources.json)。下载沿用 HTTPS/代理配置，拒绝重定向，校验后安全解压；不修改 NLTK 的安全开关。控制台显示字节比例、速度和解压文件计数，已校验 ZIP 缓存在 `.cache/nltk/`，离线初始化也会检查并使用这份缓存。运行目录保留 ZIP 和解压文件，以满足 g2p-en 的资源路径检查。

ROCm/import 检查还会限制 NLTK 只读取本次初始化的资源，并在禁止隐式下载时验证英文/中英混合文本音节计数、词性标注和分句，避免导入成功但推理时才发现词典缺失。2026-10-15 已在 AI 300 完整执行新初始化，`artifacts/init/300/20261015T090934Z-indextts2-avevuadc/manifest.json` 状态为 `prepared`：torch 2.11.0+rocm7.2、HIP 7.2.26015、Radeon 8060S/gfx1151；GPU 小运算与语言检查通过。本次没有下载或加载 IndexTTS2 主模型，不代表完整语音合成验证。

两个应用各有新的 `artifacts/init/300/<run-id>/`，不会修改已有环境。Ideogram4 使用 Python 3.12、transformers 5.8.0、Gradio 5.49.1；IndexTTS2 使用 Python 3.11、transformers 4.52.1、Gradio 5.45.0。两套都固定 torch 2.11.0+rocm7.2，IndexTTS2 同时固定匹配的 torchaudio，并使用 soundfile 读写 WAV。IndexTTS2 上游 2.8.* 约束的调整、本地模型 IO 适配、源码 commit、补丁 SHA256、实际 import path、GPU 运算、freeze 与关键 Torch 动态库身份均保存到准备清单。没有模型推理结果时 `gpu_inference_validated` 保持 false。

Ideogram4 的固定 FP8 模型由 Transformers 5.8.0 保存。旧 4.57.6 环境不能正确读取列表式 `extra_special_tokens`、`rope_parameters`，且文本编码的 causal-mask API 不匹配；缺少 protobuf 的异常会掩盖首个错误。修复采用匹配版本，并使用支持 Hugging Face Hub 1.x 的 Gradio；旧环境需重新执行 `uv run 300 init Ideogram4`。依据：[Transformers 5.8.0 配置](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py)、[Gradio 5.49.1 依赖](https://github.com/gradio-app/gradio/blob/gradio%405.49.1/requirements.txt)。

普通 Python 依赖和 ModelScope SDK 默认使用阿里云：`https://mirrors.aliyun.com/pypi/simple`。两个入口统一读取根目录 [pyproject.toml](../pyproject.toml) 的 default index；修改此处即可换源。安装命令显式传递 `--default-index`，构建子进程通过 `UV_DEFAULT_INDEX` / `PIP_INDEX_URL` 沿用同一配置，准备/下载清单记录 `package_default_index`。此配置只控制 Python 包；Python 解释器和 Git 源码各用原有下载渠道，NLTK 语言资源使用下述固定官方文件，模型权重由 ModelScope 下载。

2026-10-15 的一次 IndexTTS2 初始化在清华源下载 `werkzeug==3.1.9` 时网络超时；依赖解析已成功，初始化总时限没有触发。现已将项目、初始化和两个独立下载脚本的普通 Python 包默认源统一为阿里云，不自动切回清华。同一 Werkzeug wheel 已从阿里云完整下载并校验，与原源文件 SHA256 一致。此前失败目录保留原始源和日志，重新运行 init 才应用新配置。

300 生图/配音初始化默认使用 `UV_CONCURRENT_DOWNLOADS=4`、`UV_HTTP_CONNECT_TIMEOUT=10`、`UV_HTTP_TIMEOUT=30`、`UV_HTTP_RETRIES=2`，减少同时下载的连接数。可在命令前设置同名环境变量覆盖；实际值在控制台与 manifest 中记录。HTTP 超时针对单次连接/读取，`--build-timeout 0` 只取消整个初始化的总时限，不能关闭网络读取超时。保留 uv 的正常缓存复用和原生下载进度；未完成下载能否复用由 uv 决定。[uv 网络参数说明](https://docs.astral.sh/uv/reference/environment/#uv_http_timeout)。

300 的 torch、torchaudio、triton-rocm 改用南京大学镜像 `https://mirror.nju.edu.cn/pytorch/whl/` 的固定 wheel 直链，按应用选择 cp312 / cp311。入口不再传 `--torch-backend rocm7.2`，因为该选项会绕过镜像，转用官方海外地址。torch / torchaudio 直链携带官方索引中的 SHA256，uv 安装时校验；Triton 上游索引未公布完整 wheel SHA256，按固定 3.6.0 文件安装。实际 URL 在初始化控制台、命令计划和 manifest 的 `rocm_wheel_sources` 中可见。版本、哈希和映射在 [media_wheels.py](../lab_tools/media_wheels.py)。入口没有配置官方回退源，镜像不可用时报告失败。[南京大学 PyTorch 镜像说明](https://help.mirrorz.org/pytorch/)、[uv torch-backend 行为](https://docs.astral.sh/uv/reference/settings/#pip_torch-backend)。

2026-10-15 本机检查：NJU 的 torch / torchaudio cp311、cp312 索引 SHA256 与官方一致；所需五个 wheel 的小范围 HTTP 请求返回 206、ZIP 文件头及 NJU 最终地址，没有跳转海外。torch / Triton 各读取前 4 MiB，单次请求含连接开销约 0.17–0.23 MiB/s；此结果不能保证完整下载速度。较小的 cp311 torchaudio（337,743 字节）在该请求中完整返回，其 SHA256 与官方一致。两个应用均通过真实 uv 依赖解析，当时普通依赖使用 TUNA，大型 torch / Triton wheel 未完整下载，模型推理未验证。可复查本机证据：`artifacts/research/rocm-mirror/verification.json`、`ideogram4-direct-dry-run.log`、`indextts2-resolve.log`。其他镜像实测存在缺文件或跳回官方源的问题，不能仅凭镜像索引页面可访问就认定 wheel 在国内。

此前已中断的初始化目录保留原始命令和日志。重新执行 `uv run 300 init Ideogram4` 或 `uv run 300 init Indextts2` 会创建使用新镜像的环境；不会修改旧环境。大包安装可传 `--build-timeout 1800` 提高预算；当前测速下默认 900 秒可能不足，需要允许长时间下载时显式传 `--build-timeout 0`，仍可 Ctrl+C 中断。

```sh
uv run 300 init Ideogram4 --dry-run  # 只查看命令计划，不安装
uv run 300 init Indextts2 --check-only
uv run 300 init Ideogram4 --build-timeout 1800
uv run 300 init Indextts2 --build-timeout 0  # 明确允许不限总时间
```

默认总预算为 900 秒；安装 ROCm wheel 较慢时可提高。Linux 下下载/安装保留 uv 的终端进度，例如 `torch 1.2 GiB/5.8 GiB`、`Preparing packages (12/74)`、`Installing wheels [12/74]`（示意，实际总数以解析为准）。源码安装展开编译输出，源码快照、动态库校验和依赖冻结显示各自阶段；大文件校验显示百分比与速度。进度持续更新时不叠加「进行中」；连续至少 5 秒没有输出才补充耗时、最近输出和剩余预算。准备目录的 `progress.log` 每 5 秒保留后台摘要，各步骤 `.log` 保留完整输出。已有 init 进程仍运行旧代码，新启动才使用这些显示。

环境检查失败、超时或中断保留原目录和日志，重新运行会新建目录。当前这两个入口不支持 resume/update-sources，也不接受 `--model`；后续更新需要新环境和新证据。

## 用 ModelScope 下载模型

以下命令由用户自行执行。本轮没有执行任何模型文件下载。

```sh
uv run download Ideogram4
uv run download Indextts2
```

下载命令默认依据运行机器架构选择 300 或 700。也可显式指定机器；无需先执行 init：

```sh
uv run download Ideogram4 --machine 300
uv run download Indextts2 --machine 700
# 等价的机器入口
uv run 300 download Ideogram4
uv run 700 download Indextts2
```

| 机器 | 默认模型根目录 | Ideogram4 | IndexTTS2 |
| --- | --- | --- | --- |
| 300 | `/home/ops396/models` | `/home/ops396/models/Ideogram4` | `/home/ops396/models/Indextts2` |
| 700 | `/data/models` | `/data/models/Ideogram4` | `/data/models/Indextts2` |

700 在本轮提供下载目录选择，没有新增这两个应用的 700 CUDA 初始化/启动配方。IndexTTS2 辅助文件保存到主目录的 `auxiliary/`；不会在启动时偷偷从 Hugging Face 补下载。

```sh
uv run download Ideogram4 --dry-run   # 不安装 SDK、不联网、不写模型目录
uv run download Indextts2 --verify-only
uv run download Ideogram4 --models-dir /another/models
uv run download Indextts2 --local-dir /another/IndexTTS-2
uv run download Ideogram4 --timeout 7200
```

`--models-dir` 指定模型根目录；`--local-dir` 直接指定该模型目录。若使用自定义目录，启动时也传相同路径。下载总时间默认不限，`--timeout` 可设置预算。

下载工作进程使用独立的 uv script 环境和 ModelScope SDK 1.40.2。它按固定 ModelScope commit 和文件白名单下载，显式启用 SDK 逐文件进度条，显示字节、百分比与速度；每个仓库完成后显示整体已下载并校验的比例。随后逐文件显示 SHA256 校验进度，成功后写入模型目录 `download-identity.json`。已有缓存由 SDK 复用；失败/中断保留文件，不回退到其他下载站。

实际模型清单在 `lab_tools/media_models.json`，依据 ModelScope 的版本化 metadata 核对。主权重 SHA256 与先前读取的 Hugging Face LFS 身份相同；这里使用的是独立的 ModelScope revisions，不混用两个站的 commits：

| 用途 | ModelScope 仓库 | 固定 commit |
| --- | --- | --- |
| Ideogram4 FP8 | `ideogram-ai/ideogram-4-fp8` | `0697355979847a8e4db198738c1d5ed4453ffa24` |
| IndexTTS2 主模型 | `IndexTeam/IndexTTS-2` | `f165d7e5bd70d292969875d89d6e5d4fc8b328ca` |
| 语义编码器 | `facebook/w2v-bert-2.0` | `68cfcc9e237812bb3a1793199bec79f7cb85b96e` |
| semantic codec | `amphion/MaskGCT` | `f413985be524579156abf5fe3a6a0b4fcc03b117` |
| 说话人编码器 | `iic/speech_campplus_sv_zh-cn_16k-common` | `a045b2afcaa9c3049c98a9215a2bc274407ab237` |
| 声码器 | `nv-community/bigvgan_v2_22khz_80band_256x` | `28f63362719fc3b44bf7b9ca7b302c7fd0e34752` |

只下载推理所需文件，不下载额外精度、BigVGAN discriminator/optimizer 或 MaskGCT 其他模型。Ideogram4 约 27.54 GB；IndexTTS2 主模型加辅助文件约 8.87 GB，另外预留 SDK/临时文件空间。模型访问或协议要求仍由下载站处理；需要授权时命令会报错，不代用户接受协议。

## 快速验证 hello

环境 init 和模型 download 完成后，可以直接生成一个小样本验证实际推理：

```sh
uv run 300 hello Ideogram4
uv run 300 hello Indextts2
# 一次验证两个，串行运行
uv run 300 hello Ideogram4 Indextts2
```

Ideogram4 使用固定红船 JSON 提示词、256×256、12-step sampler 和 seed 0，生成一张 PNG，检查图片可解码、尺寸正确且非单一纯色。IndexTTS2 使用项目内置的上游演示参考人声，合成“你好，这是语音合成测试。”，检查 WAV 非空、数值有限、时长和音量合理。参考样本的固定来源、SHA256 和许可证见 [fixtures/README.md](../lab_tools/fixtures/README.md)；也可以使用自己的音频。

```sh
uv run 300 hello Indextts2 --reference /path/to/your.wav
uv run 300 hello Ideogram4 --timeout 1800
uv run 300 hello Ideogram4 --dry-run
uv run 300 hello Indextts2 --model-dir /another/IndexTTS-2
```

默认每个应用总预算 900 秒，包含准备和生成阶段；首次模型加载也计入预算。`hello` 复用已准备的独立环境和本地模型，不安装依赖、不下载模型，不启动 WebUI。缺少环境或模型会提示相应的 init/download 命令。`--dry-run` 只读检查并显示执行计划，不生成文件。

每次运行使用新的 `artifacts/hello/300/<run-id>/`，保存输入配置、参考音频快照、生成结果、验证指标和逐步骤日志；终端显示进度、超时与最终输出路径。成功退出码为 0，失败为 2，超时为 124，中断为 130。只有实际生成和输出校验都通过才记录 `status=passed` 与 `gpu_inference_validated=true`；此结果代表小样本推理链路可用，不代表全面画质或语音质量评测。

### 生成过程的资源与性能记录

两条命令默认每 2 秒采样并打印一行资源数据，显示阶段、已用时间、任务进程树 CPU、整机 CPU、进程 RSS、系统已用内存、GPU 利用率、VRAM/GTT、温度和 SoC 功耗。已有 prepared 环境直接可用，无需为监控重新 init；每次运行会保存测量代码快照，调用原有冻结 worker。

```sh
uv run 300 hello Ideogram4 --sample-interval 1
uv run 300 hello Indextts2 --sample-interval 1
```

`--sample-interval` 范围为 0.2–60 秒。进程 CPU 的 100% 表示占满一个逻辑核心，因此多线程可超过 100%；整机 CPU 为 0–100%。每个有效指标在结束时统计采样数量、最小值、最大值和算术平均值，JSON 还按运行时检查、模型加载和生成阶段分组。不可读的指标保留为 `null` 并附原因，不能解读为 0；监控错误不会替换原始推理退出码。

AI 300 的 RSS、VRAM、GTT 和 PyTorch allocated/reserved 含义不同、存在重叠，不能相加。sysfs GPU 利用率和内存读数覆盖整张卡，可能包含其他进程；APU 的 `power1_average/input` 包含 CPU，故标为 **SoC 功耗**。依据：[Linux AMDGPU 监控接口](https://docs.kernel.org/gpu/amdgpu/thermal.html)。采样最大值可能遗漏两次采样之间的短峰值；PyTorch allocator 的阶段峰值另行记录。

结束时除了资源汇总，还会打印以下性能数据：

| 项目 | 定义 |
| --- | --- |
| 总耗时 | 从 hello 开始到输出校验结束，包含等待锁和前置检查 |
| 生成进程耗时 | 子进程启动、运行时检查、import、模型加载、生成、保存和退出 |
| 模型加载耗时 | 模型构造/读取权重阶段，GPU 同步后计时 |
| 图片生成 | 文本编码、12 步去噪和解码的完整 pipeline 时间；秒/张、张/秒、百万像素/秒；PNG 保存不在此阶段内 |
| 配音生成 | `IndexTTS2.infer` 时间，含 WAV 保存；输出音频秒数、RTF、倍实时速度、字符/秒 |
| RTF | 实际生成秒数 ÷ 音频秒数，越小越快；小于 1 表示快于实时 |
| 倍实时速度 | 音频秒数 ÷ 实际生成秒数，RTF 的倒数 |

端到端图片耗时、音频 RTF 另外使用 hello 总耗时计算，避免把模型加载混入纯生成指标。资源的 min/max/mean 来自本次时间序列；单次 hello 不会伪造多轮生成耗时的分布或 p95/p99。

输出目录中新增四份记录：

- `performance.json`：统一入口，包含任务状态、阶段时间、应用性能指标、资源汇总及口径说明。
- `resources.json`：完整采样序列、总体和分阶段 min/max/mean、数据来源及限制。
- `resources.jsonl`：逐次采样实时追加，便于观察或读取中断前的数据。
- `runtime-performance.json`：模型加载/生成的同步时间、PyTorch 内存和各阶段 allocator 峰值。

失败、超时或 Ctrl+C 后仍保留已采到的数据；未完成阶段不计算吞吐率，`measurement_status=partial` 表示记录不完整。采样、GPU 同步和日志会引入少量测量开销，结果用于本机这次小样本的观测。

## 启动本地 WebUI

在运行环境和模型分别准备完成后执行：

```sh
uv run 300 Ideogram4
uv run 300 Indextts2
```

- Ideogram4：`http://127.0.0.1:7861`，输入结构化 JSON caption，点击生成，输出 PNG。
- IndexTTS2：`http://127.0.0.1:7860`，上传参考人声 WAV，输入配音文本，输出 WAV。

Ideogram4 默认 FP8 weight-only/BF16 计算、512×512、12-step sampler；IndexTTS2 明确关闭 DeepSpeed、自定义 CUDA kernel、GPT acceleration 和 torch.compile。每个 WebUI 单个实例串行处理生成请求。

```sh
uv run 300 Ideogram4 7871
uv run 300 Indextts2 --model-dir /another/IndexTTS-2
uv run 300 Ideogram4 --preparation-dir artifacts/init/300/<run-id>
uv run 300 Indextts2 --host 0.0.0.0
uv run 300 Ideogram4 --dry-run
```

启动阶段校验固定运行文件、依赖/源码身份和本地模型文件，设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`。缺少文件时提示执行 download，并停止；启动不安装依赖、不获取模型、不自动退回 CPU。复制其他机器模型后，stat 指纹变化时会重新读取 SHA256；同机未改变的文件可复用已经验证的身份。

`--dry-run` 不要求环境或模型已就绪；只显示有效 argv、目录和缺项。`--server-help` 查看已经准备的 runtime worker 帮助，不加载模型。

单次生成也可不启动 WebUI：

```sh
uv run 300 Ideogram4 --prompt-json caption.json --height 256 --width 256
uv run 300 Indextts2 --prompt "你好，这是配音测试。" --reference reference.wav
```

生成文件与请求参数写入 `artifacts/media/<应用>/`，启动记录写入 `artifacts/servers/300/<launch-id>/launch.json`。参考音频建议使用 WAV；其他格式涉及系统音频解码器，需另外配置。启动记录关联固定运行环境和模型身份；音频文件检查非空、有限且非全零，但听辨质量仍需要人工确认。

## 验证状态

2026-10-15 本机会话已具备 render/video 权限，rocminfo 正常枚举 gfx1151。独立 HIP 小探针在 Radeon 8060S 上执行返回 42；设备报告总内存 96 GiB。证据为本机 `artifacts/media-validation-20261015/hip-smoke-adrnkk2x/manifest.json`，不涉及模型加载。

2026-10-16 本机实测：使用 `/home/ops396/models` 中已有模型，两条 `hello` 均在 Radeon 8060S / gfx1151、torch 2.11.0+rocm7.2 上实际生成并通过输出检查；本轮没有下载模型。时长包含本地身份检查、模型加载、生成和输出验证，是单次观测，不能作为性能基准。

| 应用 | 实际结果 | 总耗时 | 本地验证记录 |
| --- | --- | --- | --- |
| Ideogram4 | 256×256 PNG，非纯色；目视为蓝色水面与画面上方的小红船 | 168.3 秒 | [manifest](../artifacts/hello/300/20261015T234745Z-ideogram4-td0cz7_o/manifest.json) |
| IndexTTS2 | 22050 Hz 单声道 WAV，3.135 秒，非静音且无 NaN/Inf | 72.9 秒 | [manifest](../artifacts/hello/300/20261015T233843Z-indextts2-2rccgnix/manifest.json) |

对应独立环境分别为 `artifacts/init/300/20261015T234635Z-ideogram4-s46lqg0u` 与 `artifacts/init/300/20261015T090934Z-indextts2-avevuadc`。前者已采用 Transformers 5.8.0 修复；旧环境 4.57.6 的失败证据保留在 `artifacts/hello/300/20261015T234031Z-ideogram4-25ogj67i`。没有修改模型配置或旧环境来绕过校验。

加入资源采样后，2026-10-16 本机串行复测也通过，复用相同 prepared 环境与本地模型：

| 应用 | 模型加载 | 实际生成 | hello 总耗时 | 资源采样 | 性能记录 |
| --- | --- | --- | --- | --- | --- |
| Ideogram4 | 112.805 s | 21.230 s | 138.751 s | 70 次 | [performance.json](../artifacts/hello/300/20261016T005010Z-ideogram4-vw3lns85/performance.json) |
| IndexTTS2 | 22.358 s | 37.546 s | 69.425 s | 35 次 | [performance.json](../artifacts/hello/300/20261016T005229Z-indextts2-m9ilegqt/performance.json) |

此次图片生成速率约 0.0471 张/s；音频时长 3.193 s，生成 RTF 为 11.760（约 0.085 倍实时速度），端到端 RTF 为 21.745。采样进程 RSS 峰值分别为 35.33 / 7.26 GiB；整卡 GPU 利用率采样均值为 19.67% / 49.26%，两者最大值均为 100%。这些整程均值包含加载等阶段，生成阶段的独立统计在 `resources.by_phase.generation` 中。整卡读数包含已有显存/GTT占用；不将本次最大值解读为任务独占消耗。

已逐项核对 JSONL 与 JSON samples 一致、采样数一致、汇总 min/max/mean 可由原始数据重算，两个采样器的错误列表均为空。工具测试共 229 项通过，执行命令为 `uv run --offline python -m unittest discover -s tests -v`，完整输出见 `artifacts/media-telemetry-tests.log`；`doctor` 也完成。这些检查覆盖路由、输入身份、缺失环境/模型、失败/超时/中断、串行锁、输出验证、计时/RTF 公式、GPU 映射、资源统计与监控异常隔离。已理解并验证小样本 GPU 推理链路和测量流程可用；语音未做人工听辨，较大分辨率、长文本及多轮稳定性仍未评测。下一步可打开生成文件确认主观质量，再使用自己的提示词或参考音频复测。

来源：[Ideogram4 固定源码](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2)、[IndexTTS2 v2.0.0](https://github.com/index-tts/index-tts/tree/830f6f8f94a51fea23ab1d639027a86200075a4e)、[ModelScope snapshot API](https://github.com/modelscope/modelscope/blob/v1.40.2/modelscope/hub/snapshot_download.py)。
