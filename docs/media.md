# AI 300 生图和配音命令

最后更新：2026-10-15。Ideogram4 和 IndexTTS2 的运行环境、模型下载、启动分为三个独立入口。命令不区分大小写，`Indextts2` 与 `IndexTTS2` 都接受。

## 运行环境初始化

```sh
uv run 300 init Ideogram4
uv run 300 init Indextts2
# 也可一次准备两个，按顺序执行
uv run 300 init Ideogram4 Indextts2
```

`init` 只准备固定源码、独立 Python venv、依赖、运行适配和 ROCm/import 小探针，不读取、下载或加载生图/TTS 模型与辅助权重。模型不存在也不影响环境初始化。IndexTTS2 的 NLTK 词典/标注运行资源在此阶段准备，以免启动时隐式联网获取语言资源。

两个应用各有新的 `artifacts/init/300/<run-id>/`，不会修改已有环境。Ideogram4 使用 Python 3.12、transformers 4.57.6；IndexTTS2 使用 Python 3.11、transformers 4.52.1。两套都固定 torch 2.11.0+rocm7.2，IndexTTS2 同时固定匹配的 torchaudio，并使用 soundfile 读写 WAV。IndexTTS2 上游 2.8.* 约束的调整、本地模型 IO 适配、源码 commit、补丁 SHA256、实际 import path、GPU 运算、freeze 与关键 Torch 动态库身份均保存到准备清单。没有模型推理结果时 `gpu_inference_validated` 保持 false。

普通 Python 依赖和 ModelScope SDK 默认使用清华 TUNA：`https://pypi.tuna.tsinghua.edu.cn/simple`。两个入口统一读取根目录 [pyproject.toml](../pyproject.toml) 的 default index；修改此处即可换源。安装命令显式传递 `--default-index`，构建子进程通过 `UV_DEFAULT_INDEX` / `PIP_INDEX_URL` 沿用同一配置，准备/下载清单记录 `package_default_index`。此配置只控制 Python 包；Python 解释器、Git 源码、NLTK 资源各用原有下载渠道，模型权重由 ModelScope 下载。

300 的 torch、torchaudio、triton-rocm 改用南京大学镜像 `https://mirror.nju.edu.cn/pytorch/whl/` 的固定 wheel 直链，按应用选择 cp312 / cp311。入口不再传 `--torch-backend rocm7.2`，因为该选项会绕过镜像，转用官方海外地址。torch / torchaudio 直链携带官方索引中的 SHA256，uv 安装时校验；Triton 上游索引未公布完整 wheel SHA256，按固定 3.6.0 文件安装。实际 URL 在初始化控制台、命令计划和 manifest 的 `rocm_wheel_sources` 中可见。版本、哈希和映射在 [media_wheels.py](../lab_tools/media_wheels.py)。入口没有配置官方回退源，镜像不可用时报告失败。[南京大学 PyTorch 镜像说明](https://help.mirrorz.org/pytorch/)、[uv torch-backend 行为](https://docs.astral.sh/uv/reference/settings/#pip_torch-backend)。

2026-10-15 本机检查：NJU 的 torch / torchaudio cp311、cp312 索引 SHA256 与官方一致；所需五个 wheel 的小范围 HTTP 请求返回 206、ZIP 文件头及 NJU 最终地址，没有跳转海外。torch / Triton 各读取前 4 MiB，单次请求含连接开销约 0.17–0.23 MiB/s；此结果不能保证完整下载速度。较小的 cp311 torchaudio（337,743 字节）在该请求中完整返回，其 SHA256 与官方一致。两个应用均通过真实 uv 依赖解析，普通依赖使用 TUNA，大型 torch / Triton wheel 未完整下载，模型推理未验证。可复查本机证据：`artifacts/research/rocm-mirror/verification.json`、`ideogram4-direct-dry-run.log`、`indextts2-resolve.log`。其他镜像实测存在缺文件或跳回官方源的问题，不能仅凭镜像索引页面可访问就认定 wheel 在国内。

此前已中断的初始化目录保留原始命令和日志。重新执行 `uv run 300 init Ideogram4` 或 `uv run 300 init Indextts2` 会创建使用新镜像的环境；不会修改旧环境。大包安装可传 `--build-timeout 1800` 提高预算；当前测速下默认 900 秒可能不足，需要允许长时间下载时显式传 `--build-timeout 0`，仍可 Ctrl+C 中断。

```sh
uv run 300 init Ideogram4 --dry-run  # 只查看命令计划，不安装
uv run 300 init Indextts2 --check-only
uv run 300 init Ideogram4 --build-timeout 1800
uv run 300 init Indextts2 --build-timeout 0  # 明确允许不限总时间
```

默认总预算为 900 秒；安装 ROCm wheel 较慢时可提高。每步显示完整命令、实时安装输出、日志和耗时；静默时每 20 秒报告等待时间。环境检查失败、超时或中断保留原目录和日志，重新运行会新建目录。当前这两个入口不支持 resume/update-sources，也不接受 `--model`；后续更新需要新环境和新证据。

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

工具测试共 132 项通过。固定源码的本地 IO 补丁已应用到临时副本并通过语法检查；命令路由、目录默认值、ModelScope 白名单、进度/校验流程与禁止隐式模型下载均完成工具验证。两套完整 PyTorch 环境尚未在本轮安装，用户尚未执行下载；因此没有生图/配音推理或质量结果。

来源：[Ideogram4 固定源码](https://github.com/ideogram-oss/ideogram4/tree/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2)、[IndexTTS2 v2.0.0](https://github.com/index-tts/index-tts/tree/830f6f8f94a51fea23ab1d639027a86200075a4e)、[ModelScope snapshot API](https://github.com/modelscope/modelscope/blob/v1.40.2/modelscope/hub/snapshot_download.py)。下一步：分别执行 init 与 download，再做一张 256×256 图片和一条中文短句，记录实际 runtime 与输出质量。
