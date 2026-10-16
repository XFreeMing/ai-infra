# AI 300 上运行 Ideogram4 和 IndexTTS2 的可行性

2026-10-15 更新：GPU 权限已解决，HIP 小探针返回 42；已新增运行环境 init、ModelScope download 和本地 WebUI 启动入口，见 [当前使用说明](../../../docs/media.md)。两套 Python runtime 和模型推理尚待执行。以下保留 2026-10-14 的初始调研与当时的权限阻塞记录。

研究日期：2026-10-14。目标机器：Ryzen AI Max+ 395 / Radeon 8060S / 128 GB / Ubuntu 24.04.4 / Linux 7.0.0-38 / ROCm 7.2.0。

**2026-10-14 的结论：两个模型都有本地部署路径，当时的机器环境尚不能判定 GPU 推理可用。** 当时本机进程打开 `/dev/kfd` 返回 Permission denied，ROCm 设备枚举失败；研究环境尚未安装 PyTorch。Ideogram4 应选择 FP8 路径并完成权重访问；IndexTTS2 必须替换上游的 CUDA 安装配置，并解决 Python / PyTorch 版本组合。优先验证 IndexTTS2，再验证 Ideogram4。

| 项目 | 官方接口与源码判断 | 当前 AI 300 验证状态 |
| --- | --- | --- |
| Ideogram4 | 有官方代码和权重；FP8 weight-only 转 BF16 计算，存在 ROCm 尝试路径；NF4 标为 CUDA 专用 | GPU 访问 blocked；权重匿名请求 401 GatedRepo；结构化 caption 的官方 verifier 通过；未生成图片 |
| IndexTTS2 | CPU 分支明确存在；关闭自定义 CUDA kernel / DeepSpeed 后主要使用 PyTorch 运算，存在 ROCm 尝试路径 | GPU 访问 blocked；原始 Numba 在 Python 3.12 下无可用 wheel 的解析检查失败；未生成音频 |

这里的“尝试路径”是 AI 根据源码形成的判断，不是作者对该整机的认证，也不是实际模型运行结果。GPU 设备选择、FP16/BF16 运算、模型加载和输出质量须依次验证。

## 版本与证据

- Ideogram4：官方 [ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4)，本轮读取 commit `990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`，该提交日期为 2026-06-30。
- IndexTTS2：官方 [index-tts/index-tts](https://github.com/index-tts/index-tts)，本轮固定 `v2.0.0` / commit `830f6f8f94a51fea23ab1d639027a86200075a4e`。当前 main `d9e41aac89fd00b3d71497fddb287b7f24613712` 的 README 默认介绍 2.5，不能把其 `--version 2` 选项当作 v2.0.0 的启动参数。
- 本机 readiness 执行：experiment ID `ai300-media-preflight`，run ID `20261014T095427Z-preflight-8hs28qnp`，parent commit `26d4889b5f27fcd5ffa14da8ff4e41fef7d1587e`，parent dirty=true。本轮没有修改 `sources/` 或新增子模块指针。
- 原始 stdout/stderr、环境、来源、文件 SHA256：本机 `artifacts/research/ai300-media/20261014T095427Z-preflight-8hs28qnp/manifest.json`。它记录 readiness，不是模型实验。资源清单另存本目录 `sources.json`。

观察到 CPU 型号为 `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`；hipconfig 输出 `7.2.26015-fc0010cf6a`；`os.open('/dev/kfd', O_RDWR)` 返回 errno 13；rocminfo 退出 1，提示当前用户不在 render 组。研究用 Python 3.12.3 未发现 torch、torchaudio、transformers、gradio、bitsandbytes；PATH 中未发现 ffmpeg 和 git-lfs。指定的 `/data/models`、项目 `models/` 不存在，此检查不覆盖整机其他应用的缓存。

## 共同前置条件

先让当前登录会话具备 GPU 访问权限，再准备每个模型自己的环境。AMD 的 GPU access 配置见 [ROCm 安装前提](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html)。管理员可核对 render/video 组与设备 ACL；本机通常需要以下操作并重新登录：

```sh
sudo usermod -aG render,video "$USER"
# 完整注销并重新登录后检查；旧终端会话不会自动获得新组
id
/opt/rocm/bin/rocminfo
```

仅设备节点存在、amdgpu 已加载或具有管理员权限，都不能替代普通用户进程的实际设备访问。本轮没有执行上面的系统权限修改。

AMD 的 [ROCm 7.2 Ryzen Linux 矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)列出 Max+ 395 / gfx1151，验证的组合为 PyTorch 2.9、Python 3.12、FP16。这里的本机内核和两个项目的依赖版本与该矩阵不同，因此不能直接套用“官方已验证”。

ROCm PyTorch 继续使用 `torch.cuda` API 和 `device='cuda'`，检查后端应看 `torch.version.hip`；`torch.version.cuda` 应为空。依据：[PyTorch HIP semantics](https://docs.pytorch.org/docs/2.14/notes/hip.html)。每套环境安装后先检查实际 import path，再执行小 tensor / 矩阵运算；拒绝 CPU-only 或 NVIDIA CUDA wheel：

```python
import torch
print(torch.__file__, torch.__version__, torch.version.hip, torch.version.cuda)
assert torch.version.hip and not torch.version.cuda
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))
x = torch.arange(8, device="cuda")
assert x.sum().item() == 28
torch.cuda.synchronize()
```

此探针尚未在本机执行，须在权限修复和隔离环境安装后运行。随后按模型需要验证 FP16/BF16 matmul、SDPA；Ideogram4 还需验证 float8 存储转 BF16。

## Ideogram4 的条件和启动

### 选择 FP8

官方 [README Model Zoo](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/README.md#model-zoo)将 NF4 标为 CUDA，FP8 标为 All。实际关键符号是 [quantized_loading.py 的 `Fp8Linear.forward`](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/src/ideogram4/quantized_loading.py)：FP8 权重转换为输入 dtype 后调用普通 `F.linear`。它不是原生 FP8 GEMM，因此不以 8060S 有 FP8 Tensor Core 为前提。attention 使用 PyTorch SDPA。

`run_inference.py` 的 `_default_quantization()` 只看 `torch.cuda.is_available()`；ROCm 也会返回 true，所以 **AMD 必须显式传 `--quantization fp8`**，不能依赖默认选择。`--device cuda` 在 ROCm PyTorch 中是正确写法。

### 环境和权重

[pyproject.toml](https://github.com/ideogram-oss/ideogram4/blob/990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2/pyproject.toml)要求 Python >=3.10、torch >=2.11、transformers、safetensors、accelerate、einops、Pillow、huggingface_hub、requests、bitsandbytes >=0.49.2。建议单独 Python 3.12 环境。本轮已从 [PyTorch 官方 ROCm 7.2 index](https://download.pytorch.org/whl/rocm7.2/torch/)确认 `torch-2.11.0+rocm7.2-cp312-...x86_64.whl` 存在；wheel 存在不证明其 gfx1151 kernel 可运行。

FP8 源码仍无条件 import bitsandbytes，因此必须检查该依赖在所选 ROCm 环境能正常 import。官方 [bitsandbytes 0.49.2 源码](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.49.2/bitsandbytes/__init__.py)有 CPU/CUDA 后端加载逻辑；本轮没有执行这个 import，不把 FP8 不调用 NF4 算子解释为依赖可以省略。

隔离环境的候选准备顺序如下，**尚未执行安装和 GPU 验证**。应在自己的固定版本 clone 中运行，不放入本研究工具的 `.venv`：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/rocm7.2 'torch==2.11.0+rocm7.2'
uv pip install --python .venv/bin/python -e .
# 运行前执行上面的 HIP/tensor 检查，并固定最终 freeze 与 wheel 身份
```

权重仓库 [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8)有 license gate。需要用户在 Hugging Face 获得访问并执行 `hf auth login`。本轮没有读取或写入任何 token，没有代用户接受协议；匿名文件 HEAD 请求为 401 GatedRepo。

已读取的模型 revision `ee79a7237b519f1402ceacf952f30c8a31ec5073` 文件合计约 **27.55 GB / 25.66 GiB**，包括 conditional DiT、unconditional DiT、Qwen text encoder 和 VAE。仅说“9.3B 模型”会漏掉第二个 DiT 与 text encoder。下载、安装与缓存建议预留至少 50 GB 磁盘，这是工作空间预算而非实测峰值。128 GB UMA 有容量上的尝试条件，真正 GPU 可分配内存与峰值仍需采样。

### 生成一张图片

下面命令要求环境、设备探针、模型授权和下载已经完成。入口已按固定源码的 argparse 核对，尚未生成图片：

```sh
HIP_VISIBLE_DEVICES=0 uv run --no-sync python run_inference.py \
  --device cuda --quantization fp8 \
  --no-magic-prompt --prompt "$(cat caption.json)" \
  --height 256 --width 256 --sampler-preset V4_TURBO_12 \
  --seed 0 --output first-image.png
```

`caption.json` 应是结构化 JSON，不能把普通中文一句话直接当完整 caption。本轮保存的小船 caption 已调用原始 `CaptionVerifier.verify_raw`，warnings=[]。输入在本机 `artifacts/research/ai300-media/ideogram-caption.json`。`--no-magic-prompt` 表示已提供完整 JSON；否则默认 magic prompt 需要 API key 并调用远端服务。上游还提供 Hive prompt/output moderation key 配置，用户可按部署方式配置。

当前官方入口是单次生成脚本，不是 WebUI / HTTP 服务。它的模型 loader 默认读取仓库版本，没有 CLI revision 参数；复现实验须给 loader 补齐 revision 固定或记录、校验实际缓存 snapshot。首次先做 256×256，再提升 512/1024，不能直接以 2K 默认例子评估是否兼容。

AMD 另有 [ROCm/hrx-demos Ideogram4 路径](https://github.com/ROCm/hrx-demos/blob/main/docs/quickstart.md)，其作者报告 W7900 / gfx1100 完整模型结果；这不能当作本机 gfx1151 结果，本轮不选它作为第一条安装路线。

## IndexTTS2 的条件和启动

### 设备与依赖

[infer_v2.py 的 `IndexTTS2.__init__`](https://github.com/index-tts/index-tts/blob/830f6f8f94a51fea23ab1d639027a86200075a4e/indextts/infer_v2.py)支持 CPU，模型 `.to(self.device)`；BigVGAN 在 `use_cuda_kernel=False` 时有普通 PyTorch 实现。因此 **CPU 有明确源码路径；ROCm GPU 有可尝试路径**。启动前应显式关闭 DeepSpeed、自定义 CUDA kernel、可选加速/compile，先做正确性 smoke。

不能原样执行 `uv sync --all-extras`：[v2.0.0 pyproject.toml](https://github.com/index-tts/index-tts/blob/830f6f8f94a51fea23ab1d639027a86200075a4e/pyproject.toml)把 Linux torch/torchaudio 源固定到 `cu128`，torch/torchaudio 固定 2.8.*。原始 `.python-version` 是 3.10；Numba 0.58.1 的官方范围是 Python <3.12，[版本矩阵](https://numba.readthedocs.io/en/0.58.1/user/installing.html#version-support-information)。本机 `uv pip install --dry-run --python python3.12 --only-binary :all: numba==0.58.1` 退出 1，明确提示无可用 wheel。

较少修改的基线是独立 Python **3.11** + torch/torchaudio **2.8**，先装 CPU 环境完成导入/短语音检查：

```sh
# 在固定 v2.0.0 的独立 clone 内；这是待执行的 CPU 基线，不是 GPU 配方
uv sync --python 3.11 --extra webui --no-sources --torch-backend cpu
```

GPU 候选可保留 2.8 并使用 [PyTorch ROCm 6.4 index](https://download.pytorch.org/whl/rocm6.4/torch/)的 2.8.0 wheel；需要重新选择源、生成独立 lock，再核对 gfx1151 编译支持和 tensor smoke。此组合不同于已安装的 ROCm 7.2 和 AMD 的 2.9/3.12 矩阵，**不作为已确认可用的安装命令**。另一条路线是按 gfx1151 / ROCm 7.2 环境适配 torch/torchaudio 与 Numba 约束，这会改变上游依赖，需要保存 patch、freeze 和质量对照。两条路线都不能靠强装 NVIDIA CUDA wheel 解决。

### 模型与输入

主模型 [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)公开可下载，本轮读取 revision `740dcaff396282ffb241903d150ac011cd4b1ede`，文件合计约 **5.90 GB**。用于配音还需要一段参考人声 WAV 与待合成文本。

源码初始化会加载额外模型：`facebook/w2v-bert-2.0` 的语义编码器、`amphion/MaskGCT` 的 semantic codec、`funasr/campplus` 的说话人模型、`nvidia/bigvgan_v2_22khz_80band_256x` 的 vocoder。`nvidia/` 是模型发布者，不表示默认 PyTorch 实现只能用 NVIDIA。主模型和所需辅助权重按文件清单相加约 **8.87 GB / 8.26 GiB**；不能为此下载 MaskGCT 全套或 BigVGAN discriminator/optimizer。各 auxiliary revisions 与 SHA256/LFS 身份已记入 `sources.json`。

建议预留 20–30 GB 磁盘用于模型、Python 环境和缓存；这是部署预算，实际内存峰值与速度未测量。主模型内的 Qwen emotion 模型在初始化中就会加载，不能因暂时不使用情绪功能而忽略它。音频读写还需检查 soundfile/librosa/torchaudio 及 ffmpeg；本机 PATH 未发现 ffmpeg。v2.0.0 示例音频可能需要 Git LFS 或自行提供参考 WAV。

主模型可按固定 revision 下载到启动命令使用的本地目录：

```sh
hf download IndexTeam/IndexTTS-2 \
  --revision 740dcaff396282ffb241903d150ac011cd4b1ede \
  --local-dir /data/models/IndexTTS-2
```

auxiliary 模型也应预先下载并记录实际 revisions；当前源码的 HF 加载默认不带 revision，首次启动可能联网下载，不能声称只下载主模型就已满足完整离线部署。

### 启动 WebUI 和生成音频

在环境与全部模型就绪后，v2.0.0 的 WebUI 命令是：

```sh
HIP_VISIBLE_DEVICES=0 uv run --no-sync python webui.py \
  --model_dir /data/models/IndexTTS-2 --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`，选择参考音频、填写文本。不要加 `--deepspeed` 或 `--cuda_kernel`。先完成 FP32 短句基线，FP16 tensor/质量验证后再加 `--fp16`。这个命令按 [固定版 webui.py 的 parser](https://github.com/index-tts/index-tts/blob/830f6f8f94a51fea23ab1d639027a86200075a4e/webui.py)核对；它没有 `--version 2`。GPU 不可用时上游可能自动退回 CPU，因此看见 UI 打开不能当作 ROCm 通过。

Python API 可显式要求 GPU并禁用自定义 CUDA 路径：

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="/data/models/IndexTTS-2/config.yaml",
               model_dir="/data/models/IndexTTS-2", device="cuda:0",
               use_fp16=False, use_cuda_kernel=False, use_deepspeed=False,
               use_accel=False, use_torch_compile=False)
tts.infer(spk_audio_prompt="reference.wav", text="你好，这是 AI 三百的语音验证。",
          output_path="first-speech.wav")
```

以上 API 尚未执行。验收需要 WAV 存在、采样率/时长正确、样本有限且非全零、人工听辨文本准确和无明显噪声，并记录实际设备、时间与峰值内存。CPU 基线须单独标记，不算 ROCm GPU 结果。

## 验证顺序和下一步

已理解：128 GB UMA 不排除这两个模型；Ideogram4 需要显式 FP8；IndexTTS2 需要非 CUDA 安装源、合适 Python 和关闭可选 CUDA 加速；现有 `uv run 300 init` 仅有 ds4/llama，不包含这两个新模型，本轮未添加未经模型验证的启动别名。

未解决：当前设备权限、两套实际可工作的 wheel/ABI/import 组合、Ideogram 权重访问、gfx1151 的 BF16/float8/SDPA、IndexTTS2 输出质量与吞吐。本轮没有安装大型框架、下载模型或执行 GPU kernel。PyTorch dry-run 的 wheel metadata 请求不支持 Range 并转为整包流式读取后被中止，保留日志，不记为解析通过。

下一步先修复会话的 render/video 权限，冻结干净 parent snapshot，单独准备 IndexTTS2 的 Python 3.11 CPU 基线与 ROCm 候选。小 tensor 预算 60 秒；一条 10–20 字中文语音预算 300 秒；随后 Ideogram4 首图 256×256 / 12 steps 预算 600 秒。出现 GPU 未识别、OOM、NaN/Inf、无输出或预算耗尽即停止并保留日志。通过后再接入 `uv run 300 init indextts2|ideogram4`，避免把条件性的安装方案包装成已经跑通的命令。
