"""`uv run llama`：修改下方变量后启动 llama.cpp HTTP 服务。"""
from pathlib import Path

# 用户配置：切换模型时修改 MODEL_PATH，视觉模型另填写 MMPROJ_PATH。
MODEL_PATH = Path("/data/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf")
SERVER_BINARY: Path | None = None  # None：从 uv run init 的 prepared 清单选择。
PREPARATION_DIR: Path | None = None
CUDA_VISIBLE_DEVICES = "0"
HOST = "0.0.0.0"
PORT = 2222
CONTEXT_SIZE = 4096  # 与 ds4 的 P0 初始 context 一致。
BATCH_SIZE = 4096
UBATCH_SIZE = 2048
PARALLEL = 1
GPU_LAYERS = "all"  # 固定版本支持 all；不采用自动减少 offload 层数的配置。
JINJA = True
REASONING = "off"  # P0 使用 non-thinking；日常对话可改为 on/auto。
MMPROJ_PATH: Path | None = None  # 本次文本 GGUF 没有 mmproj；替换模型后再设置。
EXTRA_ARGS: list[str] = []


def server_arguments() -> list[str]:
    args = ["-m", str(MODEL_PATH), "-c", str(CONTEXT_SIZE), "--host", HOST,
            "--port", str(PORT), "--gpu-layers", GPU_LAYERS, "--fit", "off",
            "--parallel", str(PARALLEL), "-b", str(BATCH_SIZE), "-ub", str(UBATCH_SIZE),
            "--jinja" if JINJA else "--no-jinja", "--reasoning", REASONING]
    if MMPROJ_PATH is not None:
        args += ["--mmproj", str(MMPROJ_PATH)]
    return args + EXTRA_ARGS


def main(argv: list[str] | None = None) -> int:
    from ai_infra.server_launcher import launch
    return launch("llama", MODEL_PATH, server_arguments(), cuda_devices=CUDA_VISIBLE_DEVICES,
                  binary=SERVER_BINARY, preparation_dir=PREPARATION_DIR, argv=argv,
                  configuration_file=Path(__file__))
