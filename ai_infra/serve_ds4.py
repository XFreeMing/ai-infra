"""`uv run ds4`：修改下方变量后启动 ds4 HTTP 服务。"""
from pathlib import Path

# 用户配置：模型路径、GPU、context 与服务参数均在这里调整。
MODEL_PATH = Path("/data/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf")
SERVER_BINARY: Path | None = None  # None：从 uv run init 的 prepared 清单选择。
PREPARATION_DIR: Path | None = None  # 可填 artifacts/init/<run-id> 固定一次构建。
CUDA_VISIBLE_DEVICES = "0"  # 当前是 NVIDIA CUDA；HIP_VISIBLE_DEVICES 用于 ROCm。
HOST = "0.0.0.0"
PORT = 1111
CONTEXT_SIZE = 4096  # P0 初始容量；模型声明上限为 1,048,576，不代表都应预分配。
PREFILL_CHUNK = 4096
MAX_OUTPUT_TOKENS = 512  # 请求中的 max_tokens 可覆盖；P0 探针请求设为 32。
VISION_PATH: Path | None = None  # 文本 Flash 不需要；视觉模型使用匹配的 encoder。
EXTRA_ARGS: list[str] = []  # 例如 R2 的 --dspark、--mtp-model；普通基线默认关闭投机。


def server_arguments() -> list[str]:
    args = ["--cuda", "-m", str(MODEL_PATH), "-c", str(CONTEXT_SIZE),
            "--host", HOST, "--port", str(PORT), "--prefill-chunk", str(PREFILL_CHUNK),
            "-n", str(MAX_OUTPUT_TOKENS)]
    if VISION_PATH is not None:
        args += ["--vision", str(VISION_PATH)]
    return args + EXTRA_ARGS


def main(argv: list[str] | None = None) -> int:
    from ai_infra.server_launcher import launch
    return launch("ds4", MODEL_PATH, server_arguments(), cuda_devices=CUDA_VISIBLE_DEVICES,
                  binary=SERVER_BINARY, preparation_dir=PREPARATION_DIR, argv=argv,
                  configuration_file=Path(__file__))
