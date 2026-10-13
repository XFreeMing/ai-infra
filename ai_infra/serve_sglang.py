"""uv run sglang [3333]；先编辑顶部变量，再在已准备的 CUDA venv 中启动。"""
from pathlib import Path

# 用户配置：本地目录，不填写在线 repo ID；脚本不会下载模型。
MODEL_PATH = Path('/data/models/DeepSeek-V4.1-Flash-NVFP4')
PYTHON_EXECUTABLE: Path | None = None
PREPARATION_DIR: Path | None = None
CUDA_VISIBLE_DEVICES = '0'
HOST = '0.0.0.0'
PORT = 3333
CONTEXT_SIZE = 4096  # 模型最大 1048576；先做短 context smoke。
TENSOR_PARALLEL_SIZE = 1  # 此 491 GiB checkpoint 无法装入单 GB10；多卡时同时改 GPU 可见列表。
CHUNKED_PREFILL_SIZE = 4096
MAX_RUNNING_REQUESTS = 1
MEM_FRACTION_STATIC = 0.80
QUANTIZATION: str | None = None  # 自动读取原生混合精度配置，不强制普通 FP8/NVFP4 全模型量化。
REASONING_PARSER = 'deepseek-v41'
TOOL_CALL_PARSER = 'deepseekv41'
EXTRA_ARGS: list[str] = []


def server_arguments() -> list[str]:
    args = ['--model-path', str(MODEL_PATH), '--served-model-name', MODEL_PATH.name,
            '--host', HOST, '--context-length', str(CONTEXT_SIZE), '--tp', str(TENSOR_PARALLEL_SIZE),
            '--chunked-prefill-size', str(CHUNKED_PREFILL_SIZE),
            '--max-running-requests', str(MAX_RUNNING_REQUESTS), '--mem-fraction-static', str(MEM_FRACTION_STATIC),
            '--reasoning-parser', REASONING_PARSER, '--tool-call-parser', TOOL_CALL_PARSER]
    if QUANTIZATION:
        args += ['--quantization', QUANTIZATION]
    return args + EXTRA_ARGS


def main(argv: list[str] | None = None) -> int:
    from ai_infra.native_launcher import launch
    return launch('sglang', server_arguments(), port=PORT, cuda_devices=CUDA_VISIBLE_DEVICES,
                  python=PYTHON_EXECUTABLE, preparation_dir=PREPARATION_DIR,
                  configuration_file=Path(__file__), argv=argv)
