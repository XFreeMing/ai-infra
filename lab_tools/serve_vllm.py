"""uv run vllm [5555]；先编辑顶部变量，再在已准备的 CUDA venv 中启动。"""
from pathlib import Path
from lab_tools.model_catalog import ModelKey, get_model

# 用户配置：模型必须已手动下载为本地完整 safetensors 目录。
MODEL_PATH = get_model(ModelKey.NV_COMMUNITY__DEEPSEEK_V4_1_FLASH__FP8_NVFP4__SAFETENSORS).model_path
PYTHON_EXECUTABLE: Path | None = None
PREPARATION_DIR: Path | None = None
CUDA_VISIBLE_DEVICES = '0'
HOST = '0.0.0.0'
PORT = 5555
CONTEXT_SIZE = 4096
TENSOR_PARALLEL_SIZE = 1  # 单 GB10 无法容纳该 checkpoint；配置多卡时同时改 GPU 可见列表。
MAX_NUM_SEQS = 1
MAX_NUM_BATCHED_TOKENS = 4096
GPU_MEMORY_UTILIZATION = 0.80
TOKENIZER_MODE = 'deepseek_v41'
REASONING_PARSER = 'deepseek_v41'
LANGUAGE_MODEL_ONLY = True
QUANTIZATION: str | None = None  # 根据 config.json 的 fp8 + moe_quant_algo=NVFP4 自动选择专用配置。
EXTRA_ARGS: list[str] = []


def server_arguments() -> list[str]:
    args = ['--model', str(MODEL_PATH), '--served-model-name', MODEL_PATH.name,
            '--host', HOST, '--max-model-len', str(CONTEXT_SIZE),
            '--tensor-parallel-size', str(TENSOR_PARALLEL_SIZE), '--max-num-seqs', str(MAX_NUM_SEQS),
            '--max-num-batched-tokens', str(MAX_NUM_BATCHED_TOKENS),
            '--gpu-memory-utilization', str(GPU_MEMORY_UTILIZATION), '--tokenizer-mode', TOKENIZER_MODE,
            '--reasoning-parser', REASONING_PARSER, '--enable-chunked-prefill', '--no-enable-prefix-caching']
    if LANGUAGE_MODEL_ONLY:
        args.append('--language-model-only')
    if QUANTIZATION:
        args += ['--quantization', QUANTIZATION]
    return args + EXTRA_ARGS


def main(argv: list[str] | None = None) -> int:
    from lab_tools.native_launcher import launch
    return launch('vllm', server_arguments(), port=PORT, cuda_devices=CUDA_VISIBLE_DEVICES,
                  python=PYTHON_EXECUTABLE, preparation_dir=PREPARATION_DIR,
                  configuration_file=Path(__file__), argv=argv)
