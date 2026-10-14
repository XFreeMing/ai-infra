"""uv run freetoken [6666]；模型和常用启动参数在下方配置。"""
from pathlib import Path

# 用户配置：现有完整原生 BF16 模型；不使用 ds4 的 Q2 GGUF，不下载模型。
MODEL_PATH = Path('/data/models/Qwen3.6-35B-A3B-BF16')
PYTHON_EXECUTABLE: Path | None = None
PREPARATION_DIR: Path | None = None
CUDA_VISIBLE_DEVICES = '0'
HOST = '0.0.0.0'
PORT = 6666
CONTEXT_SIZE = 4096  # 本地 config 声明最大 262144；先做短 context smoke。
NUM_KV_PAGES = 4096  # 固定小 KV cache；GB10 启动 free-memory 基线会受文件缓存影响。
DTYPE = 'bfloat16'
MAX_PREFILL_LENGTH = 4096
MAX_RUNNING_REQUESTS = 1
MAX_OUTPUT_TOKENS = 512
MEMORY_RATIO = 0.80
MOE_STRATEGY = 'auto'  # GB10 + 本模型原生 BF16 experts 自动选择 fused resident。
ATTENTION_BACKEND = 'auto'
TEXT_MODEL_ONLY = True
REASONING_PARSER = 'off'
KERNEL_CACHE_DIR: Path | None = None  # None：使用 init 构建并校验的独立 kernel cache。
EXTRA_ARGS: list[str] = []


def server_arguments() -> list[str]:
    args = ['--model', str(MODEL_PATH), '--served-model-name', MODEL_PATH.name,
            '--host', HOST, '--max-seq-len-override', str(CONTEXT_SIZE), '--num-pages', str(NUM_KV_PAGES), '--dtype', DTYPE,
            '--max-prefill-length', str(MAX_PREFILL_LENGTH), '--max-running-requests', str(MAX_RUNNING_REQUESTS),
            '--max-output-tokens', str(MAX_OUTPUT_TOKENS), '--memory-ratio', str(MEMORY_RATIO),
            '--moe-strategy', MOE_STRATEGY, '--attention-backend', ATTENTION_BACKEND,
            '--reasoning-parser', REASONING_PARSER]
    if TEXT_MODEL_ONLY:
        args.append('--text-model-only')
    return args + EXTRA_ARGS


def main(argv: list[str] | None = None) -> int:
    from lab_tools.native_launcher import launch
    extra_env = {'FREETOKEN_KERNEL_CACHE_DIR': str(KERNEL_CACHE_DIR)} if KERNEL_CACHE_DIR else None
    return launch('freetoken', server_arguments(), port=PORT, cuda_devices=CUDA_VISIBLE_DEVICES,
                  python=PYTHON_EXECUTABLE, preparation_dir=PREPARATION_DIR,
                  runtime_environment=extra_env, configuration_file=Path(__file__), argv=argv)
