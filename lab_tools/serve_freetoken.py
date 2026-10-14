"""uv run freetoken [6666]；模型配置和默认选择位于 freetoken_models.py。"""
from pathlib import Path

from lab_tools import freetoken_models
from lab_tools import model_catalog
from lab_tools.cli import WorkflowError

# 服务环境设置；模型路径、context 和推理参数统一在 freetoken_models.py 维护。
PYTHON_EXECUTABLE: Path | None = None
PREPARATION_DIR: Path | None = None
CUDA_VISIBLE_DEVICES = '0'
HOST = '0.0.0.0'
PORT = 6666
KERNEL_CACHE_DIR: Path | None = None  # None：使用 init 构建并校验的独立 kernel cache。


def server_arguments() -> list[str]:
    model = freetoken_models.selected_model()
    args = ['--model', str(model.model_path), '--served-model-name', model.model_path.name,
            '--host', HOST, '--max-seq-len-override', str(model.context_size),
            '--page-size', str(model.page_size), '--num-pages', str(model.num_kv_pages), '--dtype', model.dtype,
            '--max-prefill-length', str(model.max_prefill_length), '--max-running-requests', str(model.max_running_requests),
            '--max-output-tokens', str(model.max_output_tokens), '--memory-ratio', str(model.memory_ratio),
            '--moe-strategy', model.moe_strategy, '--attention-backend', model.attention_backend,
            '--reasoning-parser', model.reasoning_parser]
    if model.text_model_only:
        args.append('--text-model-only')
    return args + list(model.extra_args)


def main(argv: list[str] | None = None) -> int:
    from lab_tools.native_launcher import launch
    try:
        arguments = server_arguments()
    except WorkflowError as error:
        import sys
        print(f'无法启动 freetoken：{error}', file=sys.stderr)
        return 2
    extra_env = {'FREETOKEN_KERNEL_CACHE_DIR': str(KERNEL_CACHE_DIR)} if KERNEL_CACHE_DIR else None
    return launch('freetoken', arguments, port=PORT, cuda_devices=CUDA_VISIBLE_DEVICES,
                  python=PYTHON_EXECUTABLE, preparation_dir=PREPARATION_DIR,
                  runtime_environment=extra_env, configuration_file=Path(__file__),
                  configuration_files=(Path(freetoken_models.__file__), Path(model_catalog.__file__)), argv=argv)
