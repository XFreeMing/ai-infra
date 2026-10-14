"""FreeToken-specific model admission at the pinned upstream version; no downloads."""
from pathlib import Path
import json

from ai_infra.cli import WorkflowError
from ai_infra.native_model import contained_file, inspect_model

MODEL_DIR = Path('/data/models/Qwen3.6-35B-A3B-BF16')
PROFILE = Path('research/studies/freetoken-inference/NATIVE_MODEL.json')


def profile(root: Path) -> dict:
    return json.loads((root / PROFILE).read_text())


def inspect_freetoken_model(directory: Path, expected: dict | None = None) -> dict:
    # At 95ac86f GGUF_ARCH_TO_REGISTRY only contains gemma4, so a ds4/Qwen
    # GGUF must not be treated as interchangeable with native safetensors.
    if directory.suffix == '.gguf':
        raise WorkflowError('此入口使用本地 safetensors 目录；FreeToken 当前 GGUF 注册仅支持 Gemma4，不能复用 ds4/Qwen GGUF。')
    record = inspect_model(directory, expected)
    config = json.loads(contained_file(directory, 'config.json').read_text())
    quant = config.get('quantization_config') or {}
    sidecar = directory / 'hf_quant_config.json'
    side_quant = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    if config.get('model_type') in ('deepseek_v4', 'deepseek_v41') and \
            ('NVFP4' in json.dumps(quant).upper() or 'NVFP4' in json.dumps(side_quant).upper()):
        raise WorkflowError('FreeToken 95ac86f 的 DeepSeek V4/V4.1 专家 reader 仅支持 MXFP4；缺少 NVFP4 expert reader。'
                            '请更换兼容模型，或在核对上游实现后更新固定版本。')
    return record
