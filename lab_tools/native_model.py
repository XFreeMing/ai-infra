"""Local safetensors admission; this module never downloads models."""
from __future__ import annotations

import json
from pathlib import Path

from lab_tools.cli import WorkflowError
from lab_tools.prepare import sha256
from lab_tools.model_catalog import ModelKey, get_model

MODEL_KEY = ModelKey.NV_COMMUNITY__DEEPSEEK_V4_1_FLASH__FP8_NVFP4__SAFETENSORS
_REFERENCE_MODEL = get_model(MODEL_KEY)
MODEL_ID = _REFERENCE_MODEL.repository_id
MODEL_REVISION = _REFERENCE_MODEL.revision
MODEL_DIR = _REFERENCE_MODEL.model_path
PROFILE = Path('research/studies/ds4-inference/NATIVE_MODEL.json')


def profile(root: Path) -> dict:
    return json.loads((root / PROFILE).read_text())


def contained_file(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise WorkflowError(f'模型文件越出本地目录：{name}')
    if not path.is_file():
        raise WorkflowError(f'缺少模型文件：{path}；请手动执行 ModelScope 下载脚本。')
    return path


def inspect_model(directory: Path, expected: dict | None = None) -> dict:
    if not directory.is_dir():
        raise WorkflowError(f'本地模型目录不存在：{directory}；请手动运行 scripts/download_deepseek_v41_flash_nvfp4.py。')
    config_path = contained_file(directory, 'config.json')
    config = json.loads(config_path.read_text())
    index_path = contained_file(directory, 'model.safetensors.index.json')
    index = json.loads(index_path.read_text())
    names = sorted(set(index.get('weight_map', {}).values()))
    if not names or any(not name.endswith('.safetensors') for name in names):
        raise WorkflowError('模型 index 没有有效 safetensors 分片。')
    files = [contained_file(directory, name) for name in names]
    if any(path.stat().st_size == 0 for path in files):
        raise WorkflowError('模型包含空权重分片。')
    contained_file(directory, 'tokenizer.json')
    contained_file(directory, 'tokenizer_config.json')
    quant = config.get('quantization_config', {})
    if expected:
        if config.get('model_type') != expected['model_type'] or \
                quant.get('moe_quant_algo') != expected['moe_quant_algo']:
            raise WorkflowError('默认目录中的模型不是冻结的 DeepSeek V4.1 NVFP4 配置。')
        records = {item['path']: item for item in expected['files']}
        expected_shards = {name for name in records if name.endswith('.safetensors')}
        if set(names) != expected_shards:
            raise WorkflowError('权重分片集合与冻结的 ModelScope revision 不同。')
        for path in [config_path, index_path, *files]:
            record = records[path.relative_to(directory.resolve()).as_posix()]
            if path.stat().st_size != record['size_bytes']:
                raise WorkflowError(f'模型文件大小不匹配：{path}')
            if path in (config_path, index_path) and sha256(path) != record['sha256']:
                raise WorkflowError(f'模型配置/index SHA256 不匹配：{path}')
    return {'path': str(directory.resolve()), 'model_type': config.get('model_type'),
            'architectures': config.get('architectures'), 'quant_method': quant.get('quant_method'),
            'moe_quant_algo': quant.get('moe_quant_algo'), 'shard_count': len(files),
            'weight_bytes': sum(path.stat().st_size for path in files),
            'config_sha256': sha256(config_path), 'index_sha256': sha256(index_path),
            'full_weights_sha256_checked': False,
            'max_position_embeddings': config.get('text_config', config).get('max_position_embeddings')}


def verify_download(directory: Path, expected: dict) -> dict:
    """Full hashes are only requested by the explicit manual download workflow."""
    checked = []
    for item in expected['files']:
        path = contained_file(directory, item['path'])
        if path.stat().st_size != item['size_bytes'] or sha256(path) != item['sha256']:
            raise WorkflowError(f'下载文件身份不匹配：{path}')
        checked.append(item['path'])
    return {'model_id': expected['model_id'], 'revision': expected['revision'],
            'path': str(directory.resolve()), 'verified_files': checked,
            'full_weights_sha256_checked': True, 'gpu_inference_validated': False}
