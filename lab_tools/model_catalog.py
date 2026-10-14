"""各推理引擎共用的模型产物身份与本地目录。"""
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Literal

from lab_tools.cli import WorkflowError


@unique
class ModelKey(StrEnum):
    """发布者__模型版本和变体__精度或量化方案__文件格式；value 用于记录与序列化。"""

    QWEN__QWEN3_6_35B_A3B__BF16__SAFETENSORS = 'qwen/qwen3.6-35b-a3b/bf16/safetensors'
    QWEN__QWEN3_8_FLASH_NEXT__FP8_BLOCK128__SAFETENSORS = 'qwen/qwen3.8-flash-next/fp8-block128/safetensors'
    NV_COMMUNITY__DEEPSEEK_V4_1_FLASH__FP8_NVFP4__SAFETENSORS = 'nv-community/deepseek-v4.1-flash/fp8-nvfp4/safetensors'


@dataclass(frozen=True)
class ModelSpec:
    """主要权重可混合精度；量化导出信息与引擎运行 dtype 分开维护。"""

    model_name: str
    repository_id: str
    model_path: Path
    weight_format: Literal['safetensors', 'gguf']
    weight_precisions: tuple[str, ...]
    quantization_method: str | None = None
    quantization_scheme: str | None = None
    revision: str | None = None


MODEL_CATALOG: dict[ModelKey, ModelSpec] = {
    ModelKey.QWEN__QWEN3_6_35B_A3B__BF16__SAFETENSORS: ModelSpec(
        model_name='Qwen3.6-35B-A3B',
        repository_id='Qwen/Qwen3.6-35B-A3B',
        model_path=Path('/data/models/Qwen3.6-35B-A3B-BF16'),
        weight_format='safetensors',
        weight_precisions=('bf16',),
    ),
    ModelKey.QWEN__QWEN3_8_FLASH_NEXT__FP8_BLOCK128__SAFETENSORS: ModelSpec(
        model_name='Qwen3.8-Flash-Next',
        repository_id='Qwen/Qwen3.8-Flash-Next-FP8',
        model_path=Path('/data/models/Qwen3.8-Flash-Next-FP8'),
        weight_format='safetensors',
        weight_precisions=('fp8_e4m3', 'bf16'),
        quantization_method='fp8',
        quantization_scheme='fp8_block128',
    ),
    ModelKey.NV_COMMUNITY__DEEPSEEK_V4_1_FLASH__FP8_NVFP4__SAFETENSORS: ModelSpec(
        model_name='DeepSeek-V4.1-Flash',
        repository_id='nv-community/DeepSeek-V4.1-Flash-NVFP4',
        model_path=Path('/data/models/DeepSeek-V4.1-Flash-NVFP4'),
        weight_format='safetensors',
        weight_precisions=('nvfp4', 'fp8_e4m3', 'bf16'),
        quantization_method='fp8',
        quantization_scheme='nvfp4_experts',
        revision='962b2d0e8b06e475b77ab32e694b5df51f7949fb',
    ),
}


def get_model(key: ModelKey) -> ModelSpec:
    if not isinstance(key, ModelKey):
        raise WorkflowError(f'模型 KEY 必须是 ModelKey 枚举：{key!r}')
    try:
        return MODEL_CATALOG[key]
    except KeyError:
        raise WorkflowError(f'模型目录未登记：ModelKey.{key.name}') from None
