"""FreeToken 模型配置；调整 MODELS 中的对象，用 DEFAULT_MODEL 切换默认模型。"""
from dataclasses import dataclass
from pathlib import Path

from lab_tools.cli import WorkflowError
from lab_tools.model_catalog import ModelKey, get_model


DEFAULT_MODEL = ModelKey.QWEN__QWEN3_6_35B_A3B__BF16__SAFETENSORS


@dataclass(frozen=True)
class FreeTokenModelConfig:
    model: ModelKey
    model_path_override: Path | None = None
    context_size: int = 4096
    page_size: int = 1
    kv_pages: int | None = None
    dtype: str = 'bfloat16'
    max_prefill_length: int = 4096
    max_running_requests: int = 1
    max_output_tokens: int = 512
    memory_ratio: float = 0.80
    moe_strategy: str = 'auto'
    attention_backend: str = 'auto'
    text_model_only: bool = True
    reasoning_parser: str = 'off'
    extra_args: tuple[str, ...] = ()

    @property
    def model_path(self) -> Path:
        return self.model_path_override if self.model_path_override is not None else get_model(self.model).model_path

    @property
    def model_id(self) -> str:
        return get_model(self.model).repository_id

    @property
    def num_kv_pages(self) -> int:
        if self.context_size < 1 or self.page_size < 1:
            raise WorkflowError('context_size 和 page_size 必须是正整数。')
        pages = self.kv_pages
        if pages is None:
            pages = (self.context_size + self.page_size - 1) // self.page_size
        if pages < 2:
            raise WorkflowError('FreeToken KV cache 至少需要两个 pages。')
        return pages


MODELS: dict[ModelKey, FreeTokenModelConfig] = {
    ModelKey.QWEN__QWEN3_6_35B_A3B__BF16__SAFETENSORS: FreeTokenModelConfig(
        model=ModelKey.QWEN__QWEN3_6_35B_A3B__BF16__SAFETENSORS,
        context_size=262144,
        page_size=1,
    ),
    # 格式有 reader；此 FP8 checkpoint 的文本权重已超过单 GB10 总内存。
    ModelKey.QWEN__QWEN3_8_FLASH_NEXT__FP8_BLOCK128__SAFETENSORS: FreeTokenModelConfig(
        model=ModelKey.QWEN__QWEN3_8_FLASH_NEXT__FP8_BLOCK128__SAFETENSORS,
        context_size=4096,
        page_size=64,
        extra_args=('--ple-backend', 'disk'),
    ),
}


def selected_model() -> FreeTokenModelConfig:
    choices = ', '.join(f'ModelKey.{key.name}' for key in MODELS)
    if not isinstance(DEFAULT_MODEL, ModelKey):
        raise WorkflowError(f'未知 FreeToken 默认模型：{DEFAULT_MODEL}；请使用枚举，可选：{choices}')
    try:
        selected = MODELS[DEFAULT_MODEL]
    except KeyError:
        raise WorkflowError(f'FreeToken 未配置模型：ModelKey.{DEFAULT_MODEL.name}；可选：{choices}') from None
    if selected.model is not DEFAULT_MODEL:
        raise WorkflowError(f'FreeToken 配置 KEY 与模型身份不一致：ModelKey.{DEFAULT_MODEL.name}')
    get_model(selected.model)
    return selected
