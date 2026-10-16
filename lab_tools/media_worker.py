"""Frozen runtime worker: ROCm/import probe, local-only loaders and media WebUIs."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import uuid

# -I intentionally removes the research checkout from import paths. Add only this frozen runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab_tools.media_models import APPS, TORCH_VERSION, inside

LOCAL_REPOS = {
    'facebook/w2v-bert-2.0': 'auxiliary/w2v-bert-2.0',
    'amphion/MaskGCT': 'auxiliary/MaskGCT',
    'funasr/campplus': 'auxiliary/campplus',
    'nvidia/bigvgan_v2_22khz_80band_256x': 'auxiliary/bigvgan',
    'ideogram-ai/ideogram-4-fp8': '.',
}


def local_model(repo: str) -> str:
    root = Path(os.environ['AI_INFRA_MEDIA_MODEL_DIR']).resolve()
    if Path(repo).is_absolute():
        path = Path(repo).resolve()
        if path != root:
            raise ValueError(f'模型路径未登记：{path}')
    else:
        if repo not in LOCAL_REPOS:
            raise ValueError(f'模型仓库没有本地映射，禁止隐式下载：{repo}')
        path = inside(root, LOCAL_REPOS[repo])
    if not path.is_dir():
        raise FileNotFoundError(f'本地模型目录不存在：{path}；先执行 uv run download。')
    return str(path)


def local_file(repo_id: str, filename: str, **kwargs) -> str:
    path = inside(Path(local_model(repo_id)), filename)
    if not path.is_file():
        # The Ideogram loader uses EntryNotFoundError to fall back from an index to a single file.
        from huggingface_hub.errors import EntryNotFoundError
        raise EntryNotFoundError(f'本地模型文件缺失，禁止下载：{path}')
    return str(path)


def torch_probe(app: str) -> dict:
    import torch
    if str(torch.__version__) != TORCH_VERSION or not torch.version.hip or torch.version.cuda or not torch.cuda.is_available():
        raise RuntimeError(f'需要 {TORCH_VERSION} 且能访问 GPU 的 ROCm PyTorch；实际 {torch.__version__}, HIP={torch.version.hip}, CUDA={torch.version.cuda}')
    properties = torch.cuda.get_device_properties(0)
    architecture = getattr(properties, 'gcnArchName', '')
    if not architecture.startswith('gfx1151'):
        raise RuntimeError(f'预期 gfx1151，实际 {architecture}')
    result = torch.arange(8, device='cuda').sum().item()
    assert result == 28
    dtype = torch.bfloat16 if app == 'ideogram4' else torch.float32
    x = torch.ones((8, 8), device='cuda', dtype=dtype)
    y = x @ x
    assert torch.isfinite(y).all() and float(y[0, 0]) == 8
    half = torch.ones((8, 8), device='cuda', dtype=torch.float16)
    assert float((half @ half)[0, 0]) == 8
    if app == 'ideogram4':
        stored = x.to(torch.float8_e4m3fn)
        assert torch.equal(stored.to(dtype), x)
        torch.nn.functional.scaled_dot_product_attention(x.reshape(1, 1, 8, 8), x.reshape(1, 1, 8, 8), x.reshape(1, 1, 8, 8))
    torch.cuda.synchronize()
    return {'torch': str(torch.__version__), 'torch_path': str(Path(torch.__file__).resolve()),
            'torch_hip': torch.version.hip, 'torch_cuda': torch.version.cuda,
            'gpu': properties.name, 'architecture': architecture, 'total_gpu_bytes': properties.total_memory,
            'tensor_sum': result, 'matrix_dtype': str(dtype), 'gpu_inference_validated': False}


def audio_backend() -> None:
    """Use libsndfile for WAV IO so newer torchaudio doesn't require TorchCodec/FFmpeg."""
    import soundfile as sf
    import torch
    import torchaudio

    def load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kwargs):
        if not normalize:
            raise ValueError('本入口的参考音频使用 normalized float32。')
        data, rate = sf.read(path, start=frame_offset, frames=num_frames, always_2d=True, dtype='float32')
        tensor = torch.from_numpy(data)
        return (tensor.T if channels_first else tensor), rate

    def save(path, tensor, sample_rate, channels_first=True, **kwargs):
        data = tensor.detach().cpu()
        if channels_first:
            data = data.T
        sf.write(path, data.numpy(), sample_rate, subtype='PCM_16')

    torchaudio.load, torchaudio.save = load, save


def import_app(source: Path, app: str):
    sys.path.insert(0, str(source / 'src' if app == 'ideogram4' else source))
    if app == 'indextts2':
        import nltk
        # All runtime dictionaries are prepared by init. Inference must not fetch any assets.
        def no_download(*args, **kwargs):
            raise RuntimeError('NLTK 运行资源缺失；请重新 init，启动阶段禁止下载。')
        nltk.download = no_download
        audio_backend()
    module = importlib.import_module(APPS[app]['module'])
    if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError(f'实际框架 import path 不来自固定源码：{module.__file__}')
    return module


def language_probe() -> dict:
    """Exercise delayed dictionary reads with downloads already disabled."""
    import nltk
    # Do not allow a user's global data directory to hide a missing init asset.
    nltk.data.path[:] = [os.environ['NLTK_DATA']]
    for resource in ('corpora/cmudict.zip', 'taggers/averaged_perceptron_tagger.zip'):
        nltk.data.find(resource)
    from indextts.utils.text_utils import get_text_syllable_num
    english = get_text_syllable_num('Hello world')
    mixed = get_text_syllable_num('你好 world')
    tags = nltk.pos_tag(['Hello', 'world'])
    sentences = nltk.sent_tokenize('Hello world. This is a test.')
    if english != 3 or mixed != 3 or len(tags) != 2 or len(sentences) != 2:
        raise RuntimeError('NLTK 语言运行资源检查失败。')
    return {'nltk': nltk.__version__, 'english_syllables': english,
            'mixed_syllables': mixed, 'pos_tag_count': len(tags), 'sentence_count': len(sentences)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('probe', 'run'))
    parser.add_argument('--app', choices=tuple(APPS), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--expected-provenance', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--prompt', help='Ideogram4 JSON caption，或 Indextts2 待合成文本；省略时启动 WebUI。')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sampler', default='V4_TURBO_12')
    args = parser.parse_args(argv)
    if args.model_dir:
        os.environ['AI_INFRA_MEDIA_MODEL_DIR'] = str(args.model_dir.resolve())
    provenance = torch_probe(args.app)
    module = import_app(args.source, args.app)
    provenance['module_path'] = str(Path(module.__file__).resolve())
    if args.expected_provenance:
        expected = json.loads(args.expected_provenance.read_text(encoding='utf-8'))
        if any(provenance[key] != expected[key] for key in ('torch_path', 'torch', 'module_path', 'torch_hip')):
            raise RuntimeError('运行时实际 import 路径或后端版本与准备证据不同。')
    if args.phase == 'probe':
        if args.app == 'indextts2':
            provenance['language_runtime'] = language_probe()
        args.output.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(provenance, ensure_ascii=False, indent=2))
        return 0
    import torch
    args.output.mkdir(parents=True, exist_ok=True)
    if args.app == 'ideogram4':
        if args.height % 16 or args.width % 16 or not (256 <= args.height <= 2048 and 256 <= args.width <= 2048):
            raise ValueError('图像高宽必须在 256–2048 且为 16 的倍数。')
        if args.sampler not in module.PRESETS:
            raise ValueError('未知 sampler。')
        pipe = module.Ideogram4Pipeline.from_pretrained(
            config=module.Ideogram4PipelineConfig(weights_repo=str(args.model_dir.resolve())),
            device='cuda', dtype=torch.bfloat16)

        def generate(caption, height=args.height, width=args.width, seed=args.seed):
            from ideogram4.caption_verifier import CaptionVerifier
            warnings = CaptionVerifier().verify_raw(caption)
            if warnings:
                raise ValueError('caption JSON 格式错误：' + '; '.join(warnings))
            preset = module.PRESETS[args.sampler]
            image = pipe(caption, height=int(height), width=int(width), seed=int(seed), num_steps=preset.num_steps,
                         guidance_schedule=preset.guidance_schedule, mu=preset.mu, std=preset.std)[0]
            path = args.output / (uuid.uuid4().hex + '.png')
            image.save(path)
            path.with_suffix('.json').write_text(json.dumps({'app': args.app, 'caption': json.loads(caption),
                'height': int(height), 'width': int(width), 'seed': int(seed), 'sampler': args.sampler,
                'launch_manifest': os.environ.get('AI_INFRA_LAUNCH_MANIFEST'), 'output': str(path)}, ensure_ascii=False, indent=2), encoding='utf-8')
            return str(path)

        if args.prompt:
            print(generate(args.prompt))
            return 0
        import gradio as gr
        with gr.Blocks(title='Ideogram4') as demo:
            gr.Markdown('# Ideogram4\n使用结构化 JSON 描述画面。')
            caption = gr.Textbox(label='JSON 提示词', lines=12, value=json.dumps({
                'high_level_description': 'A red boat floating on a quiet blue lake.',
                'style_description': {'aesthetics': 'calm and minimal', 'lighting': 'soft morning light',
                                      'medium': 'illustration', 'art_style': 'flat vector illustration', 'color_palette': ['#E53935', '#1565C0']},
                'compositional_deconstruction': {'background': 'A calm blue lake.', 'elements': []}}, ensure_ascii=False, indent=2))
            with gr.Row():
                height = gr.Slider(256, 2048, step=16, value=args.height, label='高度')
                width = gr.Slider(256, 2048, step=16, value=args.width, label='宽度')
                seed = gr.Number(value=args.seed, precision=0, label='随机种子')
            image = gr.Image(type='filepath', label='生成图片')
            gr.Button('生成').click(generate, inputs=[caption, height, width, seed], outputs=image, concurrency_limit=1)
    else:
        tts = module.IndexTTS2(cfg_path=str(args.model_dir / 'config.yaml'), model_dir=str(args.model_dir),
                               device='cuda:0', use_fp16=False, use_cuda_kernel=False, use_deepspeed=False,
                               use_accel=False, use_torch_compile=False)

        def synthesize(reference, text):
            if not reference or not text.strip():
                raise ValueError('请提供参考人声和待合成文本。')
            path = args.output / (uuid.uuid4().hex + '.wav')
            tts.infer(spk_audio_prompt=reference, text=text, output_path=str(path))
            import soundfile as sf
            import numpy as np
            audio, rate = sf.read(path)
            if not len(audio) or not np.isfinite(audio).all() or not np.any(audio):
                raise RuntimeError('音频输出为空、全零或包含 NaN/Inf。')
            path.with_suffix('.json').write_text(json.dumps({'app': args.app, 'text': text, 'reference': str(reference),
                'sample_rate': rate, 'duration_seconds': len(audio) / rate,
                'launch_manifest': os.environ.get('AI_INFRA_LAUNCH_MANIFEST'), 'output': str(path)}, ensure_ascii=False, indent=2), encoding='utf-8')
            return str(path)

        if args.prompt:
            print(synthesize(str(args.reference) if args.reference else None, args.prompt))
            return 0
        import gradio as gr
        with gr.Blocks(title='IndexTTS2') as demo:
            gr.Markdown('# IndexTTS2\n选择参考人声，输入配音文本。')
            reference = gr.Audio(type='filepath', format='wav', label='参考人声')
            text = gr.Textbox(label='配音文本', lines=5)
            audio = gr.Audio(type='filepath', label='生成音频')
            gr.Button('生成').click(synthesize, inputs=[reference, text], outputs=audio, concurrency_limit=1)
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port or APPS[args.app]['port'], share=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
