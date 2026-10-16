"""Run in the prepared interpreter to validate a single generated smoke artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def validate_reference(path: Path) -> dict:
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError('参考音频为空、全零或含 NaN/Inf。')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'sample_rate': rate,
            'channels': audio.shape[1], 'duration_seconds': len(audio) / rate}


def validate(app: str, output: Path, *, expected_width: int | None = None,
             expected_height: int | None = None) -> dict:
    if app == 'ideogram4' and any(type(value) is not int or value <= 0
                                 for value in (expected_width, expected_height)):
        raise ValueError('图片校验必须提供正整数 expected_width 和 expected_height。')
    suffix = '.png' if app == 'ideogram4' else '.wav'
    files = list(output.glob('*' + suffix))
    if len(files) != 1:
        raise ValueError(f'预期一个 {suffix} 输出，实际 {len(files)} 个。')
    path = files[0]
    if app == 'ideogram4':
        from PIL import Image
        with Image.open(path) as image:
            image.load()
            if image.format != 'PNG':
                raise ValueError('生成图片格式不符，预期 PNG。')
            if image.size != (expected_width, expected_height):
                raise ValueError(f'生成图片尺寸不符：预期 {expected_width}×{expected_height}，'
                                 f'实际 {image.width}×{image.height}。')
            extrema = image.convert('RGB').getextrema()
            if all(low == high for low, high in extrema):
                raise ValueError('生成图片为单一纯色，快速验证未通过。')
            details = {'width': image.width, 'height': image.height, 'pixel_extrema': extrema}
    else:
        import numpy as np
        import soundfile as sf
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        if not len(audio) or not np.isfinite(audio).all() or not np.any(audio):
            raise ValueError('音频为空、全零或含 NaN/Inf。')
        duration = len(audio) / rate
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio.astype('float64') ** 2)))
        if duration < 0.2 or rms < 0.00001:
            raise ValueError('音频过短或近乎静音，快速验证未通过。')
        details = {'sample_rate': rate, 'channels': audio.shape[1], 'frames': len(audio),
                   'duration_seconds': duration, 'peak': peak, 'rms': rms}
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'passed': True, 'app': app, 'output_path': str(path.resolve()),
            'size_bytes': path.stat().st_size, 'sha256': digest, **details}


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('必须为正整数。') from exc
    if result <= 0:
        raise argparse.ArgumentTypeError('必须为正整数。')
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', required=True, choices=('ideogram4', 'indextts2'))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--output', type=Path)
    inputs.add_argument('--reference', type=Path)
    parser.add_argument('--expected-width', type=positive_integer)
    parser.add_argument('--expected-height', type=positive_integer)
    parser.add_argument('--result', required=True, type=Path)
    args = parser.parse_args()
    if (args.app == 'ideogram4' and args.output is not None
            and (args.expected_width is None or args.expected_height is None)):
        parser.error('Ideogram4 图片校验必须提供 --expected-width 和 --expected-height。')
    result = (validate_reference(args.reference) if args.reference else
              validate(args.app, args.output, expected_width=args.expected_width,
                       expected_height=args.expected_height))
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
