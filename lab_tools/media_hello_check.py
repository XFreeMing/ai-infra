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


def validate(app: str, output: Path) -> dict:
    suffix = '.png' if app == 'ideogram4' else '.wav'
    files = list(output.glob('*' + suffix))
    if len(files) != 1:
        raise ValueError(f'预期一个 {suffix} 输出，实际 {len(files)} 个。')
    path = files[0]
    if app == 'ideogram4':
        from PIL import Image
        with Image.open(path) as image:
            image.load()
            if image.format != 'PNG' or image.size != (256, 256):
                raise ValueError('生成图片格式或尺寸不符。')
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', required=True, choices=('ideogram4', 'indextts2'))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--output', type=Path)
    inputs.add_argument('--reference', type=Path)
    parser.add_argument('--result', required=True, type=Path)
    args = parser.parse_args()
    result = validate_reference(args.reference) if args.reference else validate(args.app, args.output)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
