"""CPU-only plumbing check; no model weights, GPU inference or quality claim."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch

    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(0)
    assert not torch.cuda.is_initialized(), 'CPU-only check requires an uninitialized GPU runtime'
    source = args.source.resolve()
    pipeline_path = source / 'src/ideogram4/pipeline_ideogram4.py'
    ae_path = source / 'src/ideogram4/autoencoder.py'
    ae = load('_cpu_check_autoencoder', ae_path)
    runtime = load('_cpu_check_runtime', args.runtime.resolve())
    # Execute the exact pinned decode method, without importing/loading the LLM.
    tree = ast.parse(pipeline_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Ideogram4Pipeline')
    decode = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_decode')
    namespace = {'torch': torch, 'Image': Image}
    exec(compile(ast.Module(body=[decode], type_ignores=[]), str(pipeline_path), 'exec'), namespace)
    pipeline_cls = type('SyntheticPipeline', (), {'_decode': namespace['_decode']})
    pipe = pipeline_cls()
    pipe.device, pipe.dtype = torch.device('cpu'), torch.float32
    pipe.config = SimpleNamespace(patch_size=2)
    pipe.latent_shift, pipe.latent_scale = torch.zeros(16), torch.ones(16)
    pipe.autoencoder = torch.nn.Module()
    pipe.autoencoder.decoder = ae.Decoder(resolution=16, in_channels=3, ch=32, out_ch=3,
                                        ch_mult=[1, 1], num_res_blocks=1, z_channels=4).eval()
    z = torch.randn(1, 16, 16, device='cpu')
    started = time.perf_counter()
    with torch.no_grad():
        reference = np.array(pipe._decode(z, grid_h=4, grid_w=4)[0])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metrics = runtime.RuntimeMetrics(args.output.with_name('cpu-decode-runtime.json'))
        runtime.instrument_image_subphases(pipeline_cls, metrics, 'cpu')
        actual = np.array(pipe._decode(z, grid_h=4, grid_w=4)[0])
    passed = bool(np.array_equal(reference, actual) and actual.shape == (16, 16, 3)
                  and not torch.cuda.is_initialized() and pipe.dtype == torch.float32
                  and pipe.device.type == 'cpu')
    metrics.finish('succeeded' if passed else 'failed')
    result = {
        'passed': passed, 'scope': 'synthetic small CPU decoder and exact pinned _decode method',
        'gpu_workload_executed': False, 'gpu_runtime_initialized': torch.cuda.is_initialized(),
        'full_model_loaded': False, 'full_2k_validated': False,
        'image_shape': list(actual.shape), 'pixels_equal': bool(np.array_equal(reference, actual)),
        'max_pixel_difference': int(np.abs(reference.astype(int) - actual.astype(int)).max()),
        'duration_seconds': time.perf_counter() - started, 'cpu_threads': torch.get_num_threads(),
        'python': sys.executable, 'torch': str(torch.__version__), 'torch_path': torch.__file__,
        'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'file_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__).resolve(), args.runtime.resolve(), pipeline_path, ae_path)},
        'limits': ['Random tiny decoder weights; no trained image quality validation.',
                   'CPU-to-CPU adapter check only; CUDA-to-CPU transfer tested with mocks.',
                   'Does not validate large CPU memory use or mixed GPU/CPU pipeline stability.'],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
