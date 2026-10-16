"""Measure a frozen media worker without changing its prepared environment.

This file is copied into each hello run and executed with the prepared interpreter.
It deliberately has no project imports: the original worker owns its import path.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import functools
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time


PHASE_NAMES = {'runtime_check': '运行时检查', 'model_load': '模型加载', 'generation': '生成'}
SUBPHASE_NAMES = {'text_encoding': '提示词编码', 'decode_gpu_sync': 'VAE 解码前 GPU 完成同步',
                  'vae_decode': 'VAE 解码'}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def failure_status(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt) or (isinstance(error, SystemExit) and error.code in (130, 143)):
        return 'interrupted'
    return 'failed'


class RuntimeMetrics:
    """One invocation per phase; allocator bytes include resident model tensors."""

    def __init__(self, path: Path):
        self.path = path
        self.started = time.perf_counter()
        self.cuda = None
        self.data = {
            'schema_version': 1,
            'status': 'running',
            'started_at': utc_now(),
            'active_phase': None,
            'active_subphase': None,
            'phases': {},
            'subphases': [],
            'errors': [],
            'measurement_notes': [
                'Phase wall time is measured after synchronizing the current GPU and includes completion synchronization.',
                'PyTorch allocated/reserved bytes describe its allocator, not all system or GPU memory; resident model tensors are included.',
                'Image generation includes text encoding, denoising and decoding; image file saving is outside this phase.',
                'Audio generation includes the complete IndexTTS2.infer call, including audio file saving.',
                'Subphases record host call durations without resetting allocator peaks; GPU completion is only established where explicitly recorded.',
            ],
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, self.path)

    def warning(self, phase: dict, operation: str, error: Exception) -> None:
        phase.setdefault('measurement_warnings', []).append(f'{operation}: {type(error).__name__}: {error}')

    def prepare_cuda(self, phase: dict) -> None:
        try:
            if self.cuda is None:
                candidate = importlib.import_module('torch').cuda
                if candidate.is_available():
                    self.cuda = candidate
            if self.cuda is not None:
                self.cuda.synchronize()
                self.cuda.reset_peak_memory_stats()
                phase['gpu_synchronized_at_start'] = True
            else:
                phase['gpu_synchronized_at_start'] = False
        except Exception as error:
            phase['gpu_synchronized_at_start'] = False
            self.warning(phase, 'GPU synchronization/reset', error)
        phase.update(self.memory(prefix='torch_start_', phase=phase))

    def memory(self, *, prefix: str = 'torch_', phase: dict) -> dict:
        names = {'allocated_bytes': 'memory_allocated', 'reserved_bytes': 'memory_reserved',
                 'peak_allocated_bytes': 'max_memory_allocated', 'peak_reserved_bytes': 'max_memory_reserved'}
        result = {}
        for field, method in names.items():
            if prefix == 'torch_start_' and field.startswith('peak_'):
                continue
            result[prefix + field] = None
            if self.cuda is not None:
                try:
                    result[prefix + field] = int(getattr(self.cuda, method)())
                except Exception as error:
                    self.warning(phase, method, error)
        return result

    def call(self, name: str, function, *args, request: dict | None = None, **kwargs):
        # hello performs one load and one generation. Reject nested measurements
        # rather than resetting the allocator peaks of an enclosing phase.
        if self.data['active_phase'] is not None:
            return function(*args, **kwargs)
        phase = {'status': 'running', 'started_at': utc_now(), 'duration_seconds': None}
        if request:
            phase['request'] = request
        self.data['phases'][name] = phase
        self.data['active_phase'] = name
        self.save()
        print(f'性能阶段开始：{PHASE_NAMES[name]}', flush=True)
        # Include torch import/CUDA setup in the preparation duration separately;
        # phase wall time begins only after preceding GPU work has completed.
        preparation_started = time.perf_counter()
        started = preparation_started
        succeeded = False
        try:
            self.prepare_cuda(phase)
            phase['measurement_setup_seconds'] = time.perf_counter() - preparation_started
            started = time.perf_counter()
            phase['measurement_started_at'] = utc_now()
            self.save()
            result = function(*args, **kwargs)
            if self.cuda is not None:
                try:
                    self.cuda.synchronize()
                    phase['gpu_synchronized_at_end'] = True
                except Exception as error:
                    phase['gpu_synchronized_at_end'] = False
                    self.warning(phase, 'GPU completion synchronization', error)
            else:
                phase['gpu_synchronized_at_end'] = False
            phase['status'] = 'succeeded'
            succeeded = True
            return result
        except BaseException as error:
            phase['status'] = failure_status(error)
            phase['error'] = f'{type(error).__name__}: {error}'
            self.data['errors'].append({'phase': name, 'type': type(error).__name__, 'message': str(error)})
            # Do not wait for an in-flight GPU kernel when interrupted/failed.
            phase['gpu_synchronized_at_end'] = False
            raise
        finally:
            phase['duration_seconds'] = time.perf_counter() - started
            phase['finished_at'] = utc_now()
            phase['duration_complete'] = succeeded
            phase.update(self.memory(phase=phase))
            self.data['active_phase'] = None
            self.save()
            state = '完成' if succeeded else '未完成'
            print(f'性能阶段{state}：{PHASE_NAMES[name]}｜耗时 {phase["duration_seconds"]:.3f} 秒', flush=True)

    def finish(self, status: str, error: BaseException | None = None) -> None:
        self.data.update(status=status, finished_at=utc_now(), duration_seconds=time.perf_counter() - self.started)
        if error is not None and not self.data['errors']:
            self.data['errors'].append({'phase': self.data['active_phase'], 'type': type(error).__name__, 'message': str(error)})
        self.save()

    def subphase(self, name: str, function, *args, request: dict | None = None, **kwargs):
        """Publish diagnostic boundaries without disturbing the enclosing allocator peak."""
        phase = {'name': name, 'parent_phase': self.data['active_phase'], 'status': 'running',
                 'started_at': utc_now(), 'duration_seconds': None, 'duration_complete': False}
        if request:
            phase['request'] = request
        self.data['subphases'].append(phase)
        self.data['active_subphase'] = name
        self.save()
        device = f'｜设备 {request["device"]}' if request and request.get('device') else ''
        print(f'生成子阶段开始：{SUBPHASE_NAMES[name]}{device}', flush=True)
        started = time.perf_counter()
        try:
            result = function(*args, **kwargs)
            phase.update(status='succeeded', duration_complete=True)
            return result
        except BaseException as error:
            phase.update(status=failure_status(error), error=f'{type(error).__name__}: {error}')
            self.data['errors'].append({'phase': name, 'type': type(error).__name__, 'message': str(error)})
            raise
        finally:
            phase.update(duration_seconds=time.perf_counter() - started, finished_at=utc_now())
            self.data['active_subphase'] = None
            self.save()
            state = '完成' if phase['duration_complete'] else '未完成'
            print(f'生成子阶段{state}：{SUBPHASE_NAMES[name]}{device}｜耗时 {phase["duration_seconds"]:.3f} 秒', flush=True)


def instrument_method(cls, method_name: str, metrics: RuntimeMetrics, phase: str, request=None) -> None:
    """Retain descriptors, signatures and arguments of the original method."""
    descriptor = inspect.getattr_static(cls, method_name)
    descriptor_type = type(descriptor) if isinstance(descriptor, (classmethod, staticmethod)) else None
    original = descriptor.__func__ if descriptor_type else descriptor

    @functools.wraps(original)
    def measured(*args, **kwargs):
        details = request(args, kwargs) if request else None
        return metrics.call(phase, original, *args, request=details, **kwargs)

    setattr(cls, method_name, descriptor_type(measured) if descriptor_type else measured)


def instrument_image_subphases(cls, metrics: RuntimeMetrics, decode_device: str) -> None:
    """Adapt this process only; preserve the frozen worker and upstream source."""
    if decode_device == 'cpu' and not callable(getattr(cls, '_decode', None)):
        raise RuntimeError('当前 Ideogram4 pipeline 没有可用的 _decode；CPU VAE 请求不能回退到 GPU。')
    metrics.data['decoder'] = {'requested_device': decode_device, 'effective_device': None,
                               'dtype': 'torch.float32' if decode_device == 'cpu' else None,
                               'experimental_cpu_bypass': decode_device == 'cpu',
                               'uses_original_decode': True}
    metrics.save()
    if callable(getattr(cls, '_encode_text', None)):
        original_encode = cls._encode_text

        @functools.wraps(original_encode)
        def encode(self, *args, **kwargs):
            return metrics.subphase('text_encoding', original_encode, self, *args,
                                    request={'device': 'GPU', 'gpu_completion_synchronized': False}, **kwargs)

        cls._encode_text = encode
    if not callable(getattr(cls, '_decode', None)):
        return
    original_decode = cls._decode

    @functools.wraps(original_decode)
    def decode(self, z, *args, **kwargs):
        if decode_device == 'gpu':
            metrics.data['decoder'].update(effective_device='gpu', dtype=str(self.dtype))
            return metrics.subphase('vae_decode', original_decode, self, z, *args,
                                    request={'device': 'GPU', 'dtype': str(self.dtype)}, **kwargs)
        torch = importlib.import_module('torch')
        # Fail here if earlier GPU work failed. Never label that failure CPU VAE work.
        source_device = getattr(getattr(z, 'device', None), 'type', None)
        synchronized = source_device == 'cuda'
        if synchronized:
            metrics.subphase('decode_gpu_sync', torch.cuda.synchronize,
                             request={'device': 'GPU', 'purpose': 'complete_prior_gpu_work_before_cpu_decode'})

        def cpu_decode():
            proxy = copy.copy(self)
            proxy.device = torch.device('cpu')
            proxy.dtype = torch.float32
            cpu_z = z.to(device='cpu', dtype=torch.float32)
            proxy.latent_shift = self.latent_shift.to(device='cpu', dtype=torch.float32)
            proxy.latent_scale = self.latent_scale.to(device='cpu', dtype=torch.float32)
            # Keep VAE on CPU for the lifetime of this process. Restoring it to GPU
            # would introduce more GPU operations and defeat the bypass.
            proxy.autoencoder = self.autoencoder.to(device='cpu', dtype=torch.float32)
            metrics.data['decoder']['effective_device'] = 'cpu'
            metrics.save()
            return original_decode(proxy, cpu_z, *args, **kwargs)

        return metrics.subphase('vae_decode', cpu_decode,
                                request={'device': 'CPU', 'dtype': 'torch.float32',
                                         'includes_device_transfer': source_device != 'cpu',
                                         'prior_gpu_work_synchronized': synchronized,
                                         'source_device': source_device})

    cls._decode = decode


def instrument_worker(worker, metrics: RuntimeMetrics, decode_device: str = 'gpu') -> None:
    original_probe = worker.torch_probe
    original_import = worker.import_app
    wrapped_classes = set()

    @functools.wraps(original_probe)
    def probe(*args, **kwargs):
        return metrics.call('runtime_check', original_probe, *args, **kwargs)

    @functools.wraps(original_import)
    def import_app(source, app):
        module = original_import(source, app)
        cls = module.Ideogram4Pipeline if app == 'ideogram4' else module.IndexTTS2
        if cls not in wrapped_classes:
            if app == 'ideogram4':
                instrument_image_subphases(cls, metrics, decode_device)
                instrument_method(cls, 'from_pretrained', metrics, 'model_load')
                instrument_method(cls, '__call__', metrics, 'generation',
                                  lambda args, kwargs: {key: kwargs[key] for key in ('height', 'width', 'num_steps', 'seed') if key in kwargs})
            else:
                instrument_method(cls, '__init__', metrics, 'model_load')
                instrument_method(cls, 'infer', metrics, 'generation')
            wrapped_classes.add(cls)
        return module

    worker.torch_probe, worker.import_app = probe, import_app


def load_worker(path: Path):
    name = '_ai_infra_frozen_media_worker'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'无法加载已准备的 worker：{path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, type=Path)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--decode-device', choices=('cpu', 'gpu'), default='gpu',
                        help='Ideogram4 VAE 解码设备；cpu 是实验性旁路，GPU 文本编码和采样仍保留。')
    parser.add_argument('worker_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    worker_args = args.worker_args[1:] if args.worker_args[:1] == ['--'] else args.worker_args
    if not worker_args:
        parser.error('请在 -- 后提供原 worker 参数。')
    metrics = RuntimeMetrics(args.metrics)

    def terminated(signum, frame):
        raise SystemExit(128 + signum)

    previous_handler = signal.signal(signal.SIGTERM, terminated)
    try:
        worker = load_worker(args.worker.resolve())
        instrument_worker(worker, metrics, decode_device=args.decode_device)
        code = worker.main(worker_args)
        metrics.finish('succeeded' if not code else 'failed')
        return code or 0
    except BaseException as error:
        metrics.finish(failure_status(error), error)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
