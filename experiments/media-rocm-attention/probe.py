#!/usr/bin/env python3
"""只读 ROCm BF16 SDPA 探针；不加载模型、不联网、不修改环境。

用目标运行环境的 Python 执行。每次仅允许一个 SDPA backend；同步计时包含
首次调用的编译/初始化开销。长序列禁止 MATH 和正确性基线，以免二次方显存分配。
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL 必须由调用方显式设置，本脚本只记录它。
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('必须为正整数')
    return number


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def memory(torch) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        'device_free_bytes': free,
        'device_total_bytes': total,
        'allocated_bytes': torch.cuda.memory_allocated(),
        'reserved_bytes': torch.cuda.memory_reserved(),
    }


def make_mask(torch, name: str, length: int, text_tokens: int):
    if name in ('no_mask', 'is_causal'):
        return None
    if name == 'broadcast_key_padding':
        mask = torch.ones((1, 1, 1, length), dtype=torch.bool, device='cuda:0')
        mask[..., max(1, length - max(1, length // 16)):] = False
        return mask
    # 保留真实 dense mask 行跨度 L，不进行对齐 padding，便于发现 L % 8 的限制。
    mask = torch.ones((length, length), dtype=torch.bool, device='cuda:0')
    if name == 'qwen_dense_causal_text_keys':
        mask.tril_()
        mask[:, text_tokens:] = False
    return mask.view(1, 1, length, length)


def run(args, report: dict) -> int:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    report['environment'].update({
        'torch_version': torch.__version__,
        'torch_path': str(Path(torch.__file__).resolve()),
        'torch_git_version': torch.version.git_version,
        'hip_version': torch.version.hip,
        'cuda_version': torch.version.cuda,
        'gpu_available': torch.cuda.is_available(),
    })
    save(args.output, report)
    if not torch.__version__.startswith('2.11.') or not torch.version.hip:
        raise RuntimeError('本探针要求 PyTorch 2.11.x ROCm 环境；不会安装或更换依赖。')
    if not torch.cuda.is_available():
        raise RuntimeError('ROCm GPU 不可用。')
    properties = torch.cuda.get_device_properties(0)
    report['environment'].update({
        'gpu_name': properties.name,
        'gpu_arch': getattr(properties, 'gcnArchName', None),
        'gpu_properties': str(properties),
        'device_index': 0,
        'initial_memory': memory(torch),
    })
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # 上游常用 [batch, sequence, heads, head_dim] -> transpose(1, 2)。
    shape = (1, args.sequence_length, args.heads, args.head_dim)
    q, k, v = [torch.randn(shape, dtype=torch.bfloat16, device='cuda:0').transpose(1, 2)
               for _ in range(3)]
    torch.cuda.synchronize()
    report['inputs'] = {
        'shape': list(q.shape), 'stride': list(q.stride()), 'contiguous': q.is_contiguous(),
        'dtype': str(q.dtype), 'seed': 0, 'qkv_bytes': 3 * q.numel() * q.element_size(),
        'text_tokens_effective': min(args.text_tokens, args.sequence_length),
        'memory_after_inputs': memory(torch),
    }
    save(args.output, report)
    cases = ('no_mask', 'is_causal', 'broadcast_key_padding',
             'qwen_dense_causal_text_keys', 'diffusion_dense_all_true')
    backends = (('MATH', SDPBackend.MATH), ('FLASH_ATTENTION', SDPBackend.FLASH_ATTENTION),
                ('EFFICIENT_ATTENTION', SDPBackend.EFFICIENT_ATTENTION))
    for case in cases:
        mask = make_mask(torch, case, args.sequence_length,
                         min(args.text_tokens, args.sequence_length))
        reference = None
        for backend_name, backend in backends:
            result = {
                'case': case, 'backend': backend_name, 'status': 'running',
                'is_causal': case == 'is_causal', 'dropout_p': 0.0, 'backward': False,
                'mask_shape': list(mask.shape) if mask is not None else None,
                'mask_stride': list(mask.stride()) if mask is not None else None,
                'mask_dtype': str(mask.dtype) if mask is not None else None,
                'mask_bytes': mask.numel() * mask.element_size() if mask is not None else 0,
            }
            report['results'].append(result)
            if backend_name == 'MATH' and args.sequence_length > 512:
                result.update(status='skipped', reason='sequence_length > 512，禁止 MATH 二次方分配')
                save(args.output, report)
                print(f'{case} / {backend_name}: skipped (large sequence)', flush=True)
                continue
            output = actual = None
            warning_records = []
            started = None
            baseline = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                baseline = memory(torch)
                result['memory_before'] = baseline
                torch.cuda.reset_peak_memory_stats()
                save(args.output, report)
                print(f'{case} / {backend_name}: running', flush=True)
                with warnings.catch_warnings(record=True) as warning_records:
                    warnings.simplefilter('always')
                    with torch.inference_mode(), sdpa_kernel(backends=[backend]):
                        started = time.perf_counter()
                        output = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=mask, dropout_p=0.0,
                            is_causal=case == 'is_causal')
                        torch.cuda.synchronize()
                        result['synchronized_elapsed_seconds'] = time.perf_counter() - started
                result.update(status='success', memory_after=memory(torch))
                result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                result['incremental_peak_allocated_bytes'] = max(
                    0, result['peak_allocated_bytes'] - baseline['allocated_bytes'])
                result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                if args.sequence_length <= 512:
                    actual = output.detach().float().cpu()
                    if backend_name == 'MATH':
                        reference = actual
                        result['correctness'] = {'reference': 'MATH', 'baseline': True,
                                                 'finite': bool(torch.isfinite(actual).all())}
                    elif reference is not None:
                        difference = (actual - reference).abs()
                        result['correctness'] = {
                            'reference': 'MATH', 'rtol': 0.02, 'atol': 0.02,
                            'allclose': bool(torch.allclose(actual, reference, rtol=0.02, atol=0.02)),
                            'finite': bool(torch.isfinite(actual).all()),
                            'max_absolute_error': difference.max().item(),
                            'mean_absolute_error': difference.mean().item(),
                        }
                    else:
                        result['correctness'] = {'status': 'unavailable', 'reason': 'MATH baseline failed'}
                else:
                    result['correctness'] = {'status': 'skipped', 'reason': 'sequence_length > 512'}
            except Exception as exc:
                result.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                              traceback=traceback.format_exc())
                if started is not None:
                    result['failed_wall_seconds'] = time.perf_counter() - started
                try:
                    result['memory_after_error'] = memory(torch)
                    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    if baseline is not None:
                        result['incremental_peak_allocated_bytes'] = max(
                            0, result['peak_allocated_bytes'] - baseline['allocated_bytes'])
                except Exception as memory_exc:
                    result['memory_read_error'] = str(memory_exc)
            finally:
                result['warnings'] = [str(item.message) for item in warning_records]
                save(args.output, report)
                print(f"{case} / {backend_name}: {result['status']} "
                      f"{result.get('synchronized_elapsed_seconds', result.get('error', ''))}", flush=True)
                del output, actual
        del mask, reference
    report['status'] = 'complete'
    report['final_memory'] = memory(torch)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence-length', type=positive_int, default=256)
    parser.add_argument('--heads', type=positive_int, default=32)
    parser.add_argument('--head-dim', type=positive_int, default=128)
    parser.add_argument('--text-tokens', type=positive_int, default=849)
    args = parser.parse_args()
    report = {
        'schema_version': 1, 'status': 'running',
        'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'script': {'path': str(Path(__file__).resolve()),
                   'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'configuration': {**vars(args), 'output': str(args.output), 'math_max_sequence_length': 512},
        'environment': {'python': sys.version, 'executable': sys.executable,
                        'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL':
                            os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'),
                        'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES'),
                        'ROCR_VISIBLE_DEVICES': os.environ.get('ROCR_VISIBLE_DEVICES'),
                        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')},
        'notes': ['只测试 BF16 forward；每个 case 每个 backend 执行一次，含冷启动开销。',
                  '单次强制 backend，未启用自动 MATH 回退。',
                  'timing 不含 Q/K/V 和 mask 构造、CPU 正确性比较；增量峰值以调用前 allocated 为基准。',
                  'dense bool mask 保留 sequence_length 行跨度；True 代表允许 attention。',
                  '单个 backend 不支持会记录 failed 并继续；complete 不表示所有 backend 都支持。'],
        'results': [],
    }
    save(args.output, report)
    exit_code = 0
    try:
        exit_code = run(args, report)
    except KeyboardInterrupt:
        report.update(status='interrupted', error='KeyboardInterrupt')
        exit_code = 130
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        exit_code = 1
    finally:
        report['finished_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(args.output, report)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
