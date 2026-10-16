"""Combine observed hello resource samples and synchronized runtime timings."""
from __future__ import annotations

import json
import math
from pathlib import Path


def number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return value
    return None


def ratio(numerator, denominator):
    a, b = number(numerator), number(denominator)
    return a / b if a is not None and b is not None and b > 0 else None


def summarize(manifest: dict, runtime: dict, resources: dict) -> dict:
    phases = runtime.get('phases', {})
    timings = {}
    for name in ('runtime_check', 'model_load', 'generation'):
        phase = phases.get(name, {})
        if not isinstance(phase, dict):
            phase = {}
        synchronized = (phase.get('status') == 'succeeded' and phase.get('duration_complete') is True and
                        phase.get('gpu_synchronized_at_start') is True and phase.get('gpu_synchronized_at_end') is True)
        timings[name + '_seconds'] = number(phase.get('duration_seconds')) if synchronized else None
    command = next((item for item in manifest.get('commands', []) if item['label'] == 'generate'), {})
    not_started = manifest.get('status') == 'blocked' and not command and not runtime and not resources
    timings['worker_wall_seconds'] = number(command.get('duration_seconds'))
    timings['hello_wall_seconds'] = number(manifest.get('duration_seconds'))
    completed = manifest.get('status') == 'passed'
    result = manifest.get('result', {}) if completed else {}
    seconds = timings['generation_seconds']
    metrics = {}
    if manifest['app'] == 'ideogram4':
        count = 1 if result.get('passed') else 0
        width, height = number(result.get('width')), number(result.get('height'))
        pixels = width * height if width is not None and height is not None else None
        metrics = {'image_count': count, 'width': width, 'height': height,
                   'sampler': manifest.get('configuration', {}).get('sampler'),
                   'generation_seconds_per_image': seconds if count else None,
                   'images_per_second': ratio(count, seconds) if count else None,
                   'megapixels_per_second': ratio(pixels / 1e6 if pixels else None, seconds) if count else None,
                   'end_to_end_seconds_per_image': timings['hello_wall_seconds'] if count else None}
    else:
        duration = number(result.get('duration_seconds'))
        text = manifest.get('configuration', {}).get('text', '')
        metrics = {'audio_duration_seconds': duration, 'sample_rate': result.get('sample_rate'),
                   'channels': result.get('channels'), 'text_characters': len(text),
                   'real_time_factor': ratio(seconds, duration),
                   'audio_seconds_per_compute_second': ratio(duration, seconds),
                   'characters_per_second': ratio(len(text), seconds) if result else None,
                   'end_to_end_real_time_factor': ratio(timings['hello_wall_seconds'], duration)}
    return {'schema_version': 1, 'app': manifest['app'], 'run_id': manifest['run_id'],
            'run_status': manifest['status'],
            'measurement_status': ('not_started' if not_started else 'complete' if all(timings.get(key) is not None for key in
                ('model_load_seconds', 'generation_seconds')) and resources.get('sample_count', 0) else 'partial'),
            'timings': timings, 'generation': metrics, 'runtime': runtime, 'resources': resources,
            'definitions': {
                'resource_statistics': '有效定时采样的 min/max/mean；mean 为算术平均，不是连续监控的精确峰值。',
                'cpu_percent': 'process_cpu_percent: 100% = 一个逻辑 CPU；host_cpu_percent: 整机 0–100%。',
                'generation_seconds': 'GPU synchronize 后的墙钟时间；图片包含文本编码/去噪/解码，音频包含 infer 与 WAV 保存。',
                'worker_wall_seconds': '生成子进程完整时间，包含运行时检查、import、加载模型、生成和退出。',
                'hello_wall_seconds': 'hello 完整墙钟时间，包含前置校验、等待锁、参考音频检查和输出检查。',
                'real_time_factor': 'generation_seconds / audio_duration_seconds；越小越快，<1 表示快于实时。',
                'audio_seconds_per_compute_second': 'audio_duration_seconds / generation_seconds；越大越快。',
                'memory': 'AI 300 为统一内存；RSS、VRAM/GTT、PyTorch allocated/reserved 是不同口径，不能相加。',
                'gpu_scope': 'sysfs GPU 利用率/内存为整卡数据，SoC 功耗包含 APU CPU；其他进程可能影响读数。',
                'benchmark_scope': '单次小样本的观测；没有多轮延迟分布，不能推导 p95/p99 或稳定吞吐。'},
            'artifacts': {'runtime': None if not_started else 'runtime-performance.json',
                          'resources': None if not_started else 'resources.json',
                          'resource_samples': None if not_started else 'resources.jsonl'}}


def write_performance(directory: Path, manifest: dict) -> dict:
    errors = []
    runtime = {}
    path = directory / 'runtime-performance.json'
    if path.is_file():
        try:
            runtime = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(runtime, dict) or not isinstance(runtime.get('phases', {}), dict):
                raise ValueError('运行时指标结构不正确')
        except (OSError, ValueError) as error:
            errors.append(str(error))
            runtime = {}
    resources = next((item.get('resource_summary', {}) for item in manifest.get('commands', [])
                      if item['label'] == 'generate'), {})
    report = summarize(manifest, runtime, resources)
    report['errors'] = errors
    path = directory / 'performance.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if report['measurement_status'] == 'not_started':
        print(f'本次未启动生成进程，未加载模型或采集生成性能数据。\n状态 JSON：{path}', flush=True)
        return report
    print('\n性能汇总（本次实际观测）：', flush=True)
    labels = {'hello_wall_seconds': '总耗时', 'worker_wall_seconds': '生成进程',
              'model_load_seconds': '加载模型', 'generation_seconds': '实际生成'}
    print('｜'.join(f'{label}：{report["timings"][key]:.2f} s' if report['timings'][key] is not None
                   else f'{label}：未完成/不可用' for key, label in labels.items()), flush=True)
    metrics = report['generation']
    if manifest['app'] == 'ideogram4' and metrics.get('images_per_second') is not None:
        print(f'生图：{metrics["image_count"]} 张｜{metrics["width"]}×{metrics["height"]}｜'
              f'{metrics["generation_seconds_per_image"]:.3f} s/张｜{metrics["images_per_second"]:.4f} 张/s', flush=True)
    elif metrics.get('real_time_factor') is not None:
        print(f'配音：{metrics["audio_duration_seconds"]:.3f} s 音频｜RTF {metrics["real_time_factor"]:.3f}｜'
              f'{metrics["audio_seconds_per_compute_second"]:.3f} 倍实时速度', flush=True)
    for phase_name, label in (('model_load', '加载阶段'), ('generation', '生成阶段')):
        phase = runtime.get('phases', {}).get(phase_name, {})
        if not isinstance(phase, dict):
            continue
        allocated, reserved = number(phase.get('torch_peak_allocated_bytes')), number(phase.get('torch_peak_reserved_bytes'))
        if allocated is not None and reserved is not None:
            print(f'{label} PyTorch 峰值：allocated {allocated / 2**30:.2f} GiB｜reserved {reserved / 2**30:.2f} GiB', flush=True)
    if report['measurement_status'] != 'complete':
        print('性能记录不完整；已保留可用数据，未完成的阶段不会计算吞吐率。', flush=True)
    print(f'性能 JSON：{path}', flush=True)
    return report
