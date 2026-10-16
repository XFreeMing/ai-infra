"""Read-only launch guard for the observed AI 300 Ideogram4 GPU hangs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from lab_tools.cli import WorkflowError

SMALL_IMAGE_PIXELS = 512 * 512
KNOWN_ISSUE = (
    'AI 300 的 Ideogram4 原生 1344×2016 生成已出现 GPU hang/reset 和桌面退出；'
    '底层稳定性尚未修复，需要隔离验证。'
)


def image_dimension(value: str) -> int:
    pixels = int(value)
    if not 256 <= pixels <= 2048 or pixels % 16:
        raise argparse.ArgumentTypeError('图片宽高必须为 256–2048 之间的 16 的倍数。')
    return pixels


def kfd_processes(kfd_root: Path = Path('/sys/class/kfd/kfd/proc'),
                  proc_root: Path = Path('/proc')) -> dict:
    """Snapshot other KFD clients, not GPU utilization or an atomic idle guarantee."""
    result = {'status': 'unknown', 'path': str(kfd_root), 'processes': [], 'errors': [],
              'observed_at': datetime.now(timezone.utc).isoformat()}
    try:
        entries = list(kfd_root.iterdir())
    except OSError as error:
        result['errors'].append(str(error))
        return result
    result['status'] = 'observed'
    for entry in sorted(entries, key=lambda item: item.name):
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        process = {'pid': int(entry.name), 'comm': None, 'comm_status': 'unknown'}
        try:
            process['comm'] = (proc_root / entry.name / 'comm').read_text(encoding='utf-8').strip()
            process['comm_status'] = 'observed'
        except OSError as error:
            result['errors'].append(f'PID {entry.name}: {error}')
        result['processes'].append(process)
    return result


def assess_gpu_launch(app: str, *, experimental: bool, webui: bool = False,
                      width: int = 512, height: int = 512) -> dict:
    record = {'applicable': app == 'ideogram4', 'experimental_gpu': experimental,
              'status': 'not_applicable', 'known_issue': None, 'reasons': [], 'warnings': [],
              'kfd': None}
    if app != 'ideogram4':
        return record
    record.update(known_issue=KNOWN_ISSUE, mode='webui' if webui else 'single',
                  width=width, height=height, small_image_pixel_limit=SMALL_IMAGE_PIXELS)
    if webui:
        record['reasons'].append('WebUI 无法在启动时约束后续生成尺寸，暂时禁止直接启动。')
    elif width * height > SMALL_IMAGE_PIXELS:
        record['reasons'].append(f'{width}×{height} 超出临时小尺寸试验范围（面积不超过 512×512）。')
    else:
        record['warnings'].append('小尺寸仅降低负载；当前驱动和 ROCm 组合尚未验证稳定，仍可能发生 GPU 故障。')
    record['kfd'] = kfd_processes()
    if record['kfd']['processes']:
        clients = ', '.join(f"PID {item['pid']} ({item['comm'] or '进程名 unknown'})"
                            for item in record['kfd']['processes'])
        record['reasons'].append(f'检测到其他 KFD GPU 客户端：{clients}；需要先确认并结束并发计算。')
    if record['kfd']['status'] == 'unknown':
        record['warnings'].append('无法读取 KFD 进程列表，GPU 并发状态为 unknown，不能确认 GPU 空闲。')
    if record['kfd']['errors'] and record['kfd']['status'] != 'unknown':
        record['warnings'].append('部分 KFD 进程信息读取失败，已记录 unknown 和原始错误。')
    record['status'] = ('experimental_override' if experimental else 'blocked') if record['reasons'] else 'allowed'
    return record


def require_gpu_launch(record: dict) -> None:
    if not record['applicable']:
        return
    if record['status'] == 'blocked':
        raise WorkflowError(record['known_issue'] + '\n' + '\n'.join(record['reasons']) +
                            '\n保持原参数未启动 GPU。仅隔离实验可显式使用 --experimental-gpu 越过保护；'
                            '该选项不能修复 GPU 故障。')
    if record['status'] == 'experimental_override':
        print('实验性 GPU 重试：' + record['known_issue'] + '\n' + '\n'.join(record['reasons']), flush=True)
    for warning in record['warnings']:
        print('GPU 检查：' + warning, flush=True)
