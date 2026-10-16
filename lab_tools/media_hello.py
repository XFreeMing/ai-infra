"""Bounded, local-only poster/audio inference checks on AI 300."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from lab_tools.cli import WorkflowError, _git
from lab_tools.media_gpu_guard import assess_gpu_launch, image_dimension, require_gpu_launch
from lab_tools.media_launcher import resolve_runtime
from lab_tools.media_models import APPS, app_name, model_dir, runtime_environment, verify_local
from lab_tools.media_performance import write_performance
from lab_tools.media_resources import ResourceMonitor
from lab_tools.native_prepare import initialization_lock
from lab_tools.prepare import Recorder, positive_int, sha256

REFERENCE = Path(__file__).with_name('fixtures') / 'indextts2-voice01.wav'
REFERENCE_SHA256 = 'e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8'
TEXT = '你好，这是语音合成测试。'
IMAGE_WIDTH = 1344
IMAGE_HEIGHT = 2016
DEFAULT_TIMEOUTS = {'ideogram4': 3600, 'indextts2': 900}
IMAGE_PROMPT = (
    '莫奈风格，梵高作品，伦勃朗画作，金箔岩彩画，完美融入摄影技法+油画技法+3D技法，绿白色氛围感高级艺术海报。'
    '极简主义、抽象表达、现代设计、高级感、柔和渐变。画面采用暖色调（不同色阶的蓝绿色为主），绿蓝色的朦胧质感，'
    '隐约看到蝴蝶形态。动感模糊，颜色饱满，朦胧美学。整体风格文艺雅致，富有香氛氛围感，突出诗意与意韵。'
    '细闪，细致笔触，极繁主义，超多细节，极繁风格，Octane 渲染，64K高清画质，'
    '用流动的毛笔手绘字体：“Don’t wait for inspiration”（纤细的笔画，笔画拉丝，灵动的曲线，有质感），'
    '极小的装饰性小字，极细无衬线小字体“202X”，白色极细无衬线字体水印" LUOBOS "+“ INSPIRATION ”+'
    '" AIGC "+"Every great idea begins with a spark"，质感，肌理感，梦幻的，耀变，反射，艺术抽象，诗意的，'
    '高级，杰作，极具视觉冲击力，获奖作品，大师级配色，大师级排版，高细节，高级感，超高清，布局协调。'
)
CAPTION = {
    'high_level_description': IMAGE_PROMPT,
    'style_description': {
        'aesthetics': '诗意、梦幻、文艺雅致；极简布局与极繁肌理并存',
        'lighting': '柔和渐变，暖光、细闪和金箔反射',
        'medium': '摄影、油画、3D 与金箔岩彩混合媒介海报',
        'art_style': '莫奈、梵高、伦勃朗风格融合，Octane 渲染质感',
        'color_palette': ['#176F73', '#4D9B8A', '#A8D5C2', '#FFFFFF', '#CBA45B'],
    },
    'compositional_deconstruction': {
        'background': '蓝绿与白色朦胧渐变，隐约的蝴蝶形态、动感模糊、细致笔触和金箔岩彩肌理。',
        'elements': [
            {'type': 'text', 'bbox': [360, 80, 600, 920], 'text': 'Don’t wait for inspiration',
             'desc': '流动毛笔手绘字，纤细拉丝笔画、灵动曲线与真实肌理', 'color_palette': ['#FFFFFF']},
            {'type': 'text', 'bbox': [80, 80, 115, 250], 'text': '202X',
             'desc': '极小装饰文字，极细无衬线字体', 'color_palette': ['#FFFFFF']},
            {'type': 'text', 'bbox': [850, 80, 885, 310], 'text': 'LUOBOS',
             'desc': '白色极细无衬线水印', 'color_palette': ['#FFFFFF']},
            {'type': 'text', 'bbox': [850, 350, 885, 750], 'text': 'INSPIRATION',
             'desc': '白色极细无衬线水印', 'color_palette': ['#FFFFFF']},
            {'type': 'text', 'bbox': [850, 810, 885, 920], 'text': 'AIGC',
             'desc': '白色极细无衬线水印', 'color_palette': ['#FFFFFF']},
            {'type': 'text', 'bbox': [915, 100, 945, 900], 'text': 'Every great idea begins with a spark',
             'desc': '白色极细无衬线小字', 'color_palette': ['#FFFFFF']},
        ],
    },
}


def sample_interval(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or not 0.2 <= seconds <= 60:
        raise argparse.ArgumentTypeError('采样间隔必须在 0.2–60 秒之间。')
    return seconds


def timeout_budget(app: str, override: int | None) -> int:
    return DEFAULT_TIMEOUTS[app] if override is None else override


def plan_hello(root: Path, app: str, args, directory: Path) -> dict:
    width = args.width if args.width is not None else IMAGE_WIDTH
    height = args.height if args.height is not None else IMAGE_HEIGHT
    unresolved = None
    runtime, path = None, None
    try:
        runtime, path = resolve_runtime(root, app, args.preparation_dir)
    except WorkflowError as error:
        if not args.dry_run or args.preparation_dir or not str(error).startswith('尚未初始化'):
            raise
        unresolved = str(error)
    placeholder = root / f'artifacts/init/300/<prepared-{app}>'
    python = runtime['python'] if runtime else str(placeholder / 'venv/bin/python')
    worker = runtime['worker'] if runtime else str(placeholder / 'runtime/lab_tools/media_worker.py')
    source = runtime['source'] if runtime else str(placeholder / 'source')
    models = (root / (args.model_dir or model_dir(app, '300', args.models_dir))).resolve()
    output = directory / 'output'
    command = [python, '-I', worker, 'run', '--app', app, '--source', source,
               '--model-dir', str(models), '--output', str(output),
               '--prompt', json.dumps(CAPTION, ensure_ascii=False) if app == 'ideogram4' else TEXT]
    if path:
        command += ['--expected-provenance', str(path.parent / 'provenance.json')]
    reference = (root / args.reference).resolve() if args.reference else REFERENCE
    if app == 'ideogram4':
        command += ['--height', str(height), '--width', str(width), '--seed', '0', '--sampler', 'V4_TURBO_12']
    else:
        command += ['--reference', str(directory / 'input/reference.wav')]
    worker_argv = command
    wrapper = directory / 'media_hello_runtime.py'
    decode_device = (args.decode_device or 'cpu') if app == 'ideogram4' else 'gpu'
    command = [python, '-I', str(wrapper), '--worker', worker,
               '--metrics', str(directory / 'runtime-performance.json'),
               '--decode-device', decode_device, '--', *worker_argv[3:]]
    env = {**(runtime or {}).get('environment', {}), 'HIP_VISIBLE_DEVICES': args.devices,
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'GRADIO_ANALYTICS_ENABLED': 'False',
           'PYTHONNOUSERSITE': '1', 'PYTHONPATH': '', 'AI_INFRA_MEDIA_MODEL_DIR': str(models),
           **runtime_environment(app)}
    return {'app': app, 'machine_id': '300', 'argv': command, 'worker_argv': worker_argv,
            'python': python, 'environment': env, 'sample_interval_seconds': args.sample_interval,
            'runtime_blocker': unresolved, 'preparation_manifest': str(path) if path else None,
            'model_dir': str(models), 'reference': str(reference) if app == 'indextts2' else None,
            'output_dir': str(output), 'timeout_seconds': timeout_budget(app, args.timeout),
            'model_download_requested': False, 'dependency_install_requested': False,
            'gpu_guard': assess_gpu_launch(app, experimental=args.experimental_gpu, width=width, height=height),
            'configuration': {'caption': CAPTION, 'height': height, 'width': width,
                              'sampler': 'V4_TURBO_12', 'seed': 0, 'decode_device': decode_device}
                             if app == 'ideogram4' else {'text': TEXT}}


def run_hello(root: Path, app: str, args) -> int:
    from lab_tools.machines import require_host
    timeout = timeout_budget(app, args.timeout)
    base = (root / args.output_dir).resolve()
    if args.dry_run:
        print(json.dumps(plan_hello(root, app, args, base / f'<run-id>-{app}'), ensure_ascii=False, indent=2))
        return 0
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + app + '-', dir=base))
    started = time.monotonic()
    manifest = {'schema_version': 1, 'experiment_id': 'media-hello', 'run_id': directory.name,
                'machine_id': '300', 'app': app, 'status': 'preparing', 'commands': [],
                'started_at': datetime.now(timezone.utc).isoformat(), 'parent_commit': _git(root, 'rev-parse', 'HEAD'),
                'parent_dirty': bool(_git(root, 'status', '--porcelain')), 'gpu_inference_validated': False,
                'model_download_requested': False, 'dependency_install_requested': False}
    recorder = Recorder(root, directory, manifest)
    recorder.save()

    def remaining():
        seconds = timeout - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError('hello 总时间预算已耗尽。')
        return seconds

    try:
        require_host('300')
        plan = plan_hello(root, app, args, directory)
        manifest.update(plan)
        recorder.save()
        # Reject known high-risk defaults before even reading model weights.
        if plan['gpu_guard']['status'] == 'blocked':
            require_gpu_launch(plan['gpu_guard'])
        print(f'{APPS[app]["name"]} hello：实际生成一次；预算 {timeout} 秒。\n验证目录：{directory}', flush=True)
        with initialization_lock(root, remaining()):
            manifest['preparation_manifest_sha256'] = sha256(Path(plan['preparation_manifest']))
            manifest['models'] = verify_local(app, Path(plan['model_dir']))
            inputs = directory / 'input'
            inputs.mkdir()
            (inputs / 'configuration.json').write_text(json.dumps(plan['configuration'], ensure_ascii=False, indent=2) + '\n')
            checker = directory / 'check_output.py'
            shutil.copyfile(Path(__file__).with_name('media_hello_check.py'), checker)
            manifest['checker_sha256'] = sha256(checker)
            wrapper = directory / 'media_hello_runtime.py'
            shutil.copyfile(Path(__file__).with_name('media_hello_runtime.py'), wrapper)
            manifest['runtime_observer_sha256'] = sha256(wrapper)
            manifest['measurement_sources'] = {}
            for name in ('media_resources.py', 'media_performance.py'):
                snapshot = directory / name
                shutil.copyfile(Path(__file__).with_name(name), snapshot)
                manifest['measurement_sources'][name] = sha256(snapshot)
            env = {**plan['environment'], 'AI_INFRA_LAUNCH_MANIFEST': str(directory / 'manifest.json')}
            if app == 'indextts2':
                reference = Path(plan['reference'])
                if not reference.is_file():
                    raise WorkflowError('参考音频不存在，请用 --reference 指定本地 WAV。')
                digest = sha256(reference)
                if args.reference is None and digest != REFERENCE_SHA256:
                    raise WorkflowError('内置参考音频 SHA256 不符。')
                shutil.copyfile(reference, inputs / 'reference.wav')
                recorder.require('validate-reference', [plan['python'], '-I', str(checker), '--app', app,
                    '--reference', str(inputs / 'reference.wav'), '--result', str(directory / 'reference.json')],
                    root, remaining(), env)
                manifest['reference_audio'] = json.loads((directory / 'reference.json').read_text())
            manifest['status'] = 'running'
            manifest['gpu_guard'] = assess_gpu_launch(app, experimental=args.experimental_gpu,
                width=plan['configuration'].get('width', 512), height=plan['configuration'].get('height', 512))
            recorder.save()
            require_gpu_launch(manifest['gpu_guard'])
            monitor = ResourceMonitor(directory, interval=args.sample_interval, devices=args.devices)
            recorder.require('generate', plan['argv'], root, remaining(), env, observer=monitor)
            validation_command = [plan['python'], '-I', str(checker), '--app', app,
                                  '--output', plan['output_dir'], '--result', str(directory / 'result.json')]
            if app == 'ideogram4':
                validation_command += ['--expected-width', str(plan['configuration']['width']),
                                       '--expected-height', str(plan['configuration']['height'])]
            recorder.require('validate-output', validation_command, root, remaining(), env)
            result = json.loads((directory / 'result.json').read_text())
            if (not isinstance(result, dict) or result.get('passed') is not True or
                    result.get('app') != app or not isinstance(result.get('output_path'), str)):
                raise WorkflowError('输出校验未通过。')
            artifact = Path(result['output_path']).resolve()
            if (not artifact.is_relative_to(Path(plan['output_dir']).resolve()) or
                    not artifact.is_file() or not artifact.stat().st_size or
                    sha256(artifact) != result.get('sha256')):
                raise WorkflowError('输出文件与校验记录不符。')
            manifest.update(status='passed', result=result, gpu_inference_validated=True)
            print(f'\n{APPS[app]["name"]} hello 通过。\n输出：{result["output_path"]}', flush=True)
            return 0
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
        print('hello 已中断，日志和已生成文件保留。', flush=True)
        return 130
    except (WorkflowError, OSError, ValueError) as error:
        last = manifest['commands'][-1] if manifest['commands'] else {}
        interrupted = last.get('status') == 'interrupted' or last.get('exit_code') in (130, -2, 143, -15)
        timed_out = (isinstance(error, TimeoutError) or last.get('status') == 'timed_out' or
                     time.monotonic() - started >= timeout)
        blocked = manifest.get('gpu_guard', {}).get('status') == 'blocked'
        manifest.update(status='interrupted' if interrupted else 'timed_out' if timed_out else 'blocked' if blocked else 'failed', error=str(error))
        message = '启动已拦截（未启动生成进程）' if manifest['status'] == 'blocked' else 'hello 未通过'
        print(f'{APPS[app]["name"]} {message}：{error}', flush=True)
        if manifest['status'] == 'blocked' and app == 'ideogram4':
            print('CPU VAE 解码仍保留 GPU 编码和采样；--timeout 只调整预算，不改变尺寸或解除保护。\n'
                  '先确认没有其他 GPU 计算任务，再进行小尺寸隔离验证（仍未保证驱动稳定）：\n'
                  'uv run 300 hello Ideogram4 --width 384 --height 576 --decode-device cpu --timeout 300', flush=True)
        return 130 if interrupted else 124 if timed_out else 2
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        manifest['duration_seconds'] = round(time.monotonic() - started, 3)
        report = write_performance(directory, manifest)
        manifest['performance'] = {'path': str(directory / 'performance.json'),
                                   'sha256': sha256(directory / 'performance.json'),
                                   'measurement_status': report['measurement_status']}
        recorder.save()
        print(f'验证记录：{directory / "manifest.json"}', flush=True)


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    parser = argparse.ArgumentParser(prog='300 hello', description='加载本地模型，生成并检查一张图片或一句配音；不安装依赖或下载模型。')
    parser.add_argument('targets', nargs='+', type=app_name, metavar='Ideogram4|Indextts2')
    parser.add_argument('--timeout', type=positive_int,
                        help='每个应用各自的总预算秒数；默认 Ideogram4 3600、Indextts2 900，显式指定时覆盖所有目标。')
    parser.add_argument('--sample-interval', type=sample_interval, default=2.0,
                        help='生成进程资源采样和控制台输出间隔，默认 2 秒，范围 0.2–60。')
    parser.add_argument('--reference', type=Path, help='IndexTTS2 参考人声 WAV，默认使用内置上游演示样本。')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--preparation-dir', type=Path)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--width', type=image_dimension, help='Ideogram4 单次生成宽度；默认 1344，范围 256–2048，16 的倍数。')
    parser.add_argument('--height', type=image_dimension, help='Ideogram4 单次生成高度；默认 2016，范围 256–2048，16 的倍数。')
    parser.add_argument('--decode-device', choices=('cpu', 'gpu'),
                        help='Ideogram4 hello 的 VAE 解码设备，默认 cpu；避开 GPU 卷积的实验旁路，尚未验证整条链路稳定。')
    parser.add_argument('--experimental-gpu', action='store_true',
                        help='仅隔离实验：显式越过 Ideogram4 已知 GPU 故障/并发保护，不能修复故障。')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/hello/300'))
    parser.add_argument('--dry-run', action='store_true', help='显示输入和命令，不运行模型或创建输出。')
    args = parser.parse_args(argv)
    if len(args.targets) != 1 and (args.model_dir or args.preparation_dir):
        raise WorkflowError('--model-dir / --preparation-dir 只能用于单个应用。')
    if args.reference and 'indextts2' not in args.targets:
        raise WorkflowError('--reference 只用于 Indextts2。')
    if (args.width is not None or args.height is not None) and args.targets != ['ideogram4']:
        raise WorkflowError('--width / --height 只用于单个 Ideogram4 应用，不能用于 Indextts2。')
    if args.decode_device is not None and args.targets != ['ideogram4']:
        raise WorkflowError('--decode-device 只用于单个 Ideogram4 应用，不能用于 Indextts2。')
    result = 0
    for app in args.targets:
        code = run_hello(root, app, args)
        if code == 130:
            return code
        result = max(result, code)
    return result
