"""Bounded, local-only image/audio inference smoke tests on AI 300."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import time

from lab_tools.cli import WorkflowError, _git
from lab_tools.media_launcher import resolve_runtime
from lab_tools.media_models import APPS, app_name, model_dir, verify_local
from lab_tools.native_prepare import initialization_lock
from lab_tools.prepare import Recorder, positive_int, sha256

REFERENCE = Path(__file__).with_name('fixtures') / 'indextts2-voice01.wav'
REFERENCE_SHA256 = 'e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8'
TEXT = '你好，这是语音合成测试。'
CAPTION = {
    'high_level_description': 'A red boat floating on a quiet blue lake.',
    'style_description': {'aesthetics': 'calm and minimal', 'lighting': 'soft morning light',
                          'medium': 'illustration', 'art_style': 'flat vector illustration',
                          'color_palette': ['#E53935', '#1565C0']},
    'compositional_deconstruction': {'background': 'A calm blue lake.', 'elements': []},
}


def plan_hello(root: Path, app: str, args, directory: Path) -> dict:
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
        command += ['--height', '256', '--width', '256', '--seed', '0', '--sampler', 'V4_TURBO_12']
    else:
        command += ['--reference', str(directory / 'input/reference.wav')]
    env = {**(runtime or {}).get('environment', {}), 'HIP_VISIBLE_DEVICES': args.devices,
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'GRADIO_ANALYTICS_ENABLED': 'False',
           'PYTHONNOUSERSITE': '1', 'PYTHONPATH': '', 'AI_INFRA_MEDIA_MODEL_DIR': str(models)}
    return {'app': app, 'machine_id': '300', 'argv': command, 'python': python, 'environment': env,
            'runtime_blocker': unresolved, 'preparation_manifest': str(path) if path else None,
            'model_dir': str(models), 'reference': str(reference) if app == 'indextts2' else None,
            'output_dir': str(output), 'timeout_seconds': args.timeout,
            'model_download_requested': False, 'dependency_install_requested': False,
            'configuration': {'caption': CAPTION, 'height': 256, 'width': 256, 'sampler': 'V4_TURBO_12', 'seed': 0}
                             if app == 'ideogram4' else {'text': TEXT}}


def run_hello(root: Path, app: str, args) -> int:
    from lab_tools.machines import require_host
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
        seconds = args.timeout - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError('hello 总时间预算已耗尽。')
        return seconds

    try:
        require_host('300')
        plan = plan_hello(root, app, args, directory)
        manifest.update(plan)
        recorder.save()
        print(f'{APPS[app]["name"]} hello：实际生成一次；预算 {args.timeout} 秒。\n验证目录：{directory}', flush=True)
        with initialization_lock(root, remaining()):
            manifest['preparation_manifest_sha256'] = sha256(Path(plan['preparation_manifest']))
            manifest['models'] = verify_local(app, Path(plan['model_dir']))
            inputs = directory / 'input'
            inputs.mkdir()
            (inputs / 'configuration.json').write_text(json.dumps(plan['configuration'], ensure_ascii=False, indent=2) + '\n')
            checker = directory / 'check_output.py'
            shutil.copyfile(Path(__file__).with_name('media_hello_check.py'), checker)
            manifest['checker_sha256'] = sha256(checker)
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
            recorder.save()
            recorder.require('generate', plan['argv'], root, remaining(), env)
            recorder.require('validate-output', [plan['python'], '-I', str(checker), '--app', app,
                             '--output', plan['output_dir'], '--result', str(directory / 'result.json')],
                             root, remaining(), env)
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
        interrupted = last.get('status') == 'interrupted' or last.get('exit_code') in (130, -2)
        timed_out = (isinstance(error, TimeoutError) or last.get('status') == 'timed_out' or
                     time.monotonic() - started >= args.timeout)
        manifest.update(status='interrupted' if interrupted else 'timed_out' if timed_out else 'failed', error=str(error))
        print(f'{APPS[app]["name"]} hello 未通过：{error}', flush=True)
        return 130 if interrupted else 124 if timed_out else 2
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        manifest['duration_seconds'] = round(time.monotonic() - started, 3)
        recorder.save()
        print(f'验证记录：{directory / "manifest.json"}', flush=True)


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    parser = argparse.ArgumentParser(prog='300 hello', description='加载本地模型，生成并检查一张图片或一句配音；不安装依赖或下载模型。')
    parser.add_argument('targets', nargs='+', type=app_name, metavar='Ideogram4|Indextts2')
    parser.add_argument('--timeout', type=positive_int, default=900, help='每个应用的总预算秒数，默认 900。')
    parser.add_argument('--reference', type=Path, help='IndexTTS2 参考人声 WAV，默认使用内置上游演示样本。')
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--preparation-dir', type=Path)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/hello/300'))
    parser.add_argument('--dry-run', action='store_true', help='显示输入和命令，不运行模型或创建输出。')
    args = parser.parse_args(argv)
    if len(args.targets) != 1 and (args.model_dir or args.preparation_dir):
        raise WorkflowError('--model-dir / --preparation-dir 只能用于单个应用。')
    if args.reference and 'indextts2' not in args.targets:
        raise WorkflowError('--reference 只用于 Indextts2。')
    result = 0
    for app in args.targets:
        code = run_hello(root, app, args)
        if code == 130:
            return code
        result = max(result, code)
    return result
