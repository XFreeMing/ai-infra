"""Launch prepared media runtimes with verified local models, never downloading assets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile

from lab_tools.cli import WorkflowError, _git
from lab_tools.media_models import APPS, TORCH_VERSION, files_for, model_dir, verify_local
from lab_tools.machines import require_host
from lab_tools.prepare import positive_int, sha256


def resolve_runtime(root: Path, app: str, explicit: Path | None) -> tuple[dict, Path]:
    candidates = [(root / explicit).resolve()] if explicit else sorted((root / 'artifacts/init/300').glob('*'), reverse=True)
    for directory in candidates:
        path = directory / 'manifest.json'
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if (manifest.get('app') != app or manifest.get('preparation_id') != 'media-runtime' or
                manifest.get('machine_id') != '300' or manifest.get('backend') != 'rocm' or
                manifest.get('status') != 'prepared' or manifest.get('source_commit') != APPS[app]['commit'] or
                manifest.get('torch') != TORCH_VERSION):
            continue
        python = Path(manifest['python'])
        if not python.is_file():
            raise WorkflowError(f'准备好的 Python 环境不存在：{python}')
        if _git(Path(manifest['source']), 'rev-parse', 'HEAD') != manifest['source_commit']:
            raise WorkflowError('准备好的源码 HEAD 已变化。')
        for name in ('runtime_files', 'source_files', 'language_data_files', 'runtime_extensions'):
            for filename, digest in manifest.get(name, {}).items():
                if sha256(Path(filename)) != digest:
                    raise WorkflowError(f'准备好的运行文件发生变化：{filename}')
        if sha256(directory / 'freeze.txt') != manifest['freeze_sha256']:
            raise WorkflowError('准备环境的依赖清单已变化。')
        return manifest, path
    raise WorkflowError(f'尚未初始化 {APPS[app]["name"]}；先运行 uv run 300 init {APPS[app]["name"]}。')


def launch_media(root: Path, app: str, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f'300 {APPS[app]["name"]}', description='使用 ROCm 及本地模型启动 WebUI；启动阶段不会下载模型。')
    parser.add_argument('port', type=positive_int, nargs='?', default=APPS[app]['port'])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--models-dir', type=Path, help='模型根目录，默认 /home/ops396/models。')
    parser.add_argument('--model-dir', type=Path, help='直接指定此模型目录。')
    parser.add_argument('--preparation-dir', type=Path)
    parser.add_argument('--devices', default='0')
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument('--prompt', help='单次生成：Ideogram4 为 JSON caption，Indextts2 为文本；省略则启动 WebUI。')
    prompt.add_argument('--prompt-json', type=Path, help='Ideogram4 的 JSON caption 文件。')
    parser.add_argument('--reference', type=Path, help='Indextts2 单次合成所用参考 WAV。')
    parser.add_argument('--height', type=positive_int, default=512)
    parser.add_argument('--width', type=positive_int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sampler', default='V4_TURBO_12')
    parser.add_argument('--output-dir', type=Path, default=root / 'artifacts/media' / app)
    parser.add_argument('--dry-run', action='store_true', help='显示计划；不要求已安装运行环境或已下载模型。')
    parser.add_argument('--server-help', action='store_true', help='查看运行 worker 帮助，不加载模型。')
    args = parser.parse_args(argv)
    if args.port > 65535:
        raise WorkflowError('端口必须在 1–65535。')
    if app == 'indextts2' and args.prompt_json:
        raise WorkflowError('Indextts2 使用 --prompt 文本，不能使用 --prompt-json。')
    if app == 'ideogram4' and args.reference:
        raise WorkflowError('--reference 用于 Indextts2。')
    prompt_text = (root / args.prompt_json).read_text(encoding='utf-8') if args.prompt_json else args.prompt
    directory = (root / (args.model_dir or model_dir(app, '300', args.models_dir))).resolve()
    unresolved = None
    manifest, manifest_path = None, None
    try:
        manifest, manifest_path = resolve_runtime(root, app, args.preparation_dir)
    except WorkflowError as error:
        if not args.dry_run or args.preparation_dir or not str(error).startswith('尚未初始化'):
            raise
        unresolved = str(error)
    python = Path(manifest['python']) if manifest else root / f'artifacts/init/300/<prepared-{app}>/venv/bin/python'
    worker = Path(manifest['worker']) if manifest else root / f'artifacts/init/300/<prepared-{app}>/runtime/lab_tools/media_worker.py'
    source = Path(manifest['source']) if manifest else root / f'artifacts/init/300/<prepared-{app}>/source'
    command = [str(python), '-I', str(worker), 'run', '--app', app, '--source', str(source),
               '--model-dir', str(directory), '--output', str(args.output_dir.resolve()), '--host', args.host, '--port', str(args.port),
               '--height', str(args.height), '--width', str(args.width), '--seed', str(args.seed), '--sampler', args.sampler]
    if manifest_path:
        command += ['--expected-provenance', str(manifest_path.parent / 'provenance.json')]
    if prompt_text is not None:
        command += ['--prompt', prompt_text]
    if args.reference:
        command += ['--reference', str((root / args.reference).resolve())]
    env = {**(manifest or {}).get('environment', {}), 'HIP_VISIBLE_DEVICES': args.devices,
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'GRADIO_ANALYTICS_ENABLED': 'False',
           'PYTHONNOUSERSITE': '1', 'PYTHONPATH': '', 'AI_INFRA_MEDIA_MODEL_DIR': str(directory)}
    plan = {'machine_id': '300', 'app': app, 'argv': command, 'environment': env, 'runtime_blocker': unresolved,
            'model_dir': str(directory), 'preparation_manifest': str(manifest_path) if manifest_path else None,
            'model_download_requested': False, 'gpu_inference_validated': False,
            'missing_model_files': [f['path'] for f in files_for(app) if not (directory / f['path']).is_file()]}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    require_host('300')
    if args.server_help:
        return subprocess.run([str(python), '-I', str(worker), '--help'], cwd=root, env={**os.environ, **env}, timeout=30).returncode
    if args.reference and not (root / args.reference).is_file():
        raise WorkflowError('参考音频文件不存在。')
    identity = verify_local(app, directory)
    output = root / 'artifacts/servers/300'
    output.mkdir(parents=True, exist_ok=True)
    receipt_dir = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + app + '-', dir=output))
    receipt = {**plan, 'models': identity, 'parent_commit': _git(root, 'rev-parse', 'HEAD'),
               'parent_dirty': bool(_git(root, 'status', '--porcelain')), 'status': 'launch_requested',
               'launch_id': receipt_dir.name, 'preparation_manifest_sha256': sha256(manifest_path)}
    (receipt_dir / 'launch.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'启动记录：{receipt_dir / "launch.json"}', flush=True)
    os.chdir(root)
    os.execve(python, command, {**os.environ, **env, 'AI_INFRA_LAUNCH_MANIFEST': str(receipt_dir / 'launch.json')})
    return 0
