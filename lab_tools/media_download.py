"""Explicit ModelScope downloads; never invoked by runtime init or launch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from lab_tools.cli import WorkflowError, _git
from lab_tools.media_models import app_name, catalog, detect_machine, files_for, model_dir, verify_local
from lab_tools.native_prepare import default_package_index
from lab_tools.prepare import Recorder, optional_timeout


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    parser = argparse.ArgumentParser(prog='download', description='从 ModelScope 下载固定版本模型及必要辅助模型，显示进度并校验 SHA256。')
    parser.add_argument('model', help='Ideogram4 / Indextts2，不区分大小写。')
    parser.add_argument('--machine', choices=('300', '700'), default=detect_machine())
    parser.add_argument('--models-dir', type=Path, help='模型根目录；300 默认 /home/ops396/models，700 默认 /data/models。')
    parser.add_argument('--local-dir', type=Path, help='直接指定此模型目录，优先于 --models-dir。')
    parser.add_argument('--dry-run', action='store_true', help='只显示完整文件/仓库计划，不安装 SDK、不访问网络、不写模型目录。')
    parser.add_argument('--verify-only', action='store_true', help='仅完整校验已下载文件，不安装 SDK、不访问网络。')
    parser.add_argument('--timeout', type=optional_timeout, default=None, help='下载预算秒数，默认不限时；0 同样不限时。')
    args = parser.parse_args(argv)
    try:
        app = app_name(args.model)
        directory = (args.local_dir or model_dir(app, args.machine, args.models_dir)).resolve()
        package_index = default_package_index(root)
        plan = {'app': app, 'machine_id': args.machine, 'provider': 'modelscope', 'directory': str(directory),
                'package_default_index': package_index,
                'repositories': catalog(app), 'total_bytes': sum(f['size_bytes'] for f in files_for(app)),
                'download_requested': not (args.dry_run or args.verify_only), 'huggingface_download_requested': False}
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            return 0
        if args.verify_only:
            receipt = verify_local(app, directory, reuse=False)
            receipt['verified_at'] = datetime.now(timezone.utc).isoformat()
            (directory / 'download-identity.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            return 0
        base = root / 'artifacts/downloads' / args.machine
        base.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + app + '-', dir=base))
        (output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        manifest = {**plan, 'run_id': output.name, 'parent_commit': _git(root, 'rev-parse', 'HEAD'),
                    'parent_dirty': bool(_git(root, 'status', '--porcelain')), 'status': 'downloading', 'commands': [],
                    'timeout_seconds': args.timeout, 'started_at': datetime.now(timezone.utc).isoformat()}
        recorder = Recorder(root, output, manifest)
        recorder.save()
        print(f'下载清单与实时日志：{output}', flush=True)
        # A dedicated uv script environment keeps the research environment dependency-free.
        result = recorder.command('modelscope-download', ['uv', 'run', '--no-project', '--default-index', package_index, '--script',
                                  str(root / 'scripts/download_media.py'), '--plan', str(output / 'plan.json')],
                                  root, args.timeout, {'PYTHONUNBUFFERED': '1', 'UV_DEFAULT_INDEX': package_index,
                                                       'PIP_INDEX_URL': package_index})
        manifest['exit_code'] = result['exit_code']
        manifest['status'] = 'downloaded' if result['exit_code'] == 0 else result['status']
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        recorder.save()
        return result['exit_code']
    except KeyboardInterrupt:
        if 'recorder' in locals():
            recorder.manifest.update(status='interrupted', exit_code=130)
            recorder.save()
        return 130
    except (WorkflowError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'模型下载/校验失败：{error}', file=sys.stderr)
        return 2
