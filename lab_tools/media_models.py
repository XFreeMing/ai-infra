"""Shared names, pinned source/model identities and machine-specific local paths."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import time

from lab_tools.cli import WorkflowError
from lab_tools.prepare import fingerprint

APPS = {
    'ideogram4': {'name': 'Ideogram4', 'module': 'ideogram4', 'python': '3.12', 'port': 7861,
                 'repository': 'https://github.com/ideogram-oss/ideogram4.git',
                 'commit': '990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2'},
    'indextts2': {'name': 'Indextts2', 'module': 'indextts.infer_v2', 'python': '3.11', 'port': 7860,
                 'repository': 'https://github.com/index-tts/index-tts.git',
                 'commit': '830f6f8f94a51fea23ab1d639027a86200075a4e'},
}
TORCH_VERSION = '2.11.0+rocm7.2'


def app_name(value: str) -> str:
    key = value.casefold()
    if key not in APPS:
        raise WorkflowError('模型名称请选择 Ideogram4 或 Indextts2（不区分大小写）。')
    return key


def models_root(machine: str) -> Path:
    if machine == '300':
        return Path('/home/ops396/models')
    if machine == '700':
        return Path('/data/models')
    raise WorkflowError('模型下载目录目前只配置了 300 和 700。')


def detect_machine() -> str:
    return '700' if platform.machine().lower() in ('aarch64', 'arm64') else '300'


def catalog(app: str) -> list[dict]:
    return json.loads(Path(__file__).with_name('media_models.json').read_text(encoding='utf-8'))['models'][app]


def model_dir(app: str, machine: str = '300', base: Path | None = None) -> Path:
    return (base or models_root(machine)) / APPS[app]['name']


def inside(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise WorkflowError(f'模型路径超出目录：{relative}')
    return path


def files_for(app: str) -> list[dict]:
    return [{**file, 'path': (Path(repo['directory']) / file['path']).as_posix()}
            for repo in catalog(app) for file in repo['files']]


def hash_progress(path: Path) -> str:
    import hashlib
    before = fingerprint(path.stat())
    size, read, started, announced = path.stat().st_size, 0, time.monotonic(), 0.0
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
            read += len(block)
            if time.monotonic() - announced >= 20:
                print(f'校验 {path.name}：{read / max(size, 1):.1%}｜{read / 1024**3:.2f}/{size / 1024**3:.2f} GiB', flush=True)
                announced = time.monotonic()
    if before != fingerprint(path.stat()):
        raise WorkflowError(f'校验期间模型发生变化：{path}')
    print(f'校验完成 {path.name}｜{time.monotonic() - started:.1f} 秒', flush=True)
    return digest.hexdigest()


def verify_local(app: str, directory: Path, *, reuse: bool = True) -> dict:
    """Require the full catalog; reuse hashes only with matching stat and expected SHA256."""
    receipt = directory / 'download-identity.json'
    previous = json.loads(receipt.read_text(encoding='utf-8')) if reuse and receipt.is_file() else {}
    cached = {item['path']: item for item in previous.get('files', [])}
    records = []
    expected_files = files_for(app)
    for index, expected in enumerate(expected_files, 1):
        path = inside(directory, expected['path'])
        if not path.is_file() or path.stat().st_size != expected['size_bytes']:
            raise WorkflowError(f'模型文件缺失/大小不符：{path}；先运行 uv run download {APPS[app]["name"]}。')
        stat = fingerprint(path.stat())
        known = cached.get(expected['path'], {})
        if known.get('stat') == stat and known.get('sha256') == expected['sha256']:
            digest = known['sha256']
        else:
            print(f'文件身份 [{index}/{len(expected_files)}]：{expected["path"]}', flush=True)
            digest = hash_progress(path)
        if digest != expected['sha256']:
            raise WorkflowError(f'模型 SHA256 不符，保留原文件：{path}')
        records.append({**expected, 'stat': stat})
    return {'schema_version': 1, 'app': app, 'provider': 'modelscope', 'directory': str(directory.resolve()),
            'repositories': [{k: repo[k] for k in ('model_id', 'revision', 'directory')} for repo in catalog(app)],
            'files': records, 'verified': True}
