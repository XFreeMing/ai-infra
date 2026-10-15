# /// script
# requires-python = ">=3.12"
# dependencies = ["modelscope==1.40.2"]
# [[tool.uv.index]]
# name = "aliyun"
# url = "https://mirrors.aliyun.com/pypi/simple"
# default = true
# ///
"""ModelScope SDK worker, invoked only by `uv run download`, never by init/launch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def digest_file(path: Path) -> tuple[str, list]:
    before = path.stat()
    identity = [before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns]
    digest = hashlib.sha256()
    size, done, announced = before.st_size, 0, 0.0
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
            done += len(block)
            if time.monotonic() - announced >= 20:
                print(f'SHA256 {path.name}：{done / max(size, 1):.1%}｜{done / 1024**3:.2f}/{size / 1024**3:.2f} GiB', flush=True)
                announced = time.monotonic()
    after = path.stat()
    if identity != [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]:
        raise ValueError(f'校验期间文件变化：{path}')
    return digest.hexdigest(), identity


def execute(plan: dict, snapshot_download, progress_callback=None) -> dict:
    directory = Path(plan['directory']).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    repos = plan['repositories']
    completed_bytes = 0
    total = plan['total_bytes']
    for index, repo in enumerate(repos, 1):
        destination = (directory / repo['directory']).resolve()
        if not destination.is_relative_to(directory):
            raise ValueError('辅助模型目标路径超出目录。')
        destination.mkdir(parents=True, exist_ok=True)
        missing = sum(f['size_bytes'] for f in repo['files'] if not (destination / f['path']).is_file()
                      or (destination / f['path']).stat().st_size != f['size_bytes'])
        if shutil.disk_usage(destination).free < missing + 1024**3:
            raise ValueError(f'磁盘空间不足：{destination}，缺失文件约 {missing / 1024**3:.2f} GiB，另预留 1 GiB。')
        print(f'\n仓库 [{index}/{len(repos)}]：{repo["model_id"]}\n版本：{repo["revision"]}\n目标：{destination}', flush=True)
        print('ModelScope 将显示逐文件下载字节、百分比与速度；已有缓存由 SDK 复用。', flush=True)
        # Explicit file patterns avoid downloading optimizer/training copies and other model variants.
        progress = {'progress_callbacks': [progress_callback]} if progress_callback is not None else {}
        snapshot_download(model_id=repo['model_id'], revision=repo['revision'], local_dir=str(destination),
                          allow_file_pattern=[f['path'] for f in repo['files']], max_workers=4, **progress)
        for file_index, expected in enumerate(repo['files'], 1):
            path = (destination / expected['path']).resolve()
            if not path.is_relative_to(destination) or not path.is_file() or path.stat().st_size != expected['size_bytes']:
                raise ValueError(f'下载后的文件缺失或大小错误：{path}')
            print(f'完整校验 [{file_index}/{len(repo["files"])}]：{expected["path"]}', flush=True)
            digest, stat = digest_file(path)
            if digest != expected['sha256']:
                raise ValueError(f'SHA256 不匹配，文件保留：{path}')
            records.append({**expected, 'path': path.relative_to(directory).as_posix(), 'stat': stat})
        completed_bytes += sum(f['size_bytes'] for f in repo['files'])
        print(f'总进度（已完成下载与校验）：{completed_bytes / max(total, 1):.1%}｜{completed_bytes / 1024**3:.2f}/{total / 1024**3:.2f} GiB', flush=True)
    receipt = {'schema_version': 1, 'app': plan['app'], 'machine_id': plan['machine_id'], 'provider': 'modelscope',
               'directory': str(directory), 'repositories': [{k: r[k] for k in ('model_id', 'revision', 'directory')} for r in repos],
               'files': records, 'verified': True, 'verified_at': datetime.now(timezone.utc).isoformat()}
    (directory / 'download-identity.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'全部模型下载和校验完成：{directory}', flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    try:
        from modelscope import snapshot_download
        from modelscope.hub.callback import TqdmCallback
        execute(json.loads(args.plan.read_text(encoding='utf-8')), snapshot_download, TqdmCallback)
        return 0
    except Exception as error:
        print(f'下载未完成，保留缓存与文件：{error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
