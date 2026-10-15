"""Prepare pinned NLTK language data without fetching a mutable downloader index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
from urllib.error import URLError
from urllib.request import build_opener, HTTPRedirectHandler
import zipfile


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f'固定 NLTK 资源不接受重定向：{newurl}')


def catalog() -> dict:
    data = json.loads(Path(__file__).with_name('nltk_resources.json').read_text())
    prefix = f'https://raw.githubusercontent.com/nltk/nltk_data/{data["commit"]}/packages/'
    for package in data['packages']:
        expected = prefix + package['subdir'] + '/' + package['id'] + '.zip'
        if package['url'] != expected or len(data['commit']) != 40:
            raise ValueError('NLTK 资源 URL 未固定到官方仓库 commit。')
    return data


def verified(path: Path, package: dict) -> bool:
    if not path.is_file() or path.stat().st_size != package['size']:
        return False
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return digest == package['sha256_checksum']


def acquire(package: dict, cache: Path, *, offline: bool = False) -> Path:
    archive = cache / package['sha256_checksum'] / (package['id'] + '.zip')
    if verified(archive, package):
        print(f'缓存校验通过：{package["id"]}｜SHA256 {package["sha256_checksum"]}', flush=True)
        return archive
    if offline:
        raise ValueError(f'offline：缺少已校验的 NLTK 资源缓存：{archive}')
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Fixed upstream URLs and content hashes replace NLTK's dynamic index. Normal
    # HTTPS/proxy configuration is retained; no NLTK security flag is disabled.
    opener = build_opener(NoRedirect())
    for attempt in range(1, 4):
        started = time.monotonic()
        print(f'下载 {package["id"]} [{attempt}/3]：{package["url"]}', flush=True)
        partial = None
        try:
            with tempfile.NamedTemporaryFile(dir=archive.parent, suffix='.partial', delete=False) as output:
                partial = Path(output.name)
                with opener.open(package['url'], timeout=30) as response:
                    if response.geturl() != package['url']:
                        raise ValueError('NLTK 下载地址发生变化。')
                    received, announced = 0, started - 1
                    while block := response.read(65536):
                        received += len(block)
                        if received > package['size']:
                            raise ValueError(f'NLTK 资源超出固定大小：{package["id"]}')
                        output.write(block)
                        now = time.monotonic()
                        if now - announced >= 1 or received == package['size']:
                            print(f'{package["id"]}：{received / package["size"]:.1%}｜'
                                  f'{received / 1024**2:.2f}/{package["size"] / 1024**2:.2f} MiB｜'
                                  f'{received / 1024**2 / max(now - started, 0.001):.2f} MiB/s', flush=True)
                            announced = now
            if not verified(partial, package):
                raise ValueError(f'NLTK 资源大小或 SHA256 不匹配：{package["id"]}')
            partial.replace(archive)
            print(f'SHA256 校验通过：{package["id"]}', flush=True)
            return archive
        except (OSError, URLError) as error:
            print(f'下载失败 [{attempt}/3]：{package["id"]}｜{error}', flush=True)
            if attempt == 3:
                raise
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
    raise RuntimeError('NLTK 下载未完成。')


def extract(package: dict, archive: Path, directory: Path) -> None:
    if not verified(archive, package):
        raise ValueError(f'拒绝解压未通过 SHA256 校验的资源：{archive}')
    target = (directory / package['subdir']).resolve()
    if not target.is_relative_to(directory.resolve()):
        raise ValueError('NLTK 解压目录超出目标路径。')
    with zipfile.ZipFile(archive) as data:
        members = data.infolist()
        if sum(member.file_size for member in members) > package['unzipped_size']:
            raise ValueError('NLTK 解压大小超出固定资源清单。')
        for member in members:
            path = PurePosixPath(member.filename)
            destination = target / path
            if (path.is_absolute() or '..' in path.parts or '\\' in member.filename or
                    not path.parts or path.parts[0] != package['id'] or
                    stat.S_ISLNK(member.external_attr >> 16) or
                    not destination.resolve().is_relative_to(target)):
                raise ValueError(f'不安全的 NLTK ZIP 路径：{member.filename}')
        target.mkdir(parents=True, exist_ok=True)
        print(f'解压 {package["id"]}：0/{len(members)} 文件', flush=True)
        for count, member in enumerate(members, 1):
            data.extract(member, target)
            print(f'解压 {package["id"]}：{count}/{len(members)} 文件', flush=True)
    for name in package['expected_files']:
        if not (directory / name).is_file():
            raise ValueError(f'NLTK 解压后缺少文件：{name}')
    # g2p_en checks ZIP paths during import, even when extracted data is present.
    destination = target / (package['id'] + '.zip')
    if destination.is_symlink():
        raise ValueError(f'NLTK ZIP 目标不能是符号链接：{destination}')
    shutil.copyfile(archive, destination)


def prepare(directory: Path, cache: Path, *, offline: bool = False) -> dict:
    data = catalog()
    directory.mkdir(parents=True, exist_ok=True)
    for number, package in enumerate(data['packages'], 1):
        print(f'语言运行资源 [{number}/{len(data["packages"])}]：{package["id"]}', flush=True)
        archive = acquire(package, cache, offline=offline)
        extract(package, archive, directory)
    receipt = {**data, 'verified': True, 'directory': str(directory.resolve())}
    (directory / 'resource-identity.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print('NLTK 语言运行资源准备完成。', flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    prepare(args.directory, args.cache_dir, offline=args.offline)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
