"""Prepare the frozen ds4 P0 inputs and isolated CUDA builds using the stdlib."""
from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

from lab_tools.cli import WorkflowError, _git, _reject_hidden_changes, _stop_process_group, _write_manifest, doctor
from lab_tools.source_tracking import source_revision

STUDY = Path('research/studies/ds4-inference')
OLD_DS4 = 'd6b516abb4a286a423fdb4071a19f86a3d9ab9c0'
SUPPORT = 'DeepSeek-V4-Flash-DSpark-support-0731.gguf'
SCORE_LLAMA_SHA256 = 'a23dd490a85fc0f3a3c8c1af870c6d645b6721eb93bb4d3ed439b684082e57d0'
OUTPUT_POLL_SECONDS = 0.2
PROGRESS_INTERVAL_SECONDS = 20.0
CUDA_PROBE = r'''#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
__global__ void smoke(int *p) { *p = 42; }
int main() {
    cudaDeviceProp p; int *d, h = 0;
    cudaError_t e = cudaGetDeviceProperties(&p, 0);
    if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); return 1; }
    printf("%s compute=%d.%d total_bytes=%zu\n", p.name, p.major, p.minor, p.totalGlobalMem);
    if (!strstr(p.name, "GB10") || p.major != 12 || p.minor != 1) return 2;
    e = cudaMalloc(&d, sizeof(int));
    if (e == cudaSuccess) { smoke<<<1,1>>>(d); e = cudaDeviceSynchronize(); }
    if (e == cudaSuccess) e = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); return 3; }
    cudaFree(d); printf("kernel_result=%d\n", h); return h == 42 ? 0 : 4;
}
'''


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('必须是正整数。')
    return number


def optional_timeout(value: str) -> int | None:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('超时必须是非负整数；0 表示不限时。')
    return number or None


def add_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--engines', nargs='+', choices=('ds4', 'llama', 'sglang', 'vllm', 'freetoken', 'all'),
                        default=['ds4', 'llama'], help='默认 ds4/llama P0；可选 sglang、vllm、freetoken 或 all。')
    parser.add_argument('--dry-run', action='store_true', help='只生成准备计划，不安装、下载或构建。')
    parser.add_argument('--models-dir', type=Path, default=Path('/data/models'))
    parser.add_argument('--model', type=Path, help='P0：同一 R0 GGUF；FreeToken：本地 safetensors 模型目录（不下载）。')
    parser.add_argument('--cuda-home', type=Path, help='CUDA toolkit 根目录；默认探测 CUDA_HOME、PATH 和 /usr/local/cuda。')
    parser.add_argument('--jobs', type=positive_int, default=4)
    parser.add_argument('--update-sources', action='store_true', help='先将所有干净 submodule 快进到 main/master，再冻结本次构建版本。')
    parser.add_argument('--frozen-p0', action='store_true', help='ds4/llama 重现 R0_MODEL 中的历史 P0 对照、R2 旧版和 scorer。默认构建当前源码。')
    parser.add_argument('--build-timeout', type=optional_timeout, default=None,
                        help='构建墙钟预算（秒）；默认不限时，0 也表示不限时。')
    parser.add_argument('--download-timeout', type=optional_timeout, default=None,
                        help='DSpark support 下载预算（秒）；默认不限时，0 也表示不限时；支持续传。')
    parser.add_argument('--offline', action='store_true', help='禁止 Git/HF 下载；缺项保存为 blocked。')
    parser.add_argument('--check-only', action='store_true', help='只盘点及冻结输入；不下载、不全文件哈希、不构建。')
    parser.add_argument('--resume', type=Path, help='恢复 blocked/interrupted 准备；验证并复用已构建二进制，保留旧日志。')


def sha256(path: Path, limit: int | None = None) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        remaining = limit
        while remaining is None or remaining > 0:
            chunk = source.read(min(8 * 1024 * 1024, remaining) if remaining is not None else 8 * 1024 * 1024)
            if not chunk:
                if remaining:
                    raise WorkflowError(f'文件被截断：{path}')
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    after = path.stat()
    if fingerprint(before) != fingerprint(after):
        raise WorkflowError(f'校验期间文件发生变化：{path}')
    return digest.hexdigest()


def fingerprint(stat: os.stat_result) -> list[int]:
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def inventory(directory: Path, reference: dict) -> tuple[list[dict], Path | None]:
    records = []
    candidates = []
    if directory.is_dir():
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or not (path.name.endswith('.gguf') or '.gguf.' in path.name):
                continue
            exact = path.name == reference['artifact']['filename']
            complete = path.suffix == '.gguf'
            records.append({'path': str(path.resolve()), 'size_bytes': path.stat().st_size,
                            'complete_filename': complete, 'frozen_p0_candidate': exact and complete})
            if exact and complete:
                candidates.append(path.resolve())
    # Other ds4-supported families are outside this frozen same-file experiment.
    return records, candidates[0] if candidates else None


def verify_model(path: Path, reference: dict, full: bool) -> dict:
    expected = reference['artifact']
    header = reference['header']
    record = {'path': str(path.resolve()), 'expected': expected, 'verified': False}
    if path.suffix != '.gguf' or not path.is_file():
        raise WorkflowError(f'需要完整的 .gguf 文件：{path}')
    record['stat'] = fingerprint(path.stat())
    record['size_bytes'] = path.stat().st_size
    if record['size_bytes'] != expected['size_bytes']:
        raise WorkflowError(f'GGUF 大小与 R0 冻结身份不同：{path}')
    with path.open('rb') as source:
        if source.read(4) != b'GGUF':
            raise WorkflowError('文件不是 GGUF。')
    record['header_hash_bytes'] = header['tensor_data_start']
    record['header_sha256'] = sha256(path, header['tensor_data_start'])
    if record['header_sha256'] != header['header_sha256']:
        raise WorkflowError('GGUF header/tensor directory 与 R0 冻结候选不同。')
    if full:
        print(f'校验完整模型 SHA256：{path}', flush=True)
        started = time.monotonic()
        record['sha256'] = sha256(path)
        record['hash_duration_seconds'] = round(time.monotonic() - started, 3)
        if record['sha256'] != expected['sha256_reported_by_hf_lfs']:
            raise WorkflowError('模型 SHA256 与冻结的 HF LFS 身份不同；停止构建。')
        record['verified'] = True
    return record


class Recorder:
    def __init__(self, root: Path, directory: Path, manifest: dict):
        self.root, self.directory, self.manifest = root, directory, manifest

    def save(self) -> None:
        _write_manifest(self.directory, self.manifest)

    def command(self, label: str, argv: list[str], cwd: Path, timeout: float | None = None,
                env: dict | None = None) -> dict:
        index = len(self.manifest['commands'])
        overrides = {'PYTHONUNBUFFERED': '1', **(env or {})}
        record = {'label': label, 'argv': argv, 'cwd': str(cwd), 'timeout_seconds': timeout,
                  'environment_overrides': overrides, 'log': f'{index:02d}-{label}.log',
                  'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()}
        self.manifest['commands'].append(record)
        self.save()
        started = time.monotonic()
        log_path = self.directory / record['log']
        print(f'\n准备步骤 [{index + 1}]：{label}', flush=True)
        descriptions = {
            'build-deps': '安装构建依赖（包括 PyTorch/CUDA 运行库），不下载模型权重。',
            'framework-source-build': '安装框架依赖并从固定源码构建扩展，不下载模型权重。',
            'kernel-source-build': '从源码编译 CUDA kernels / kernel cache。',
            'cuda-import-probe': '检查实际 import 路径与 CUDA 小探针，不加载模型。',
        }
        for suffix, description in descriptions.items():
            if label.endswith(suffix):
                print(f'内容：{description}', flush=True)
                break
        budget = '不限时（Ctrl+C 可中断）' if timeout is None else f'{timeout:.1f} 秒'
        print(f'命令：{shlex.join(argv)}\n工作目录：{cwd}\n'
              f'超时预算：{budget}\n日志：{log_path.resolve()}', flush=True)
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        def forward_output(reader) -> bool:
            # Keep the raw log unchanged, even for split UTF-8 / carriage-return
            # progress output. Read a fixed snapshot so a noisy child cannot
            # starve the timeout check by continuously appending to the log.
            end = os.fstat(reader.fileno()).st_size
            copied = False
            while reader.tell() < end:
                data = reader.read(min(65536, end - reader.tell()))
                if not data:
                    break
                copied = True
                sys.stdout.write(decoder.decode(data))
            if copied:
                sys.stdout.flush()
            return copied

        try:
            with log_path.open('wb') as log, log_path.open('rb') as reader:
                process = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, env={**os.environ, **overrides})
                try:
                    deadline = started + max(0.01, timeout) if timeout is not None else None
                    next_progress = started + PROGRESS_INTERVAL_SECONDS
                    last_output = started
                    while True:
                        if forward_output(reader):
                            last_output = time.monotonic()
                        now = time.monotonic()
                        code = process.poll()
                        if code is not None:
                            record.update(exit_code=code, status='succeeded' if code == 0 else 'failed')
                            break
                        remaining = deadline - now if deadline is not None else None
                        if remaining is not None and remaining <= 0:
                            record.update(exit_code=124, status='timed_out')
                            break
                        if now >= next_progress:
                            budget = '不限时' if remaining is None else f'剩余预算 {remaining:.1f} 秒'
                            print(f'\n进行中：{label}｜已用 {now - started:.1f} 秒｜'
                                  f'{now - last_output:.1f} 秒无新增输出｜{budget}', flush=True)
                            next_progress = now + PROGRESS_INTERVAL_SECONDS
                        try:
                            process.wait(timeout=OUTPUT_POLL_SECONDS if remaining is None
                                         else min(OUTPUT_POLL_SECONDS, remaining))
                        except subprocess.TimeoutExpired:
                            pass
                except KeyboardInterrupt:
                    record.update(exit_code=130, status='interrupted')
                    raise
                finally:
                    _stop_process_group(process)
                    forward_output(reader)
                    tail = decoder.decode(b'', final=True)
                    if tail:
                        sys.stdout.write(tail)
                        sys.stdout.flush()
        except OSError as error:
            record.update(exit_code=127, status='launch_error', error=str(error))
        finally:
            record['duration_seconds'] = round(time.monotonic() - started, 3)
            record['finished_at'] = datetime.now(timezone.utc).isoformat()
            self.save()
            status = {'succeeded': '完成', 'failed': '失败', 'timed_out': '超时',
                      'launch_error': '启动失败', 'interrupted': '已中断'}.get(record['status'], '未完成')
            print(f'\n步骤{status}：{label}｜耗时 {record["duration_seconds"]:.1f} 秒｜'
                  f'退出码 {record.get("exit_code", "未知")}\n日志：{log_path.resolve()}', flush=True)
            if record.get('error'):
                print(f'详情：{record["error"]}', flush=True)
        return record

    def require(self, *args, **kwargs) -> dict:
        record = self.command(*args, **kwargs)
        if record['exit_code']:
            raise WorkflowError(f'{record["label"]} 失败，查看 {self.directory / record["log"]}')
        return record


def cuda_home(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    configured = os.environ.get('CUDA_HOME')
    if configured:
        return Path(configured).resolve()
    nvcc = shutil.which('nvcc')
    if nvcc:
        return Path(nvcc).resolve().parents[1]
    path = Path('/usr/local/cuda')
    return path.resolve() if (path / 'bin/nvcc').is_file() else None


def prepare_sources(root: Path, reference: dict, recorder: Recorder, offline: bool,
                    frozen: bool = True) -> dict:
    sources = {}
    for name, commit in reference['engines'].items():
        path = root / 'sources' / name
        if not (path / '.git').exists():
            if offline:
                raise WorkflowError(f'离线模式缺少 sources/{name}；先初始化固定子模块。')
            recorder.require(f'init-{name.replace(".", "-")}',
                             ['git', 'submodule', 'update', '--init', '--depth', '1', '--', f'sources/{name}'], root)
        _reject_hidden_changes(path)
        if _git(path, 'status', '--porcelain', '--untracked-files=all'):
            raise WorkflowError(f'sources/{name} 必须干净；不会重置本地修改。')
        if frozen:
            try:
                _git(path, 'cat-file', '-e', commit + '^{commit}')
            except WorkflowError:
                if offline:
                    raise WorkflowError(f'offline：缺少历史 {name} 提交 {commit}。')
                recorder.require(f'fetch-frozen-{name}', ['git', 'fetch', '--depth', '1', 'origin', commit], path)
        else:
            commit, _ = source_revision(root, name, _git)
        sources[name] = {'path': str(path), 'commit': commit}
    if not frozen:
        return sources
    ds4 = Path(sources['ds4']['path'])
    try:
        _git(ds4, 'cat-file', '-e', OLD_DS4 + '^{commit}')
    except WorkflowError:
        if offline:
            raise WorkflowError('离线模式缺少 R2 旧提交对象。')
        recorder.require('fetch-ds4-old', ['git', 'fetch', '--depth', '1', 'origin', OLD_DS4], ds4)
    parents = [line[7:] for line in _git(ds4, 'cat-file', '-p', reference['engines']['ds4']).splitlines()
               if line.startswith('parent ')]
    if parents != [OLD_DS4]:
        raise WorkflowError('R2 ds4 父提交不匹配。')
    # Frozen versions have no nested source dependencies.
    for source in sources.values():
        if any(line.startswith('160000 ') for line in _git(Path(source['path']), 'ls-files', '--stage').splitlines()):
            raise WorkflowError('检测到未处理的嵌套源码依赖，需先锁定。')
    return sources


def prepare_support(root: Path, models: Path, reference: dict, recorder: Recorder, offline: bool,
                    timeout: int | None = None) -> dict:
    artifact = reference['artifact']
    api = f'https://huggingface.co/api/models/{artifact["repo"]}/revision/{artifact["revision"]}?blobs=true'
    cache = root / 'artifacts' / 'models' / artifact['revision']
    receipt = cache / 'support-identity.json'
    if offline:
        if not receipt.is_file():
            raise WorkflowError('R2 缺少冻结的 DSpark support 身份；联网运行 init 可自动准备。')
        identity = json.loads(receipt.read_text())
        if identity.get('api_url') != api or identity.get('filename') != SUPPORT:
            raise WorkflowError('DSpark support 缓存身份与当前冻结 revision 不同。')
    else:
        print('检查固定 HF revision 的 DSpark support 身份。', flush=True)
        with urllib.request.urlopen(api, timeout=None if timeout is None else min(30, timeout)) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise WorkflowError('HF API 响应超出预算。')
        data = json.loads(raw)
        if data.get('sha') != artifact['revision']:
            raise WorkflowError('HF 返回的 revision 与冻结身份不同。')
        item = next((item for item in data['siblings'] if item['rfilename'] == SUPPORT), None)
        if not item or not item.get('lfs', {}).get('sha256'):
            raise WorkflowError('冻结 HF revision 中缺少带 LFS 身份的 DSpark support。')
        identity = {'api_url': api, 'api_response_sha256': hashlib.sha256(raw).hexdigest(),
                    'filename': SUPPORT, 'size_bytes': item['lfs']['size'], 'sha256': item['lfs']['sha256']}
        cache.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(identity, indent=2) + '\n')
    candidates = sorted(models.rglob(SUPPORT)) if models.is_dir() else []
    target = next((p for p in candidates if p.is_file()), cache / SUPPORT)
    if not target.is_file():
        if offline:
            raise WorkflowError('R2 DSpark support 本地文件缺失，离线模式不下载。')
        url = f'https://huggingface.co/{artifact["repo"]}/resolve/{artifact["revision"]}/{SUPPORT}'
        pending = cache / (SUPPORT + '.incomplete')
        print(f'下载缺少的 DSpark support 到 {cache}，保留已有主模型。', flush=True)
        deadline = time.monotonic() + timeout if timeout is not None else None
        offset = pending.stat().st_size if pending.is_file() else 0
        if offset > identity['size_bytes']:
            raise WorkflowError('已有 incomplete 大于冻结文件大小；请移走后重试。')
        if offset < identity['size_bytes']:
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=None if timeout is None else min(30, timeout)) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise WorkflowError('服务器未接受续传 Range；保留 incomplete，请移走后重试。')
                with pending.open('ab' if offset else 'wb') as output:
                    total = offset
                    progress_at = time.monotonic()
                    while chunk := response.read(8 * 1024 * 1024):
                        total += len(chunk)
                        if total > identity['size_bytes']:
                            raise WorkflowError('DSpark support 下载超出冻结大小；保留 incomplete。')
                        if deadline is not None and time.monotonic() > deadline:
                            raise WorkflowError(f'DSpark support 下载超出 {timeout} 秒预算；保留 incomplete。')
                        output.write(chunk)
                        if time.monotonic() - progress_at >= 30:
                            print(f'DSpark support：{total / 1024**3:.2f}/{identity["size_bytes"] / 1024**3:.2f} GiB', flush=True)
                            progress_at = time.monotonic()
        if pending.stat().st_size != identity['size_bytes'] or sha256(pending) != identity['sha256']:
            raise WorkflowError('DSpark support 下载校验失败；保留 incomplete。')
        pending.replace(target)
    if target.stat().st_size != identity['size_bytes'] or sha256(target) != identity['sha256']:
        raise WorkflowError(f'DSpark support 身份不匹配：{target}；不会覆盖已有文件。')
    return {**identity, 'path': str(target.resolve()), 'verified': True}


def freeze_inputs(root: Path, directory: Path) -> dict:
    output = directory / 'inputs'
    output.mkdir(exist_ok=True)
    paths = {}
    for name in ('R0_MODEL.json', 'R0_INPUTS.json'):
        source = root / STUDY / name
        shutil.copyfile(source, output / name)
        paths[name] = {'path': str(output / name), 'sha256': sha256(source)}
    cases = json.loads((output / 'R0_INPUTS.json').read_text())['cases']
    for case in cases:
        for field, suffix, expected in [('prompt', 'prompt', 'prompt_utf8_sha256'),
                                         ('candidate_rendered_prompt', 'rendered', 'rendered_utf8_sha256')]:
            raw = case[field].encode('utf-8')
            if hashlib.sha256(raw).hexdigest() != case[expected]:
                raise WorkflowError(f'输入探针 {case["id"]} 的字节 hash 不匹配。')
            (output / f'{case["id"]}.{suffix}.txt').write_bytes(raw)
    return paths


def adapt_score_llama(source: Path, directory: Path) -> dict:
    if sha256(source) != SCORE_LLAMA_SHA256:
        raise WorkflowError('score_llama 源码与固定版本 hash 不同，拒绝自动适配。')
    original = source.read_text()
    old = '    model_params.use_mmap = true;'
    new = '    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;'
    if original.count(old) != 1:
        raise WorkflowError('score_llama mmap 适配位置不唯一。')
    adapted = original.replace(old, new)
    directory.mkdir(exist_ok=True)
    target = directory / 'score_llama.cpp'
    target.write_text(adapted)
    patch = directory / 'score_llama.patch'
    patch.write_text(''.join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True),
                                                fromfile=str(source), tofile=str(target))))
    return {'upstream_source': str(source), 'upstream_sha256': SCORE_LLAMA_SHA256,
            'adapted_source': str(target), 'adapted_sha256': sha256(target),
            'patch': str(patch), 'patch_sha256': sha256(patch),
            'change': 'use_mmap=true -> load_mode=LLAMA_LOAD_MODE_MMAP; scoring/tokenization unchanged'}


def build_all(root: Path, sources: dict, home: Path, args: argparse.Namespace, recorder: Recorder) -> dict:
    deadline = time.monotonic() + args.build_timeout if args.build_timeout is not None else None
    env = {'PATH': str(home / 'bin') + os.pathsep + os.environ.get('PATH', ''), 'CUDA_HOME': str(home)}
    previous = recorder.manifest.get('builds', {})
    builds = {}

    def run(label, argv, cwd):
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            raise WorkflowError('构建总预算耗尽。')
        recorder.require(label, argv, cwd, remaining, env)

    targets = [('ds4-new', 'ds4', sources['ds4']['commit']),
               ('llama', 'llama.cpp', sources['llama.cpp']['commit'])]
    if getattr(args, 'frozen_p0', False):
        targets.insert(1, ('ds4-old', 'ds4', OLD_DS4))
    for name, engine, commit in targets:
        destination = recorder.directory / name
        cached = previous.get(name, {})
        if cached.get('status') == 'built' and cached.get('commit') == commit:
            if Path(cached['source']) != destination or _git(destination, 'rev-parse', 'HEAD') != commit:
                raise WorkflowError(f'{name} 的恢复源码路径/提交已变化。')
            _reject_hidden_changes(destination)
            if _git(destination, 'status', '--porcelain', '--untracked-files=all'):
                raise WorkflowError(f'{name} 的恢复源码包含改动。')
            for binary in cached['binaries']:
                path = Path(binary['path'])
                if not path.resolve().is_relative_to(destination.resolve()) or sha256(path) != binary['sha256']:
                    raise WorkflowError(f'{name} 的已构建二进制身份已变化。')
            print(f'复用已校验构建：{name}', flush=True)
            builds[name] = cached
            continue
        if destination.exists():
            raise WorkflowError(f'{name} 上次构建未完成；保留原目录，请新建准备而非覆盖。')
        builds[name] = {'commit': commit, 'source': str(destination), 'status': 'preparing'}
        recorder.manifest['builds'] = builds
        recorder.save()
        run(f'clone-{name}', ['git', 'clone', '--no-hardlinks', '--no-checkout', '--', sources[engine]['path'], str(destination)], root)
        run(f'checkout-{name}', ['git', 'checkout', '--detach', commit], destination)
        if engine == 'ds4':
            run(f'build-{name}', ['make', f'-j{args.jobs}', 'cuda-spark', f'CUDA_HOME={home}'], destination)
            binaries = [destination / binary for binary in ('ds4', 'ds4-server', 'ds4-bench')]
        else:
            build = destination / 'build-p0'
            run('configure-llama', ['cmake', '-S', str(destination), '-B', str(build), '-DGGML_CUDA=ON',
                                    '-DCMAKE_CUDA_ARCHITECTURES=121a-real', f'-DCUDAToolkit_ROOT={home}',
                                    f'-DCMAKE_CUDA_COMPILER={home / "bin/nvcc"}', '-DCMAKE_BUILD_TYPE=Release',
                                    '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_TESTS=OFF'], destination)
            run('build-llama', ['cmake', '--build', str(build), '--parallel', str(args.jobs), '--target',
                                'llama-cli', 'llama-server', 'llama-bench', 'llama-tokenize'], destination)
            binaries = [build / 'bin' / binary for binary in ('llama-cli', 'llama-server', 'llama-bench', 'llama-tokenize')]
        builds[name]['binaries'] = []
        for binary in binaries:
            if not binary.is_file():
                raise WorkflowError(f'构建后缺少二进制：{binary}')
            linkage = recorder.require(f'ldd-{binary.name}-{name}', ['ldd', str(binary)], destination,
                                       None if deadline is None else 30, env)
            if 'not found' in (recorder.directory / linkage['log']).read_text(errors='replace'):
                raise WorkflowError(f'二进制动态库缺失：{binary}')
            builds[name]['binaries'].append({'path': str(binary), 'sha256': sha256(binary), 'linkage_log': linkage['log']})
        builds[name]['status'] = 'built'
        recorder.save()
    if not getattr(args, 'frozen_p0', False):
        return builds
    newer = Path(builds['ds4-new']['source'])
    llama = Path(builds['llama']['source'])
    run('build-score-official', ['make', f'-j{args.jobs}', 'gguf-tools/quality-testing/score_official', f'CUDA_HOME={home}', 'CUDA_ARCH=sm_121'], newer)
    adapter = adapt_score_llama(newer / 'gguf-tools/quality-testing/score_llama.cpp', recorder.directory / 'adapters')
    recorder.manifest['score_llama_adapter'] = adapter
    recorder.save()
    lib = llama / 'build-p0/bin'
    run('build-score-llama-adapted', ['c++', '-O3', '-Wall', '-Wextra', '-std=c++17',
                                    '-I' + str(llama / 'include'), '-I' + str(llama / 'ggml/include'),
                                    adapter['adapted_source'], '-o', str(newer / 'gguf-tools/quality-testing/score_llama'),
                                    '-L' + str(lib), '-Wl,-rpath,' + str(lib),
                                    '-lllama', '-lggml', '-lggml-base', '-lggml-cpu', '-ldl', '-lpthread'], newer)
    for name in ('score_official', 'score_llama'):
        binary = newer / 'gguf-tools/quality-testing' / name
        linkage = recorder.require(f'ldd-{name}', ['ldd', str(binary)], newer,
                                   None if deadline is None else 30, env)
        text = (recorder.directory / linkage['log']).read_text(errors='replace')
        if 'not found' in text or (name == 'score_llama' and str(llama / 'build-p0/bin/libllama') not in text):
            raise WorkflowError(f'{name} 未链接到准备的动态库。')
        builds['ds4-new']['binaries'].append({'path': str(binary), 'sha256': sha256(binary), 'linkage_log': linkage['log']})
    return builds


def prepare(root: Path, args: argparse.Namespace) -> tuple[Path, int]:
    if args.update_sources:
        if args.offline or args.dry_run or args.check_only or args.resume:
            raise WorkflowError('--update-sources 不能与 offline/dry-run/check-only/resume 合用；更新后请新建构建。')
        from lab_tools.source_tracking import update_sources
        update_sources(root)
    engines = set(getattr(args, 'engines', ['ds4', 'llama']))
    if 'all' in engines:
        engines = {'ds4', 'llama', 'sglang', 'vllm', 'freetoken'}
    from lab_tools.native_prepare import prepare_native, prepare_plan
    if args.dry_run:
        return prepare_plan(root, args, engines)
    if engines & {'sglang', 'vllm', 'freetoken'}:
        if engines & {'ds4', 'llama'}:
            if args.resume:
                raise WorkflowError('--resume 请指定原目录对应的单独引擎组，不要混合 P0 与 native。')
            legacy_directory, code = prepare_p0(root, args)
            if code:
                return legacy_directory, code
        return prepare_native(root, args, engines & {'sglang', 'vllm', 'freetoken'})
    # ds4/llama 是同文件 P0 对照，保留两引擎及 scorer 的成套准备。
    return prepare_p0(root, args)


def prepare_p0(root: Path, args: argparse.Namespace) -> tuple[Path, int]:
    if os.name != 'posix':
        raise WorkflowError('init 目前需要 Linux/macOS；GB10 构建需要 Linux ARM64。')
    base = root / 'artifacts' / 'init'
    base.mkdir(parents=True, exist_ok=True)
    directory = args.resume.resolve() if args.resume else Path(tempfile.mkdtemp(
        prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-ds4-p0-'), dir=base))
    previous = None
    if args.resume:
        if not directory.is_relative_to(base.resolve()):
            raise WorkflowError('只能恢复本工作台 artifacts/init 中的准备目录。')
        previous = json.loads((directory / 'manifest.json').read_text())
        if not isinstance(previous, dict) or previous.get('preparation_id') != 'ds4-p0' or previous.get('schema_version') != 1:
            raise WorkflowError('恢复目录不是支持的 ds4-p0 准备清单。')
        if previous.get('status') not in ('blocked', 'interrupted') or args.check_only:
            raise WorkflowError('只能恢复 blocked/interrupted 准备；prepared 目录保持不可变。')
        if not isinstance(previous.get('inputs'), dict) or set(previous['inputs']) != {'R0_MODEL.json', 'R0_INPUTS.json'}:
            raise WorkflowError('恢复清单缺少冻结输入身份；请新建准备。')
        for name, record in previous['inputs'].items():
            if not isinstance(record, dict) or not isinstance(record.get('sha256'), str):
                raise WorkflowError('恢复清单的输入身份格式不正确。')
            if sha256(root / STUDY / name) != record['sha256']:
                raise WorkflowError('冻结输入已变化，请新建准备目录。')
        attempt = len(list(directory.glob('attempt-*.json'))) + 1
        shutil.copyfile(directory / 'manifest.json', directory / f'attempt-{attempt}.json')
    manifest = {'schema_version': 1, 'preparation_id': 'ds4-p0', 'run_id': directory.name,
                'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'preparing', 'commands': [],
                'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'ready_for_validation': False, 'gpu_inference_validated': False,
                'pending_validation': ['actual token ID alignment', 'same-file loading and 32-token generation',
                                       'logits/teacher-forced correctness', 'R1 timings', 'R2 rewind triggers and state correctness'],
                'blockers': []}
    if previous:
        manifest['commands'] = previous['commands']
        manifest['builds'] = previous.get('builds', {})
        manifest['previous_attempt'] = f'attempt-{attempt}.json'
    manifest['inherited_build_environment'] = {
        name: os.environ[name] for name in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS',
                                           'NVCC', 'NVCCFLAGS', 'CUDA_ARCH', 'CUDA_HOME', 'LD_LIBRARY_PATH',
                                           'CMAKE_PREFIX_PATH', 'CPATH', 'LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
                                           'CUDA_DEVICE_ORDER', 'CUDACXX', 'CUDAHOSTCXX') if name in os.environ
    }
    manifest['tooling_sha256'] = {
        str(path.relative_to(root)): sha256(path)
        for path in (root / 'lab_tools').glob('*.py') if path.is_file()
    }
    recorder = Recorder(root, directory, manifest)
    recorder.save()
    print(f'准备目录：{directory}', flush=True)
    exit_code = 2
    try:
        manifest['environment'] = doctor(root)
        manifest['parent_commit'] = manifest['environment']['project_commit']
        if previous and previous.get('parent_commit') != manifest['parent_commit']:
            raise WorkflowError('主仓库提交已变化，请新建准备目录。')
        manifest['inputs'] = freeze_inputs(root, directory)
        reference = json.loads((directory / 'inputs/R0_MODEL.json').read_text())
        records, candidate = inventory(args.models_dir, reference)
        manifest['model_inventory'] = records
        model = args.model.resolve() if args.model else candidate
        if model is None:
            raise WorkflowError(f'{args.models_dir} 中未找到冻结的 Flash 0731 Q2；可用 --model 指定同一 artifact。')
        manifest['model'] = verify_model(model, reference, full=not args.check_only)
        meminfo = Path('/proc/meminfo')
        manifest['memory'] = meminfo.read_text() if meminfo.is_file() else None
        usage = shutil.disk_usage(directory)
        manifest['disk'] = {'path': str(directory), 'free_bytes': usage.free, 'total_bytes': usage.total}
        home = cuda_home(args.cuda_home)
        manifest['cuda_home'] = str(home) if home else None
        if previous and previous.get('cuda_home') != manifest['cuda_home']:
            raise WorkflowError('CUDA toolkit 路径已变化，请新建准备目录。')
        versions = []
        for tool in ('cc', 'c++', 'make', 'cmake'):
            versions.append(recorder.command(f'version-{tool.replace("+", "p")}', [tool, '--version'], root))
        if home:
            versions.append(recorder.command('version-nvcc', [str(home / 'bin/nvcc'), '--version'], root))
        if args.check_only:
            manifest['status'] = 'inventory_only'
            manifest['blockers'].append('check-only：完整 SHA256、support、CUDA 探针及构建未执行。')
            return directory, 2
        manifest['source_mode'] = 'frozen-p0' if args.frozen_p0 else 'current-checkout'
        manifest['sources'] = prepare_sources(root, reference, recorder, args.offline, args.frozen_p0)
        if previous and previous.get('sources') != manifest['sources']:
            raise WorkflowError('恢复目录的源码版本与本次选择不同，请新建初始化。')
        recorder.save()
        if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
            raise WorkflowError('P0 的 sm_121a 构建需要 Linux ARM64 GB10 环境。')
        if not home or not (home / 'bin/nvcc').is_file():
            raise WorkflowError('缺少 nvcc；安装 ARM64 CUDA toolkit 后通过 --cuda-home 指定。')
        if any(record['exit_code'] for record in versions):
            raise WorkflowError('编译工具缺失，查看 version 日志。')
        if usage.free < 10 * 1024**3:
            raise WorkflowError('构建目录可用磁盘低于 10 GiB；需先腾出空间。')
        probe = directory / 'cuda-probe.cu'
        probe.write_text(CUDA_PROBE)
        recorder.require('compile-cuda-probe', [str(home / 'bin/nvcc'), '-arch=sm_121a', str(probe), '-o', str(directory / 'cuda-probe')],
                         root, None if args.build_timeout is None else 120)
        runtime = recorder.command('run-cuda-probe', [str(directory / 'cuda-probe')],
                                   root, None if args.build_timeout is None else 30)
        manifest['cuda_smoke_passed'] = runtime['exit_code'] == 0
        if runtime['exit_code']:
            manifest['blockers'].append('CUDA 设备/kernel 探针失败；核对设备访问权限、驱动与 runtime 日志。')
        manifest['builds'] = build_all(root, manifest['sources'], home, args, recorder)
        if args.frozen_p0:
            support_started = time.monotonic()
            manifest['support'] = prepare_support(root, args.models_dir, reference, recorder, args.offline,
                                                  args.download_timeout)
            manifest['support']['preparation_duration_seconds'] = round(time.monotonic() - support_started, 3)
        manifest['ready_for_validation'] = not manifest['blockers']
        manifest['status'] = 'prepared' if manifest['ready_for_validation'] else 'blocked'
        exit_code = 0 if manifest['ready_for_validation'] else 2
    except (WorkflowError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        manifest['status'] = 'blocked'
        manifest['blockers'].append(str(error))
        print(f'准备未完成：{error}', flush=True)
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
        manifest['blockers'].append('准备被中断；保留日志与中间产物。')
        exit_code = 130
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        recorder.save()
        (directory / 'README.md').write_text(
            '# ds4 P0 环境准备\n\n' + f'状态：{manifest["status"]}。详见 manifest.json 与逐步骤日志。\n\n'
            '已理解：本目录固定输入、模型身份、环境与实际构建路径。\n\n'
            '未解决：token 对齐、加载/短生成、质量、R1 性能和 R2 rewind 验证仍待执行。\n\n'
            '下一步：先处理 manifest.json 的 blockers；prepared 后按 R0.md 顺序验证。'
            '长实验应在已提交的干净独立主仓库 clone 中重新 init。\n', encoding='utf-8')
    return directory, exit_code
