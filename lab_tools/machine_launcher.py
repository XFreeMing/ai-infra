"""Launch only builds matching the selected machine, backend, source and checksums."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile

from lab_tools.cli import WorkflowError, _git, _reject_hidden_changes, _submodules
from lab_tools.machines import PROFILES, require_host, require_rocm, rocm_report
from lab_tools.prepare import fingerprint, positive_int, sha256
from lab_tools.source_tracking import source_revision
from lab_tools.media_models import models_root


def model_record(path: Path, *, full: bool = True) -> dict:
    if path.suffix.lower() != '.gguf' or not path.is_file():
        raise WorkflowError(f'需要完整的本地 GGUF：{path}；用 --model 指定，不自动下载。')
    before = fingerprint(path.stat())
    with path.open('rb') as handle:
        if handle.read(4) != b'GGUF':
            raise WorkflowError(f'不是 GGUF：{path}')
    record = {'path': str(path), 'stat': before, 'size_bytes': path.stat().st_size,
              'sha256': sha256(path) if full else None}
    if before != fingerprint(path.stat()):
        raise WorkflowError(f'读取期间模型发生变化：{path}')
    return record


def resolve_build(root: Path, machine: str, engine: str, explicit: Path | None) -> tuple[Path, dict, Path]:
    name = 'llama.cpp' if engine == 'llama' else 'ds4'
    commit, _ = source_revision(root, name, _git)
    directories = [(root / explicit).resolve()] if explicit else sorted(
        (root / 'artifacts/init' / machine).glob('*'), reverse=True)
    for directory in directories:
        path = directory / 'manifest.json'
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding='utf-8'))
        build = manifest.get('builds', {}).get(engine)
        if (manifest.get('machine_id') != machine or manifest.get('backend') != PROFILES[machine]['backend']
                or manifest.get('status') != 'prepared' or not build or build.get('commit') != commit):
            continue
        source = Path(build['source'])
        _reject_hidden_changes(source)
        if _git(source, 'rev-parse', 'HEAD') != commit or _git(source, 'status', '--porcelain', '--untracked-files=no'):
            raise WorkflowError('已准备的引擎源码发生变化；请新建准备。')
        if _submodules(source) != build.get('nested_submodules', []):
            raise WorkflowError('已准备的嵌套源码版本或状态发生变化。')
        for binary in build['binaries']:
            if sha256(Path(binary['path'])) != binary['sha256']:
                raise WorkflowError(f'二进制/动态库发生变化：{binary["path"]}')
        server = Path(build['server'])
        if str(server) not in {record['path'] for record in build['binaries']}:
            raise WorkflowError('服务二进制不在构建身份清单中。')
        if machine == '300' and not manifest.get('gpu_smoke_passed'):
            raise WorkflowError('ROCm 准备缺少 GPU kernel smoke。')
        return server, manifest, path
    raise WorkflowError(f'尚无匹配 AI {machine}/{engine} 的 prepared 构建；运行 uv run {machine} init {engine}。')


def launch_machine(root: Path, machine: str, engine: str, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f'{machine} {engine}', description='启动本机服务，权重须在本地。')
    parser.add_argument('port', type=positive_int, nargs='?', default=1111 if engine == 'ds4' else 2222)
    parser.add_argument('--model', type=Path, help='本地 GGUF；默认读取准备清单的 model，否则 models/model.gguf。')
    parser.add_argument('--preparation-dir', type=Path)
    parser.add_argument('--context', type=positive_int, default=2048 if machine == '100' else 4096)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--devices', default='0', help='300 的 HIP_VISIBLE_DEVICES。')
    parser.add_argument('--dry-run', action='store_true', help='显示计划，不要求已准备或已下载模型。')
    parser.add_argument('--server-help', action='store_true', help='显示已构建服务帮助，不加载模型。')
    separator = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    if args.port > 65535:
        raise WorkflowError('端口必须在 1–65535。')
    extra = argv[separator + 1:]
    manifest = None
    manifest_path = None
    unresolved = None
    try:
        binary, manifest, manifest_path = resolve_build(root, machine, engine, args.preparation_dir)
    except WorkflowError as error:
        # An explicit or already prepared selection with corrupt evidence must fail even in dry-run.
        if not args.dry_run or args.preparation_dir or not str(error).startswith('尚无匹配'):
            raise
        binary = root / f'artifacts/init/{machine}/<prepared-run>/{engine}/' / (
            'source/ds4-server' if engine == 'ds4' else 'build/bin/Release/llama-server.exe' if machine == '100' else 'build/bin/llama-server')
        unresolved = str(error)
    default_model = models_root('300') / 'model.gguf' if machine == '300' else Path('models/model.gguf')
    model = (root / (args.model or (manifest or {}).get('model', {}).get('path', default_model))).resolve()
    command = [str(binary), '-m', str(model), '-c', str(args.context), '--host', args.host, '--port', str(args.port)]
    if engine == 'ds4':
        command += ['--rocm', '--prefill-chunk', '4096', '-n', '512']
    else:
        command += ['--gpu-layers', '0' if machine == '100' else 'all', '--fit', 'off', '--parallel', '1',
                    '-b', '512' if machine == '100' else '4096', '-ub', '128' if machine == '100' else '2048']
    command += extra
    paths = {'model': model}
    flags = {'-m': 'model', '--model': 'model', '--mmproj': 'mmproj', '--vision': 'vision', '--mtp-model': 'mtp_model'}
    for index, value in enumerate(command):
        flag, sep, inline = value.partition('=')
        if flag not in flags:
            continue
        if not sep and index + 1 >= len(command):
            raise WorkflowError(f'{flag} 缺少模型路径。')
        path = (root / (inline if sep else command[index + 1])).resolve()
        paths[flags[flag]] = path
        if sep:
            command[index] = flag + '=' + str(path)
        else:
            command[index + 1] = str(path)
    env = (manifest or {}).get('builds', {}).get(engine, {}).get('runtime_environment', {}).copy()
    if machine == '300':
        env['HIP_VISIBLE_DEVICES'] = args.devices
    plan = {'machine_id': machine, 'backend': PROFILES[machine]['backend'], 'engine': engine,
            'argv': command, 'environment_overrides': env, 'framework_blocker': unresolved,
            'preparation_manifest': str(manifest_path) if manifest_path else None,
            'model_paths': {name: str(path) for name, path in paths.items()}, 'gpu_inference_validated': False}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    require_host(machine)
    if machine == '300':
        require_rocm(rocm_report(Path(env['ROCM_PATH'])))
    full_env = {**os.environ, **env}
    if args.server_help:
        return subprocess.run([str(binary), '--help'], cwd=root, env=full_env, timeout=30).returncode
    if machine == '300':
        probe = manifest['gpu_probe']
        if sha256(Path(probe['binary'])) != probe['sha256']:
            raise WorkflowError('HIP 探针二进制发生变化。')
        result = subprocess.run([probe['binary']], env=full_env, capture_output=True, text=True, timeout=15)
        plan['gpu_probe'] = {'exit_code': result.returncode, 'output': result.stdout, 'error': result.stderr}
        if result.returncode:
            raise WorkflowError(f'启动前 HIP kernel smoke 失败：{result.stdout} {result.stderr}')
    # Reuse a full-file checksum only when the stat fingerprint still matches.
    recorded = (manifest or {}).get('model', {})
    identities = {}
    for name, path in paths.items():
        if name == 'model' and recorded.get('path') == str(path) and recorded.get('sha256') and recorded.get('stat') == fingerprint(path.stat()):
            identities[name] = recorded
        else:
            identities[name] = model_record(path)
    parent = root / 'artifacts/servers' / machine
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + engine + '-', dir=parent))
    receipt = {**plan, 'kind': 'service_launch_request', 'launch_id': directory.name,
               'requested_at': datetime.now(timezone.utc).isoformat(), 'parent_commit': _git(root, 'rev-parse', 'HEAD'),
               'parent_dirty': bool(_git(root, 'status', '--porcelain')), 'models': identities,
               'binary_sha256': sha256(binary), 'preparation_manifest_sha256': sha256(manifest_path),
               'configuration_sha256': {str(path): sha256(path) for path in
                                        (Path(__file__), Path(__file__).with_name('machines.py'))}}
    (directory / 'launch.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'启动记录：{directory / "launch.json"}', flush=True)
    if os.name == 'nt':
        return subprocess.run(command, cwd=root, env=full_env).returncode
    os.chdir(root)
    os.execve(binary, command, full_env)
    return 0
