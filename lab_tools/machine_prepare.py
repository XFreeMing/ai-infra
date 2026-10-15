"""Isolated ds4 ROCm and llama.cpp HIP/Windows CPU preparation, with raw evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from lab_tools.cli import WorkflowError, _git, _reject_hidden_changes, _submodules, doctor
from lab_tools.machines import PROFILES, require_host, require_rocm, rocm_home
from lab_tools.prepare import Recorder, sha256
from lab_tools.source_tracking import source_revision, update_sources

SOURCE_NAMES = {'ds4': 'ds4', 'llama': 'llama.cpp'}
# Check allocation, host/device copy and a kernel, rather than treating rocminfo as a smoke test.
HIP_PROBE = r'''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
__global__ void smoke(int *p) { *p = 42; }
#define CHECK(call) do { hipError_t e = (call); if(e != hipSuccess) { \
    fprintf(stderr, "%s\n", hipGetErrorString(e)); return 1; } } while(0)
int main() {
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    printf("gpu=%s arch=%s total_bytes=%zu\n", prop.name, prop.gcnArchName, prop.totalGlobalMem);
    if (strncmp(prop.gcnArchName, "gfx1151", 7)) return 2;
    int *device, host = 0;
    CHECK(hipMalloc(&device, sizeof(int)));
    hipLaunchKernelGGL(smoke, dim3(1), dim3(1), 0, 0, device);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(&host, device, sizeof(int), hipMemcpyDeviceToHost)); CHECK(hipFree(device));
    printf("kernel_result=%d\n", host); return host == 42 ? 0 : 3;
}
'''


def build_plan(root: Path, directory: Path, machine: str, engine: str, args) -> dict:
    name = SOURCE_NAMES[engine]
    commit, pin_ref = source_revision(root, name, _git)
    source = directory / engine / 'source'
    build = directory / engine / 'build'
    home = rocm_home(args.rocm_home)
    env = {'HIP_VISIBLE_DEVICES': '0', 'ROCM_PATH': str(home), 'HIP_PATH': str(home),
           'HIPCXX': str(home / 'llvm/bin/clang'),
           'PATH': str(home / 'bin') + os.pathsep + os.environ.get('PATH', '')} if machine == '300' else {}
    env['NINJA_STATUS'] = '[%f/%t %p | %e s] '
    local = root / 'sources' / name
    url = _git(root, 'config', '-f', '.gitmodules', '--get', f'submodule.sources/{name}.url')
    if (local / '.git').exists():
        commands = [('clone', ['git', 'clone', '--progress', '--no-hardlinks', '--no-checkout', str(local), str(source)], root)]
    else:
        commands = [('git-init', ['git', 'init', str(source)], root),
                    ('remote', ['git', '-C', str(source), 'remote', 'add', 'origin', url], root),
                    ('fetch', ['git', '-C', str(source), 'fetch', '--progress', '--depth=1', 'origin', commit], root)]
    commands.append(('checkout', ['git', '-C', str(source), 'checkout', '--detach', commit], root))
    git = ['git', '-C', str(source)]
    if args.offline:
        git += ['-c', 'protocol.allow=never', '-c', 'protocol.file.allow=always']
    commands.append(('submodules', [*git, 'submodule', 'update', '--progress', '--init', '--recursive'], root))
    if engine == 'ds4':
        commands.append(('build', ['make', f'-j{args.jobs}', 'strix-halo',
                                   f'HIPCC={home / "bin/hipcc"}', 'ROCM_ARCH=gfx1151'], source))
    else:
        configure = ['cmake', '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                     '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DGGML_CUDA=OFF',
                     '-DGGML_HIP=' + ('ON' if machine == '300' else 'OFF'), '-DGGML_VULKAN=OFF']
        if machine == '300':
            configure += ['-DGPU_TARGETS=gfx1151', f'-DCMAKE_PREFIX_PATH={home}',
                          f'-DCMAKE_HIP_COMPILER={home / "llvm/bin/clang"}']
        if args.offline:
            configure += ['-DFETCHCONTENT_FULLY_DISCONNECTED=ON']
        commands += [('configure', configure, root),
                     ('build', ['cmake', '--build', str(build), '--verbose', '--config', 'Release', '--parallel', str(args.jobs),
                                '--target', 'llama-server', 'llama-cli', 'llama-bench'], root)]
    return {'commit': commit, 'pin_ref': pin_ref, 'url': url, 'source': str(source), 'build': str(build),
            'environment': env, 'requires_local_source': args.offline,
            'commands': [{'label': label, 'argv': argv, 'cwd': str(cwd)} for label, argv, cwd in commands]}


def prepare_machine(root: Path, machine: str, args) -> tuple[Path, int]:
    started = time.monotonic()
    if machine == '300' and not args.dry_run and not args.check_only:
        from lab_tools.native_prepare import initialization_lock
        with initialization_lock(root, args.build_timeout):
            return _prepare_machine(root, machine, args, started)
    return _prepare_machine(root, machine, args, started)


def _prepare_machine(root: Path, machine: str, args, started: float) -> tuple[Path, int]:
    if args.resume:
        raise WorkflowError(f'{machine} 准备失败后请新建目录；当前尚不支持 --resume。原日志和构建保留。')
    if args.frozen_p0 or args.cuda_home:
        raise WorkflowError('--frozen-p0 / --cuda-home 仅适用于 700；300 使用 --rocm-home。')
    if machine == '100' and args.rocm_home:
        raise WorkflowError('100 使用 CPU 后端；--rocm-home 属于 300。')
    if args.update_sources:
        if args.offline or args.dry_run or args.check_only:
            raise WorkflowError('--update-sources 不能与 offline/dry-run/check-only 合用。')
        require_host(machine)
        update_sources(root)
    base = root / 'artifacts/init' / machine
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = Path(tempfile.mkdtemp(prefix=f'{stamp}-{PROFILES[machine]["backend"]}-', dir=base))
    manifest = {'schema_version': 1, 'preparation_id': f'machine-{machine}', 'machine_id': machine,
                'backend': PROFILES[machine]['backend'], 'run_id': directory.name,
                'parent_commit': _git(root, 'rev-parse', 'HEAD'), 'started_at': datetime.now(timezone.utc).isoformat(),
                'parent_dirty': bool(_git(root, 'status', '--porcelain')),
                'selected_engines': args.engines, 'status': 'planned', 'commands': [], 'blockers': [], 'builds': {},
                'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'declared_profile': PROFILES[machine], 'profile_source': 'user supplied; not measured',
                'gpu_smoke_passed': False, 'gpu_inference_validated': False, 'model_download_requested': False,
                'tooling_sha256': {str(p.relative_to(root)): sha256(p) for p in (root / 'lab_tools').glob('*.py')},
                'plans': {engine: build_plan(root, directory, machine, engine, args) for engine in args.engines}}
    recorder = Recorder(root, directory, manifest)
    manifest['inherited_build_environment'] = {name: os.environ[name] for name in
        ('PATH', 'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'LD_LIBRARY_PATH', 'CMAKE_GENERATOR',
         'ROCM_PATH', 'HIP_PATH', 'HIP_VISIBLE_DEVICES', 'INCLUDE', 'LIB', 'CL', '_CL_') if name in os.environ}
    if machine == '300':
        home = rocm_home(args.rocm_home)
        manifest['gpu_probe_plan'] = [
            [str(home / 'bin/hipcc'), '--offload-arch=gfx1151', str(directory / 'hip-probe.cpp'), '-o', str(directory / 'hip-probe')],
            [str(directory / 'hip-probe')]]
        manifest['gpu_probe_source_sha256'] = hashlib.sha256(HIP_PROBE.encode()).hexdigest()
    recorder.save()
    print(f'AI {machine} 准备目录：{directory}', flush=True)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return directory, 0

    def step(label, argv, cwd=root, env=None, limit=None):
        remaining = args.build_timeout - (time.monotonic() - started) if args.build_timeout else None
        if remaining is not None and remaining <= 0:
            raise WorkflowError('构建预算耗尽；原始日志保留，请新建准备目录。')
        timeout = remaining if limit is None else min(remaining, limit) if remaining is not None else limit
        return recorder.require(label, argv, cwd, timeout, env)

    try:
        manifest['environment'] = doctor(root)
        models_dir = (root / args.models_dir).resolve()
        manifest['model_inventory'] = [{'path': str(p), 'size_bytes': p.stat().st_size}
                                       for p in sorted(models_dir.glob('*.gguf')) if p.is_file()]
        require_host(machine)
        if machine == '300':
            from lab_tools.machines import rocm_report
            manifest['environment']['rocm'] = rocm_report(rocm_home(args.rocm_home))
            require_rocm(manifest['environment']['rocm'])
        for tool in ('git', 'cmake', *(['make', 'cc', 'c++'] if machine == '300' else [])):
            if not shutil.which(tool):
                raise WorkflowError(f'缺少 {tool}；100 在 Visual Studio C++ Developer PowerShell 中运行。')
        if args.model:
            from lab_tools.machine_launcher import model_record
            manifest['model'] = model_record((root / args.model).resolve(), full=not args.check_only)
        if args.check_only:
            raise WorkflowError('check-only：仅完成盘点；GPU kernel 探针和引擎构建未执行。')
        manifest['status'] = 'preparing'
        if machine == '300':
            home = rocm_home(args.rocm_home)
            env = next(iter(manifest['plans'].values()))['environment']
            probe_source = directory / 'hip-probe.cpp'
            probe_binary = directory / 'hip-probe'
            probe_source.write_text(HIP_PROBE, encoding='utf-8')
            step('hipcc-version', [str(home / 'bin/hipcc'), '--version'], env=env, limit=10)
            step('compile-hip-probe', [str(home / 'bin/hipcc'), '--offload-arch=gfx1151',
                                     str(probe_source), '-o', str(probe_binary)], env=env, limit=60)
            result = step('run-hip-probe', [str(probe_binary)], env=env, limit=15)
            manifest['gpu_smoke_passed'] = True
            manifest['gpu_probe'] = {'binary': str(probe_binary), 'sha256': sha256(probe_binary),
                                     'log': result['log'], 'environment': env}
            recorder.save()
        for engine, plan in manifest['plans'].items():
            source = Path(plan['source'])
            source.parent.mkdir(parents=True, exist_ok=True)
            if args.offline and not (root / 'sources' / SOURCE_NAMES[engine] / '.git').exists():
                raise WorkflowError(f'offline：缺少本地 {engine} 源码；先初始化固定子模块。')
            for command in plan['commands']:
                step(engine + '-' + command['label'], command['argv'], Path(command['cwd']), plan['environment'])
            _reject_hidden_changes(source)
            if _git(source, 'rev-parse', 'HEAD') != plan['commit'] or _git(source, 'status', '--porcelain', '--untracked-files=no'):
                raise WorkflowError(f'{engine} 构建源码发生变化。')
            if engine == 'ds4':
                binaries = [source / name for name in ('ds4', 'ds4-server', 'ds4-bench')]
            else:
                build = Path(plan['build'])
                binaries = []
                for name in ('llama-server', 'llama-cli', 'llama-bench'):
                    found = list(build.rglob(name + ('.exe' if machine == '100' else '')))
                    if len(found) != 1:
                        raise WorkflowError(f'构建后应有一个 {name} 二进制，实际 {len(found)} 个。')
                    binaries += found
                # Bind runtime libraries as well as the server to this preparation.
                binaries += [p for p in build.rglob('*') if p.is_file() and
                             ('.so' in p.name or p.suffix.lower() in ('.dll', '.dylib'))]
            records = [{'path': str(p), 'sha256': sha256(p)} for p in binaries]
            server = next(p for p in binaries if p.name in (engine + '-server', engine + '-server.exe'))
            if machine == '300':
                linkage = step(engine + '-ldd', ['ldd', str(server)], env=plan['environment'], limit=10)
                if 'not found' in (directory / linkage['log']).read_text(errors='replace'):
                    raise WorkflowError(f'{engine} 服务动态库缺失。')
                device = step(engine + '-list-devices', [str(server), '--list-devices'], env=plan['environment'], limit=30) if engine == 'llama' else None
                if device and not any(line.lstrip().startswith('ROCm') for line in (directory / device['log']).read_text(errors='replace').splitlines()):
                    raise WorkflowError('llama-server 未列出 ROCm 设备，不能标记为 prepared。')
            step(engine + '-server-help', [str(server), '--help'], env=plan['environment'], limit=30)
            nested = _submodules(source)
            if any(not item['initialized'] or item['commit'] != item['pinned_commit'] or item['dirty'] for item in nested):
                raise WorkflowError(f'{engine} 嵌套子模块未固定或不干净。')
            manifest['builds'][engine] = {'commit': plan['commit'], 'source': str(source),
                                         'nested_submodules': nested, 'binaries': records,
                                         'server': str(server), 'runtime_environment': plan['environment']}
            recorder.save()
        manifest['status'] = 'prepared'
        return directory, 0
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
        manifest['blockers'].append('用户中断；构建与原始日志保留。')
        return directory, 130
    except (WorkflowError, OSError, ValueError) as error:
        manifest['status'] = 'blocked'
        manifest['blockers'].append(str(error))
        print(str(error), flush=True)
        return directory, 2
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        manifest['duration_seconds'] = round(time.monotonic() - started, 3)
        recorder.save()
