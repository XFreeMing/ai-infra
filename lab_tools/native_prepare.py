"""Plan and build isolated, pinned CUDA environments; no model downloads."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
import time
import tomllib
from contextlib import contextmanager

from lab_tools.cli import WorkflowError, _git, _reject_hidden_changes
from lab_tools.native_model import MODEL_KEY, MODEL_DIR, PROFILE, inspect_model, profile
from lab_tools.prepare import Recorder, cuda_home, sha256
from lab_tools.source_tracking import source_revision

# Audited defaults for planning without a local checkout. Actual init snapshots
# the selected checkout and reads its build requirements before installing.
SOURCES = {
    'sglang': {'commit': '684fea8af3b11a5b51ec9966c02959a32ad3421c',
               'url': 'https://github.com/sgl-project/sglang.git', 'torch': '2.14.1'},
    'vllm': {'commit': 'd5e62627a3ece4b2d045cbb2b22ac26c061c8313',
             'url': 'https://github.com/vllm-project/vllm.git', 'torch': '2.13.0'},
    'freetoken': {'commit': '9c635a157798da4aa43e6d051e92620521ed1858',
                  'url': 'https://github.com/FlashML-org/FreeToken.git', 'torch': '2.14.1'},
}
PYTHON_VERSION = '3.12'
TORCH_BACKEND = 'cu130'
CUDA_ARCH = '12.1a'
DEFAULT_PYPI_INDEX = 'https://pypi.tuna.tsinghua.edu.cn/simple'


@contextmanager
def initialization_lock(root: Path, timeout: int | None):
    """Serialize builds in this workspace without touching uv's shared cache."""
    import fcntl
    path = root / '.cache/native-init.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    announced = started - 20
    with path.open('a') as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if timeout is not None and now - started >= timeout:
                    raise WorkflowError('等待其他 native init 完成时耗尽构建预算。')
                if now - announced >= 20:
                    print(f'等待其他 native init 完成：已等待 {now - started:.0f} 秒；Ctrl+C 可中断。', flush=True)
                    announced = now
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_step(recorder: Recorder, label: str, argv: list[str], cwd: Path,
             started: float, budget: int | None, env: dict | None = None) -> dict:
    while True:
        remaining = None if budget is None else budget - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            raise WorkflowError('构建墙钟预算已用完；日志已保留，可使用 --resume。')
        record = recorder.command(label, argv, cwd, remaining, env)
        if record['exit_code'] == 0:
            return record
        output = (recorder.directory / record['log']).read_text(errors='replace')
        # UV_LOCK_TIMEOUT=0 means immediate timeout, not unlimited. Retry only
        # this idempotent cache-lock failure; network/compiler failures still stop.
        if (argv[:3] == ['uv', 'pip', 'install'] and record['exit_code'] == 2
                and all(marker in output for marker in
                        ('Failed to acquire lock', 'Timeout (', 'UV_LOCK_TIMEOUT'))):
            record['retry_reason'] = 'uv distribution cache lock timeout'
            recorder.save()
            print('uv 缓存仍被其他进程占用，继续等待并重试；未清理缓存或终止其他进程。', flush=True)
            continue
        raise WorkflowError(f'{label} 失败，查看 {recorder.directory / record["log"]}')


def elfutils_headers(source: Path, step, recorder: Recorder, env: dict, offline: bool) -> dict:
    """Extract matching Ubuntu/Debian dev packages locally; no sudo/system install."""
    if Path('/usr/include/elfutils/libdwfl.h').is_file() and Path('/usr/include/libelf.h').is_file():
        step('elfutils-header-probe', ['c++', '-x', 'c++', '-fsyntax-only',
                                       '-include', 'elfutils/libdwfl.h', '/dev/null'], source, env)
        return {'origin': 'system', 'include': '/usr/include'}
    prefix = source.parent / 'elfutils'
    packages = prefix / 'packages'
    packages.mkdir(parents=True, exist_ok=True)
    for name in ('libdw', 'libelf'):
        version_record = step(f'{name}-runtime-version',
                              ['dpkg-query', '-W', '-f=${Version}', f'{name}1t64'])
        version = (recorder.directory / version_record['log']).read_text().strip()
        # Packages must match the installed runtime's exact version.
        if not version or any(c.isspace() for c in version):
            raise WorkflowError(f'无法核实 {name} runtime 版本。')
        matches = list(packages.glob(f'{name}-dev_*.deb'))
        if not matches:
            if offline:
                raise WorkflowError(f'offline：缺少 {name}-dev={version}，请先在线初始化或安装对应开发包。')
            step(f'{name}-headers-download', ['apt-get', 'download', f'{name}-dev={version}'], packages)
            matches = list(packages.glob(f'{name}-dev_*.deb'))
        if len(matches) != 1:
            raise WorkflowError(f'{name} 开发包数量异常：{packages}')
        metadata = step(f'{name}-headers-version', ['dpkg-deb', '-f', str(matches[0]), 'Version'])
        if (recorder.directory / metadata['log']).read_text().strip() != version:
            raise WorkflowError(f'{name} 开发包与系统 runtime 版本不一致。')
        step(f'{name}-headers-extract', ['dpkg-deb', '-x', str(matches[0]), str(prefix / 'sysroot')])
    include = prefix / 'sysroot/usr/include'
    for header in ('elfutils/libdwfl.h', 'libelf.h'):
        if not (include / header).is_file():
            raise WorkflowError(f'缺少构建头文件：{include / header}')
    env['CPATH'] = str(include) + (':' + os.environ['CPATH'] if os.environ.get('CPATH') else '')
    probe = prefix / 'probe.cc'
    probe.write_text('#include <elfutils/libdwfl.h>\nint main() { return 0; }\n')
    step('elfutils-header-probe', ['c++', '-fsyntax-only', str(probe)], source, env)
    return {'origin': 'local deb extraction', 'include': str(include),
            'packages': [{'path': str(p), 'sha256': sha256(p)} for p in sorted(packages.glob('*.deb'))]}


def default_package_index(root: Path) -> str:
    config_path = root / 'pyproject.toml'
    if not config_path.is_file():
        return DEFAULT_PYPI_INDEX
    config = tomllib.loads(config_path.read_text())
    indexes = config.get('tool', {}).get('uv', {}).get('index', [])
    defaults = [entry['url'] for entry in indexes if entry.get('default')]
    if len(defaults) > 1:
        raise WorkflowError('pyproject.toml 只能配置一个 default package index。')
    return defaults[0] if defaults else DEFAULT_PYPI_INDEX

PROBE = r'''
import importlib, importlib.metadata, json, os, pathlib, platform, sys, torch
engine, source, output, torch_requirement = sys.argv[1:]
source = pathlib.Path(source).resolve()
module = importlib.import_module(engine)
actual = pathlib.Path(module.__file__).resolve()
assert actual.is_relative_to(source), f"wrong import provenance: {actual}"
assert torch.cuda.is_available(), "PyTorch has no accessible CUDA device"
assert torch.version.cuda, "CPU-only PyTorch"
from packaging.requirements import Requirement
assert Requirement(torch_requirement).specifier.contains(torch.__version__.split('+')[0], prereleases=True), "unexpected torch version"
assert torch.cuda.get_device_capability(0) == (12, 1), "expected GB10 sm_121"
tensor = torch.arange(8, device='cuda')
result = tensor.sum().item()
torch.cuda.synchronize()
assert result == 28
store_correct = None
if engine == 'freetoken':
    extension = importlib.import_module('freetoken.kernel._pinned_tensor')
    for name in ('_cpu_moe', '_row_store'):
        importlib.import_module('freetoken.kernel.' + name)
    importlib.import_module('sgl_kernel')
    importlib.import_module('flashinfer')
    from freetoken.kernel.aot import DEFAULT_STORE_ELEMENT_SIZES
    from freetoken.kernel.store import store_cache
    os.environ['FREETOKEN_DISABLE_JIT'] = '1'
    width = DEFAULT_STORE_ELEMENT_SIZES[0] // 2
    k = torch.arange(2 * width, dtype=torch.float32, device='cuda').reshape(2, width).to(torch.bfloat16)
    v = k + 1
    indices = torch.tensor([1, 3], device='cuda', dtype=torch.int64)
    kc = torch.zeros((4, width), device='cuda', dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    store_cache(kc, vc, indices, k, v)
    torch.cuda.synchronize()
    store_correct = bool(torch.equal(kc[indices], k) and torch.equal(vc[indices], v))
    assert store_correct, 'FreeToken CUDA store_cache output mismatch'
else:
    extension = importlib.import_module('sgl_kernel' if engine == 'sglang' else 'vllm._C_stable_libtorch')
extension_root = pathlib.Path(extension.__file__).resolve()
if engine == 'freetoken':
    paths = sorted((source / 'python/freetoken').rglob('*.so'))
    paths += sorted(pathlib.Path(os.environ['FREETOKEN_KERNEL_CACHE_DIR']).rglob('*.so'))
else:
    paths = sorted((extension_root.parent if engine == 'sglang' else source / 'vllm').rglob('*.so'))
assert paths, "no native extension files"
record = {'engine': engine, 'module_path': str(actual), 'extension_import_path': str(extension_root),
          'version': importlib.metadata.version(engine), 'python': sys.executable,
          'platform': platform.platform(), 'torch': torch.__version__, 'torch_path': torch.__file__,
          'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
          'capability': list(torch.cuda.get_device_capability(0)), 'cuda_tensor_result': result,
          'freetoken_store_kernel_correct': store_correct,
          'extensions': [{'path': str(p), 'size_bytes': p.stat().st_size} for p in paths],
          'gpu_inference_validated': False}
pathlib.Path(output).write_text(json.dumps(record, indent=2))
'''

FREETOKEN_KERNEL_BUILDER = r'''
import json, pathlib, sys
from freetoken.kernel.aot import compile_and_package_kernels
out, build, receipt = map(pathlib.Path, sys.argv[1:])
# Each compiler already uses MAX_JOBS; serialize specs to avoid nested parallelism.
paths = compile_and_package_kernels(out_dir=out, build_dir=build, jobs=1, verbose=True)
receipt.write_text(json.dumps({'kernel_count': len(paths), 'paths': list(map(str, paths))}, indent=2))
'''


def model_configuration(root: Path, engine: str, args=None) -> dict:
    if engine == 'freetoken':
        from lab_tools import freetoken_model, freetoken_models
        selected = freetoken_models.selected_model()
        path = selected.model_path
        if args and args.model:
            path = args.model
        elif args and args.models_dir != Path('/data/models'):
            path = args.models_dir / path.name
        frozen = selected.model_path.resolve() == freetoken_model.MODEL_DIR.resolve()
        model_profile = freetoken_model.profile(root) if frozen else {
            'model_id': selected.model_id, 'revision': None, 'weight_bytes': None}
        return {'path': str(path.absolute()), 'model_key': selected.model.value, 'profile': model_profile,
                'profile_file': str(freetoken_model.PROFILE) if frozen else None}
    return {'path': str(MODEL_DIR), 'model_key': MODEL_KEY.value,
            'profile': profile(root), 'profile_file': str(PROFILE)}


def new_directory(root: Path, suffix: str) -> Path:
    parent = root / 'artifacts/init'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return Path(tempfile.mkdtemp(prefix=f'{stamp}-{suffix}-', dir=parent))


def torch_requirement(engine: str, source: Path) -> str:
    if engine == 'vllm':
        text = (source / 'requirements/build/cuda.txt').read_text()
        requirements = [line.strip().split('#')[0].strip() for line in text.splitlines()]
    else:
        path = source / ('python/pyproject.toml' if engine == 'sglang' else 'pyproject.toml')
        requirements = tomllib.loads(path.read_text())['build-system']['requires']
    matches = [item for item in requirements if re.match(r'^torch(?=[<>=!~ ;]|$)', item)]
    if len(matches) != 1:
        raise WorkflowError(f'{engine} 的上游 torch 构建要求不唯一；需核对新版布局。')
    return matches[0]


def build_plan(engine: str, directory: Path, home: Path, jobs: int, offline: bool,
               package_index: str | None = None, revision: str | None = None,
               torch_spec: str | None = None) -> dict:
    source = directory / engine / 'source'
    environment = directory / engine / 'venv'
    python = environment / 'bin/python'
    torch_spec = torch_spec or 'torch==' + SOURCES[engine]['torch']
    package_index = package_index or default_package_index(Path(__file__).resolve().parents[1])
    # Child commands run inside upstream clones, whose uv configuration may
    # differ from the parent workspace. Pass the parent's default explicitly.
    # uv verbose hides download bars; enable it only for source-build steps.
    uv = ['uv', 'pip', 'install', '--python', str(python), '--default-index', package_index,
          '--torch-backend', TORCH_BACKEND]
    if offline:
        uv.append('--offline')
    env = {'CUDA_HOME': str(home), 'CUDACXX': str(home / 'bin/nvcc'),
           'PATH': str(environment / 'bin') + ':' + str(home / 'bin') + ':' + os.environ.get('PATH', ''),
           'TORCH_CUDA_ARCH_LIST': CUDA_ARCH, 'MAX_JOBS': str(jobs), 'NVCC_THREADS': '1',
           'CMAKE_BUILD_PARALLEL_LEVEL': str(jobs), 'CUDA_VISIBLE_DEVICES': '0',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'VLLM_TARGET_DEVICE': 'cuda', 'VLLM_USE_PRECOMPILED': '0',
           'VLLM_PRECOMPILED_WHEEL_LOCATION': '', 'VLLM_DOCKER_BUILD_CONTEXT': '0',
           'VLLM_USE_PRECOMPILED_RUST': '0', 'SGLANG_BUILD_RUST_EXTS': 'none',
           'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1', 'UV_LOCK_TIMEOUT': '3600',
           'NINJA_STATUS': '[%f/%t %p | %e s] '}
    venv_args = ['uv', 'venv', '--python', PYTHON_VERSION, str(environment)]
    if offline:
        venv_args.append('--offline')
    commands = [('venv', venv_args)]
    if engine == 'sglang':
        commands += [
            ('build-deps', uv + [torch_spec, 'scikit-build-core>=0.10', 'cmake>=3.26.1',
                                  'ninja', 'wheel', 'setuptools>=61', 'setuptools-rust>=1.11', 'setuptools-scm>=8']),
            ('kernel-source-build', uv + ['--verbose', '--no-build-isolation-package', 'sglang-kernel',
                                          str(source / 'python/sglang/kernels/aot')]),
            ('framework-source-build', uv + ['--verbose', '--no-build-isolation-package', 'sglang', '--editable', str(source / 'python')]),
        ]
    elif engine == 'vllm':
        env['DIST_EXTRA_CONFIG'] = str(directory / engine / 'build-config.cfg')
        commands += [
            ('build-deps', uv + ['-r', str(source / 'requirements/build/cuda.txt')]),
            ('framework-source-build', uv + ['--verbose', '--no-build-isolation-package', 'vllm', '--editable', str(source)]),
        ]
    elif engine == 'freetoken':
        cache = directory / engine / 'kernel-cache'
        env.update({'TVM_FFI_CUDA_ARCH_LIST': '12.1', 'FREETOKEN_KERNEL_CACHE_DIR': str(cache),
                    'FREETOKEN_DISABLE_KERNEL_CACHE': '0', 'FREETOKEN_DISABLE_JIT': '0',
                    'FREETOKEN_ALLOW_CUDA_MISMATCH': '0',
                    'TORCH_EXTENSIONS_DIR': str(directory / engine / 'torch-extensions'),
                    'TRITON_CACHE_DIR': str(directory / engine / 'triton-cache')})
        commands += [
            ('build-deps', uv + [torch_spec, 'setuptools>=77', 'wheel', 'ninja']),
            ('framework-source-build', uv + ['--verbose', '--no-build-isolation-package', 'freetoken', '--editable', str(source) + '[accel]']),
            ('kernel-source-build', [str(python), '-I', str(directory / 'build-freetoken-kernels.py'),
                                     str(cache), str(directory / engine / 'kernel-build'),
                                     str(directory / engine / 'kernel-build.json')]),
        ]
    else:
        raise WorkflowError(f'未知原生框架：{engine}')
    commands += [('freeze', ['uv', 'pip', 'freeze', '--python', str(python)]),
                 ('cuda-import-probe', [str(python), '-I', str(directory / 'probe.py'), engine,
                                       str(source), str(directory / engine / 'provenance.json'), torch_spec])]
    command_records = [{'label': label, 'argv': argv} for label, argv in commands]
    if engine == 'sglang':
        for command in command_records:
            if command['label'] == 'kernel-source-build':
                command['environment'] = {'SKBUILD_CMAKE_DEFINE': 'SGL_KERNEL_COMPILE_THREADS=1',
                                          'SKBUILD_BUILD_DIR': str(directory / engine / 'kernel-build-cache')}
    return {'commit': revision or SOURCES[engine]['commit'], 'source': str(source),
            'torch_requirement': torch_spec, 'package_default_index': package_index,
            'python': str(python), 'environment': env,
            'commands': command_records}


def manifest_base(root: Path, directory: Path, args, engines: set[str]) -> dict:
    home = cuda_home(args.cuda_home) or Path('/usr/local/cuda')
    revisions = {engine: source_revision(root, engine, _git) for engine in engines & SOURCES.keys()}
    models = {engine: model_configuration(root, engine, args) for engine in sorted(engines & SOURCES.keys())}
    primary = next(iter(models.values())) if models else model_configuration(root, 'sglang')
    model = primary['profile']
    package_index = default_package_index(root)
    plans = {engine: build_plan(engine, directory, home, args.jobs, args.offline, package_index,
                               revisions[engine][0],
                               torch_requirement(engine, root / 'sources' / engine)
                               if (root / 'sources' / engine / '.git').exists() else None)
             for engine in sorted(engines & SOURCES.keys())}
    return {'schema_version': 1, 'preparation_id': 'native-cuda', 'machine_id': '700', 'backend': 'cuda', 'run_id': directory.name,
            'parent_commit': _git(root, 'rev-parse', 'HEAD'), 'started_at': datetime.now(timezone.utc).isoformat(),
            'selected_engines': sorted(engines), 'status': 'planned', 'commands': [], 'blockers': [],
            'cuda_home': str(home), 'build_timeout_seconds': args.build_timeout,
            'package_default_index': package_index, 'torch_backend': TORCH_BACKEND,
            'plans': plans, 'frameworks': {}, 'gpu_inference_validated': False,
            'source_pin_refs': {engine: revisions[engine][1] for engine in plans},
            'model_download_requested': False, 'model_key': primary.get('model_key'), 'model_id': model['model_id'],
            'model_revision': model['revision'], 'model_dir': primary['path'], 'models': models,
            'model_weight_bytes': model['weight_bytes'],
            'tooling_sha256': {str(p.relative_to(root)): sha256(p) for p in (root / 'lab_tools').glob('*.py')},
            'model_profile_sha256': sha256(root / primary['profile_file']) if primary['profile_file'] else None,
            'platform_support_notes': ['FreeToken upstream install docs target Linux x86_64; ARM64 source build and CUDA probes must pass.']
                                      if 'freetoken' in engines else [],
            'inherited_build_environment': {name: os.environ[name] for name in
                ('CC', 'CXX', 'CUDAHOSTCXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS',
                 'CMAKE_ARGS', 'CMAKE_PREFIX_PATH', 'LD_LIBRARY_PATH') if name in os.environ},
            'legacy_p0_plan': (('prepare_p0 --frozen-p0: 固定 GGUF、历史 ds4 两版本、llama.cpp、CUDA probe、scorer、DSpark support'
                                if args.frozen_p0 else
                                'prepare_p0: 固定 GGUF、当前 ds4/llama.cpp、CUDA probe；不下载 support，不构建历史 scorer')
                               if engines & {'ds4', 'llama'} else None)}


def prepare_plan(root: Path, args, engines: set[str]) -> tuple[Path, int]:
    directory = new_directory(root, 'plan')
    manifest = manifest_base(root, directory, args, engines)
    Recorder(root, directory, manifest).save()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return directory, 0


def prepare_native(root: Path, args, engines: set[str]) -> tuple[Path, int]:
    if args.check_only:
        if args.resume:
            raise WorkflowError('--check-only 请单独执行，不要与 --resume 合用。')
        return _prepare_native(root, args, engines)
    started = time.monotonic()
    try:
        with initialization_lock(root, args.build_timeout):
            return _prepare_native(root, args, engines, started)
    except KeyboardInterrupt:
        print('等待初始化锁时已中断；原准备目录未修改。', flush=True)
        return args.resume.resolve() if args.resume else root / 'artifacts/init', 130


def _prepare_native(root: Path, args, engines: set[str], started: float | None = None) -> tuple[Path, int]:
    directory = args.resume.resolve() if args.resume else new_directory(root, 'native-cuda')
    manifest = manifest_base(root, directory, args, engines)
    previous = None
    if args.resume:
        if not directory.is_relative_to((root / 'artifacts/init').resolve()):
            raise WorkflowError('--resume 必须指定此 workspace 的 artifacts/init 目录。')
        previous = json.loads((directory / 'manifest.json').read_text())
        if previous.get('preparation_id') != 'native-cuda' or previous.get('status') not in ('blocked', 'interrupted'):
            raise WorkflowError('native --resume 仅接受 blocked/interrupted 原生准备目录。')
        if previous['selected_engines'] != sorted(engines) or previous['cuda_home'] != manifest['cuda_home']:
            raise WorkflowError('--resume 的引擎和 CUDA_HOME 必须与原准备一致。')
        if any(previous['plans'][e]['commit'] != manifest['plans'][e]['commit'] for e in engines):
            raise WorkflowError('--resume 源码版本不一致。')
        attempt = len(previous.get('resume_history', [])) + 1
        snapshot = directory / f'manifest-before-resume-{attempt:02d}-{len(previous["commands"]):04d}.json'
        if snapshot.exists():
            raise WorkflowError(f'历史快照已存在：{snapshot}')
        snapshot.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + '\n')
        manifest['commands'] = previous['commands']
        manifest['resume_history'] = previous.get('resume_history', []) + [str(snapshot)]
        # Preserve successfully prepared engines, whose provenance is rechecked below.
        manifest['frameworks'] = previous.get('frameworks', {})
        for engine, framework in manifest['frameworks'].items():
            for extension in framework['extensions']:
                if sha256(Path(extension['path'])) != extension['sha256']:
                    raise WorkflowError(f'已完成的 {engine} 扩展 hash 变化，不能恢复。')
    recorder = Recorder(root, directory, manifest)
    recorder.save()
    print(f'原生 CUDA 准备目录：{directory}', flush=True)
    code = 2
    started = time.monotonic() if started is None else started
    try:
        manifest['local_models'] = {}
        for engine, configuration in manifest['models'].items():
            path = Path(configuration['path'])
            expected = configuration['profile']
            if engine == 'freetoken':
                from lab_tools.freetoken_model import MODEL_DIR as FT_MODEL_DIR, inspect_freetoken_model
                inspector = inspect_freetoken_model
                expected = expected if path.resolve() == FT_MODEL_DIR.resolve() else None
            else:
                inspector = inspect_model
            try:
                manifest['local_models'][engine] = inspector(path, expected)
            except WorkflowError as error:
                manifest['local_models'][engine] = {'ready': False, 'reason': str(error)}
        if args.check_only:
            manifest['status'] = 'inventory_only'
            return directory, 2
        if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
            raise WorkflowError('此构建配置针对 Linux ARM64 GB10；其他平台需先调整 CUDA_ARCH。')
        home = Path(manifest['cuda_home'])
        if not (home / 'bin/nvcc').is_file():
            raise WorkflowError('找不到 CUDA nvcc；请设置 --cuda-home。')
        for tool in ('uv', 'git', 'cmake', 'c++'):
            if not shutil.which(tool):
                raise WorkflowError(f'缺少构建工具：{tool}')
        recorder.require('nvcc-version', [str(home / 'bin/nvcc'), '--version'], root)
        (directory / 'probe.py').write_text(PROBE)
        (directory / 'build-freetoken-kernels.py').write_text(FREETOKEN_KERNEL_BUILDER)
        manifest['status'] = 'preparing'
        for engine in sorted(engines):
            plan = manifest['plans'][engine]
            source = Path(plan['source'])
            source.parent.mkdir(parents=True, exist_ok=True)
            local = root / 'sources' / engine

            def step(label, argv, cwd=root, env=None):
                return run_step(recorder, f'{engine}-{label}', argv, cwd, started, args.build_timeout, env)

            if source.exists():
                if not previous or _git(source, 'rev-parse', 'HEAD') != plan['commit']:
                    raise WorkflowError(f'无法复用源码目录：{source}')
                _reject_hidden_changes(source)
                if _git(source, 'status', '--porcelain', '--untracked-files=no'):
                    raise WorkflowError(f'恢复前发现 tracked 源码修改：{source}')
            elif (local / '.git').exists():
                step('clone', ['git', 'clone', '--progress', '--no-hardlinks', '--no-checkout', str(local), str(source)])
            else:
                if args.offline:
                    raise WorkflowError(f'offline：缺少本地 {engine} 源码。')
                step('git-init', ['git', 'init', str(source)])
                step('remote', ['git', '-C', str(source), 'remote', 'add', 'origin', SOURCES[engine]['url']])
                step('fetch', ['git', '-C', str(source), 'fetch', '--progress', '--depth=1', 'origin', plan['commit']])
            step('checkout', ['git', '-C', str(source), 'checkout', '--detach', plan['commit']])
            if torch_requirement(engine, source) != plan['torch_requirement']:
                raise WorkflowError(f'{engine} torch 要求与计划不一致；初始化本地子模块后重新生成计划。')
            git = ['git', '-C', str(source)]
            if args.offline:
                git += ['-c', 'protocol.allow=never', '-c', 'protocol.file.allow=always']
            step('submodules', git + ['submodule', 'update', '--progress', '--init', '--recursive'])
            manifest.setdefault('sources', {})[engine] = {
                'commit': _git(source, 'rev-parse', 'HEAD'),
                'nested_submodules': _git(source, 'submodule', 'status', '--recursive'),
                'path': str(source), 'url': SOURCES[engine]['url']}
            if engine == 'vllm' and not manifest['frameworks'].get(engine, {}).get('prepared'):
                manifest['elfutils'] = elfutils_headers(source, step, recorder, plan['environment'], args.offline)
                # setuptools editable builds normally delete their temporary
                # CMake tree on failure. Retain object files for subsequent resumes.
                config = source.parent / 'build-config.cfg'
                # setuptools-rust renames vLLM's command when wrapping it;
                # editable_wheel reinitializes it under that class name.
                # Avoid [build_ext], which would also affect third-party builds.
                config.write_text(''.join(f'[{name}]\nbuild_temp = {source.parent / "build-cache"}\n'
                                          for name in ('cmake_build_ext', 'build_ext_rust_extension')))
                manifest['build_config_sha256'] = sha256(config)
                recorder.save()
            for command in plan['commands']:
                if command['label'] == 'venv' and previous and Path(plan['python']).is_file():
                    continue
                if previous and manifest['frameworks'].get(engine, {}).get('prepared'):
                    if command['label'] not in ('freeze', 'cuda-import-probe'):
                        continue
                step(command['label'], command['argv'], source,
                     {**plan['environment'], **command.get('environment', {})})
            _reject_hidden_changes(source)
            if _git(source, 'status', '--porcelain', '--untracked-files=no'):
                raise WorkflowError(f'{engine} 构建过程中 tracked 源码发生修改；保存失败日志并停止。')
            provenance = json.loads((source.parent / 'provenance.json').read_text())
            if engine == 'freetoken':
                if provenance.get('freetoken_store_kernel_correct') is not True:
                    raise WorkflowError('FreeToken 自有 CUDA store kernel smoke 未通过。')
                provenance['runtime_environment'] = {name: plan['environment'][name] for name in
                    ('CUDA_HOME', 'PATH', 'TVM_FFI_CUDA_ARCH_LIST', 'FREETOKEN_KERNEL_CACHE_DIR',
                     'FREETOKEN_DISABLE_KERNEL_CACHE', 'FREETOKEN_DISABLE_JIT', 'FREETOKEN_ALLOW_CUDA_MISMATCH',
                     'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR')}
            for extension in provenance['extensions']:
                extension['sha256'] = sha256(Path(extension['path']))
            manifest['frameworks'][engine] = {**provenance, 'source': str(source),
                                             'commit': plan['commit'], 'prepared': True}
            recorder.save()
        manifest['status'] = 'prepared'
        code = 0
    except (WorkflowError, OSError, ValueError, KeyError) as error:
        manifest['status'] = 'blocked'
        manifest['blockers'].append(str(error))
        print(f'准备未完成：{error}', flush=True)
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
        manifest['blockers'].append('用户中断；保留构建日志。')
        code = 130
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        recorder.save()
        (directory / 'README.md').write_text(
            '# 原生 CUDA 框架准备\n\n'
            f'状态：{manifest["status"]}，详见 manifest.json 和逐步骤日志。\n\n'
            '已理解：框架各有固定源码与独立 venv；模型下载必须另行手动执行。\n\n'
            '未解决：CUDA/import/kernel 小探针不证明完整模型加载、全部 kernel 与输出正确性。\n\n'
            '下一步：核对每个模型的容量、格式与 kernel 支持，准备完整权重，再串行执行 32-token 模型 smoke。\n')
    return directory, code
