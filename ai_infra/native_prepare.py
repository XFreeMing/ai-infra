"""Plan and build isolated, pinned CUDA environments; no model downloads."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time

from ai_infra.cli import WorkflowError, _git, _reject_hidden_changes
from ai_infra.native_model import MODEL_DIR, PROFILE, inspect_model, profile
from ai_infra.prepare import Recorder, cuda_home, sha256

# Dependency/layout expectations come from these exact gitlinks, not latest docs.
SOURCES = {
    'sglang': {'commit': 'f006fb49ecd93322802ddebff80821f8e6296893',
               'url': 'https://github.com/sgl-project/sglang.git', 'torch': '2.14.1'},
    'vllm': {'commit': '2534a96fcf165202f5358335f8b0705175159fd0',
             'url': 'https://github.com/vllm-project/vllm.git', 'torch': '2.13.0'},
    'freetoken': {'commit': '95ac86f21567bc244c7a9d91972838fbf049b4ed',
                  'url': 'https://github.com/FlashML-org/FreeToken.git', 'torch': '2.14.1'},
}
PYTHON_VERSION = '3.12'
TORCH_BACKEND = 'cu130'
CUDA_ARCH = '12.1a'

PROBE = r'''
import importlib, importlib.metadata, json, os, pathlib, platform, sys, torch
engine, source, output = sys.argv[1:]
source = pathlib.Path(source).resolve()
module = importlib.import_module(engine)
actual = pathlib.Path(module.__file__).resolve()
assert actual.is_relative_to(source), f"wrong import provenance: {actual}"
assert torch.cuda.is_available(), "PyTorch has no accessible CUDA device"
assert torch.version.cuda, "CPU-only PyTorch"
assert torch.__version__.split('+')[0] == {'sglang': '2.14.1', 'vllm': '2.13.0', 'freetoken': '2.14.1'}[engine], "unexpected torch version"
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
    extension = importlib.import_module('sgl_kernel' if engine == 'sglang' else 'vllm._C')
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
        from ai_infra import freetoken_model, serve_freetoken
        path = serve_freetoken.MODEL_PATH
        if args and args.model:
            path = args.model
        elif args and args.models_dir != Path('/data/models'):
            path = args.models_dir / path.name
        return {'path': str(path.absolute()), 'profile': freetoken_model.profile(root),
                'profile_file': str(freetoken_model.PROFILE)}
    return {'path': str(MODEL_DIR), 'profile': profile(root), 'profile_file': str(PROFILE)}


def new_directory(root: Path, suffix: str) -> Path:
    parent = root / 'artifacts/init'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return Path(tempfile.mkdtemp(prefix=f'{stamp}-{suffix}-', dir=parent))


def build_plan(engine: str, directory: Path, home: Path, jobs: int, offline: bool) -> dict:
    source = directory / engine / 'source'
    environment = directory / engine / 'venv'
    python = environment / 'bin/python'
    uv = ['uv', 'pip', 'install', '--python', str(python), '--torch-backend', TORCH_BACKEND]
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
           'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1'}
    venv_args = ['uv', 'venv', '--python', PYTHON_VERSION, str(environment)]
    if offline:
        venv_args.append('--offline')
    commands = [('venv', venv_args)]
    if engine == 'sglang':
        commands += [
            ('build-deps', uv + ['torch==2.14.1', 'scikit-build-core>=0.10', 'cmake>=3.26.1',
                                  'ninja', 'wheel', 'setuptools>=61', 'setuptools-rust>=1.11', 'setuptools-scm>=8']),
            ('kernel-source-build', uv + ['--no-build-isolation',
                                          str(source / 'python/sglang/kernels/aot')]),
            ('framework-source-build', uv + ['--no-build-isolation', '--editable', str(source / 'python')]),
        ]
    elif engine == 'vllm':
        commands += [
            ('build-deps', uv + ['-r', str(source / 'requirements/build/cuda.txt')]),
            ('framework-source-build', uv + ['--no-build-isolation', '--editable', str(source)]),
        ]
    elif engine == 'freetoken':
        cache = directory / engine / 'kernel-cache'
        env.update({'TVM_FFI_CUDA_ARCH_LIST': '12.1', 'FREETOKEN_KERNEL_CACHE_DIR': str(cache),
                    'FREETOKEN_DISABLE_KERNEL_CACHE': '0', 'FREETOKEN_DISABLE_JIT': '0',
                    'FREETOKEN_ALLOW_CUDA_MISMATCH': '0',
                    'TORCH_EXTENSIONS_DIR': str(directory / engine / 'torch-extensions'),
                    'TRITON_CACHE_DIR': str(directory / engine / 'triton-cache')})
        commands += [
            ('build-deps', uv + ['torch==2.14.1', 'setuptools>=77', 'wheel', 'ninja']),
            ('framework-source-build', uv + ['--no-build-isolation', '--editable', str(source) + '[accel]']),
            ('kernel-source-build', [str(python), str(directory / 'build-freetoken-kernels.py'),
                                     str(cache), str(directory / engine / 'kernel-build'),
                                     str(directory / engine / 'kernel-build.json')]),
        ]
    else:
        raise WorkflowError(f'未知原生框架：{engine}')
    commands += [('freeze', ['uv', 'pip', 'freeze', '--python', str(python)]),
                 ('cuda-import-probe', [str(python), str(directory / 'probe.py'), engine,
                                       str(source), str(directory / engine / 'provenance.json')])]
    return {'commit': SOURCES[engine]['commit'], 'source': str(source),
            'python': str(python), 'environment': env,
            'commands': [{'label': label, 'argv': argv} for label, argv in commands]}


def manifest_base(root: Path, directory: Path, args, engines: set[str]) -> dict:
    home = cuda_home(args.cuda_home) or Path('/usr/local/cuda')
    for engine in engines & SOURCES.keys():
        # A newly added submodule is usable before its first parent commit;
        # record the staged gitlink rather than substituting a floating ref.
        ref = f':sources/{engine}' if engine == 'freetoken' else f'HEAD:sources/{engine}'
        pin = _git(root, 'rev-parse', ref)
        if pin != SOURCES[engine]['commit']:
            raise WorkflowError(f'{engine} gitlink 已变化；请先核对依赖与构建布局，再更新 native_prepare.SOURCES。')
    models = {engine: model_configuration(root, engine, args) for engine in sorted(engines & SOURCES.keys())}
    primary = next(iter(models.values())) if models else model_configuration(root, 'sglang')
    model = primary['profile']
    plans = {engine: build_plan(engine, directory, home, args.jobs, args.offline)
             for engine in sorted(engines & SOURCES.keys())}
    return {'schema_version': 1, 'preparation_id': 'native-cuda', 'run_id': directory.name,
            'parent_commit': _git(root, 'rev-parse', 'HEAD'), 'started_at': datetime.now(timezone.utc).isoformat(),
            'selected_engines': sorted(engines), 'status': 'planned', 'commands': [], 'blockers': [],
            'cuda_home': str(home), 'build_timeout_seconds': args.build_timeout,
            'plans': plans, 'frameworks': {}, 'gpu_inference_validated': False,
            'source_pin_refs': {engine: (f':sources/{engine}' if engine == 'freetoken' else f'HEAD:sources/{engine}')
                                for engine in plans},
            'model_download_requested': False, 'model_id': model['model_id'],
            'model_revision': model['revision'], 'model_dir': primary['path'], 'models': models,
            'model_weight_bytes': model['weight_bytes'],
            'tooling_sha256': {str(p.relative_to(root)): sha256(p) for p in (root / 'ai_infra').glob('*.py')},
            'model_profile_sha256': sha256(root / primary['profile_file']),
            'platform_support_notes': ['FreeToken upstream install docs target Linux x86_64; ARM64 source build and CUDA probes must pass.']
                                      if 'freetoken' in engines else [],
            'inherited_build_environment': {name: os.environ[name] for name in
                ('CC', 'CXX', 'CUDAHOSTCXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS',
                 'CMAKE_ARGS', 'CMAKE_PREFIX_PATH', 'LD_LIBRARY_PATH') if name in os.environ},
            'legacy_p0_plan': 'prepare_p0: 固定 GGUF、ds4 两版本、llama.cpp、CUDA probe、scorer、DSpark support'
                               if engines & {'ds4', 'llama'} else None}


def prepare_plan(root: Path, args, engines: set[str]) -> tuple[Path, int]:
    directory = new_directory(root, 'plan')
    manifest = manifest_base(root, directory, args, engines)
    Recorder(root, directory, manifest).save()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return directory, 0


def prepare_native(root: Path, args, engines: set[str]) -> tuple[Path, int]:
    directory = new_directory(root, 'native-cuda')
    manifest = manifest_base(root, directory, args, engines)
    recorder = Recorder(root, directory, manifest)
    recorder.save()
    print(f'原生 CUDA 准备目录：{directory}', flush=True)
    code = 2
    started = time.monotonic()
    try:
        manifest['local_models'] = {}
        for engine, configuration in manifest['models'].items():
            path = Path(configuration['path'])
            expected = configuration['profile']
            if engine == 'freetoken':
                from ai_infra.freetoken_model import MODEL_DIR as FT_MODEL_DIR, inspect_freetoken_model
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
            source.parent.mkdir(parents=True)
            local = root / 'sources' / engine

            def step(label, argv, cwd=root, env=None):
                remaining = args.build_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise WorkflowError('构建墙钟预算已用完；保留日志，下次新建准备目录。')
                return recorder.require(f'{engine}-{label}', argv, cwd, remaining, env)

            if (local / '.git').exists():
                step('clone', ['git', 'clone', '--no-hardlinks', '--no-checkout', str(local), str(source)])
            else:
                if args.offline:
                    raise WorkflowError(f'offline：缺少本地 {engine} 源码。')
                step('git-init', ['git', 'init', str(source)])
                step('remote', ['git', '-C', str(source), 'remote', 'add', 'origin', SOURCES[engine]['url']])
                step('fetch', ['git', '-C', str(source), 'fetch', '--depth=1', 'origin', plan['commit']])
            step('checkout', ['git', '-C', str(source), 'checkout', '--detach', plan['commit']])
            git = ['git', '-C', str(source)]
            if args.offline:
                git += ['-c', 'protocol.allow=never', '-c', 'protocol.file.allow=always']
            step('submodules', git + ['submodule', 'update', '--init', '--recursive'])
            manifest.setdefault('sources', {})[engine] = {
                'commit': _git(source, 'rev-parse', 'HEAD'),
                'nested_submodules': _git(source, 'submodule', 'status', '--recursive'),
                'path': str(source), 'url': SOURCES[engine]['url']}
            for command in plan['commands']:
                step(command['label'], command['argv'], source, plan['environment'])
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
