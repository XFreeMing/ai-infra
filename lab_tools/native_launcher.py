"""Launch locally staged native models in the corresponding isolated CUDA venv."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from lab_tools.cli import WorkflowError, _git, _reject_hidden_changes
from lab_tools.native_model import MODEL_KEY, MODEL_DIR, inspect_model, profile
from lab_tools.prepare import sha256
from lab_tools.source_tracking import source_revision


def positive_port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError('端口必须在 1–65535。')
    return number


def resolve_python(root: Path, engine: str, preparation_dir: Path | None,
                   explicit: Path | None) -> tuple[Path, dict | None]:
    if explicit:
        return explicit.absolute(), None
    directories = [(root / preparation_dir).resolve()] if preparation_dir else sorted(
        (root / 'artifacts/init').glob('*'), reverse=True)
    pin, _ = source_revision(root, engine, _git)
    for directory in directories:
        path = directory / 'manifest.json'
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        record = manifest.get('frameworks', {}).get(engine, {})
        if record.get('prepared') and record.get('commit') == pin:
            # Do not resolve the venv Python symlink: doing so selects the base
            # interpreter and loses the isolated environment's site-packages.
            python = Path(record['python']).absolute()
            if not python.is_file():
                raise WorkflowError(f'已准备的 Python 环境不存在：{python}')
            if _git(Path(record['source']), 'rev-parse', 'HEAD') != pin:
                raise WorkflowError('准备好的框架源码 commit 已变化。')
            _reject_hidden_changes(Path(record['source']))
            if _git(Path(record['source']), 'status', '--porcelain', '--untracked-files=no'):
                raise WorkflowError('准备好的框架存在 tracked 源码修改；请新建准备目录。')
            for extension in record.get('extensions', []):
                if sha256(Path(extension['path'])) != extension['sha256']:
                    raise WorkflowError(f'CUDA 扩展发生变化：{extension["path"]}')
            return python, {'manifest': str(path), 'framework': record}
    raise WorkflowError(f'尚未准备 {engine}；先运行 uv run init --engines {engine}。')


def capacity_error(gpu: str, weight_bytes: int, host_memory: int) -> str | None:
    if 'GB10' in gpu and weight_bytes >= host_memory:
        return (f'GB10 容量不足：权重 {weight_bytes / 1024**3:.1f} GiB，'
                f'本机总内存 {host_memory / 1024**3:.1f} GiB；减小 context 无法解决权重容量。')
    return None


def effective_arguments(arguments: list[str], engine: str, port: int, extra: list[str]) -> tuple[list[str], Path, int]:
    command = [*arguments, '--port', str(port), *extra]
    model_flags = {'--model-path'} if engine == 'sglang' else {'--model'}
    if engine == 'freetoken':
        model_flags.add('--model-path')
    path = None
    tp = 1
    for index, value in enumerate(command):
        key, sep, inline = value.partition('=')
        if key in model_flags | {'--tensor-parallel-size', '--tp-size', '--tp'}:
            if not sep and index + 1 == len(command):
                raise WorkflowError(f'{key} 缺少值。')
            argument = inline if sep else command[index + 1]
            if key in model_flags:
                path = Path(argument).absolute()
                if sep:
                    command[index] = key + '=' + str(path)
                else:
                    command[index + 1] = str(path)
            else:
                tp = int(argument)
    if path is None or tp < 1:
        raise WorkflowError('需要本地模型目录以及正整数 tensor parallel size。')
    return command, path, tp


def launch(engine: str, arguments: list[str], *, port: int, cuda_devices: str,
           python: Path | None, preparation_dir: Path | None, configuration_file: Path,
           runtime_environment: dict | None = None,
           configuration_files: tuple[Path, ...] = (),
           argv: list[str] | None = None, root: Path | None = None) -> int:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    parser = argparse.ArgumentParser(prog=engine, description='从独立 CUDA 环境启动本地模型；不会下载权重。')
    parser.add_argument('port', type=positive_port, nargs='?', default=port)
    parser.add_argument('--dry-run', action='store_true', help='显示启动计划；不要求已下载权重/已构建框架。')
    parser.add_argument('--server-help', action='store_true', help='显示已准备的上游服务帮助，不加载模型。')
    raw = list(sys.argv[1:] if argv is None else argv)
    separator = raw.index('--') if '--' in raw else len(raw)
    args = parser.parse_args(raw[:separator])
    extra = raw[separator + 1:]
    try:
        arguments, model, tp = effective_arguments(arguments, engine, args.port, extra)
        env = {'CUDA_VISIBLE_DEVICES': cuda_devices, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1'}
        prepared = None
        unresolved = None
        try:
            executable, prepared = resolve_python(root, engine, preparation_dir, python)
        except WorkflowError as error:
            if not args.dry_run:
                raise
            executable = root / f'artifacts/init/<prepared-run>/{engine}/venv/bin/python'
            unresolved = str(error)
        modules = {'sglang': 'sglang.launch_server', 'vllm': 'vllm.entrypoints.openai.api_server',
                   'freetoken': 'freetoken.cli'}
        module = modules[engine]
        prefix = ['serve'] if engine == 'freetoken' else []
        command = [str(executable), '-I', '-m', module, *prefix, *arguments]
        if prepared:
            env = {**prepared['framework'].get('runtime_environment', {}), **env}
        env.update(runtime_environment or {})
        if engine == 'freetoken':
            from lab_tools.freetoken_model import MODEL_DIR as DEFAULT_MODEL, inspect_freetoken_model, profile as model_profile
            inspector = inspect_freetoken_model
        else:
            DEFAULT_MODEL, inspector, model_profile = MODEL_DIR, inspect_model, profile
        default_profile = model_profile(root)
        default_model_id = default_profile['model_id']
        default_weight_bytes = default_profile['weight_bytes']
        default_model_key = MODEL_KEY.value
        default_model_path = DEFAULT_MODEL
        if engine == 'freetoken':
            from lab_tools.freetoken_models import selected_model
            selected = selected_model()
            default_model_id = selected.model_id
            default_model_key = selected.model.value
            default_model_path = selected.model_path
            if selected.model_path.resolve() != DEFAULT_MODEL.resolve():
                default_weight_bytes = None
        expected = default_profile if model.resolve() == DEFAULT_MODEL.resolve() else None
        if args.dry_run:
            try:
                model_record = inspector(model, expected)
            except WorkflowError as error:
                model_record = {'ready': False, 'reason': str(error)}
            print(json.dumps({'engine': engine, 'argv': command, 'environment': env,
                              'framework_blocker': unresolved, 'model': model_record,
                              'default_model_id': default_model_id,
                              'default_model_key': default_model_key,
                              'default_weight_bytes': default_weight_bytes,
                              'gpu_inference_validated': False}, ensure_ascii=False, indent=2))
            print(shlex.join(command))
            return 0
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise WorkflowError(f'Python 不存在或不可执行：{executable}')
        full_env = {**os.environ, **env}
        if args.server_help:
            return subprocess.run([str(executable), '-I', '-m', module, *prefix, '--help'], env=full_env, timeout=60).returncode
        model_record = inspector(model, expected)
        probe = '''import importlib,json,pathlib,sys,torch
module=importlib.import_module(sys.argv[1])
assert torch.cuda.is_available() and torch.version.cuda, 'CUDA PyTorch required'
assert torch.cuda.device_count() >= int(sys.argv[2]), 'tensor parallel exceeds visible CUDA devices'
if sys.argv[1]=='freetoken':
    for name in ('_pinned_tensor','_cpu_moe','_row_store'):
        importlib.import_module('freetoken.kernel.'+name)
else:
    importlib.import_module('sgl_kernel' if sys.argv[1]=='sglang' else 'vllm._C_stable_libtorch')
x=torch.arange(8,device='cuda'); result=x.sum().item(); torch.cuda.synchronize(); assert result==28
print(json.dumps({'gpu':torch.cuda.get_device_name(0),'torch_cuda':torch.version.cuda,
                  'module_path':str(pathlib.Path(module.__file__).resolve()),'cuda_tensor_result':result}))'''
        checked = subprocess.run([str(executable), '-I', '-c', probe, engine, str(tp)], env=full_env,
                                 capture_output=True, text=True, timeout=60)
        if checked.returncode:
            raise WorkflowError(f'启动前 CUDA/import 检查失败：{checked.stderr[-4000:]}')
        gpu = json.loads(checked.stdout.strip().splitlines()[-1])
        if prepared and gpu['module_path'] != prepared['framework']['module_path']:
            raise WorkflowError('实际 import path 与准备记录不同。')
        meminfo = Path('/proc/meminfo')
        memory = next((int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                       if line.startswith('MemTotal:')), 0) if meminfo.is_file() else 0
        # FreeToken V4.1 streams Engram tables; file bytes are not all resident.
        streaming_engram = engine == 'freetoken' and model_record['model_type'] == 'deepseek_v41'
        failure = capacity_error(gpu['gpu'], model_record['weight_bytes'], memory) if memory and not streaming_engram else None
        if failure:
            raise WorkflowError(failure)
        maximum = model_record['max_position_embeddings']
        length_flag = {'sglang': '--context-length', 'vllm': '--max-model-len',
                       'freetoken': '--max-seq-len-override'}[engine]
        length = None
        for index, value in enumerate(arguments):
            key, sep, inline = value.partition('=')
            if key == length_flag:
                length = int(inline if sep else arguments[index + 1])
        if length is not None and (length < 1 or (maximum and length > maximum)):
            raise WorkflowError(f'context {length} 超出模型支持范围：1–{maximum}。')
        parent = root / 'artifacts/servers'
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory = Path(tempfile.mkdtemp(prefix=f'{stamp}-{engine}-', dir=parent))
        receipt = {'engine': engine, 'parent_commit': _git(root, 'rev-parse', 'HEAD'),
                   'status': 'launch_requested', 'argv': command, 'environment': env,
                   'prepared': prepared, 'model': model_record,
                   'model_key': default_model_key if model.resolve() == default_model_path.resolve() else None,
                   'cuda_probe': gpu,
                   'configuration_sha256': sha256(configuration_file),
                   'configuration_files_sha256': {str(path.absolute()): sha256(path)
                                                  for path in (configuration_file, *configuration_files)},
                   'gpu_inference_validated': False}
        (directory / 'launch.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        print(f'启动记录：{directory / "launch.json"}', flush=True)
        os.execve(executable, command, full_env)
    except (WorkflowError, OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
        print(f'无法启动 {engine}：{error}', file=sys.stderr)
        return 2
    return 0
