"""Prepare isolated ROCm runtime environments without loading/downloading models."""
from __future__ import annotations

from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import tomllib

from lab_tools.cli import WorkflowError, _git, doctor
from lab_tools.media_models import APPS, TORCH_VERSION
from lab_tools.media_wheels import MIRROR, wheel_requirements, wheel_sources
from lab_tools.machines import require_host, require_rocm, rocm_home, rocm_report
from lab_tools.native_prepare import default_package_index, initialization_lock
from lab_tools.nltk_prepare import catalog as language_catalog
from lab_tools.prepare import Recorder, sha256


def download_environment() -> dict[str, str]:
    # Limit connection pressure while retaining uv's native progress and cache.
    # HTTP limits apply to individual reads/connections, not the whole init.
    defaults = {'UV_CONCURRENT_DOWNLOADS': '4', 'UV_HTTP_TIMEOUT': '30',
                'UV_HTTP_CONNECT_TIMEOUT': '10', 'UV_HTTP_RETRIES': '2'}
    return {key: os.environ.get(key, value) for key, value in defaults.items()}


def patch_source(source: Path, app: str) -> dict:
    """Recorded adapters redirect only model IO; the neural computations stay upstream."""
    edits = {}
    if app == 'ideogram4':
        edits['src/ideogram4/pipeline_ideogram4.py'] = [
            ('from huggingface_hub import hf_hub_download', 'from lab_tools.media_worker import local_file as hf_hub_download')]
    else:
        edits['pyproject.toml'] = [('"torch==2.8.*"', '"torch==' + TORCH_VERSION + '"'),
                                  ('"torchaudio==2.8.*"', '"torchaudio==' + TORCH_VERSION + '"')]
        edits['indextts/infer_v2.py'] = [
            ('from huggingface_hub import hf_hub_download', 'from lab_tools.media_worker import local_file as hf_hub_download, local_model'),
            ('SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")',
             'SeamlessM4TFeatureExtractor.from_pretrained(local_model("facebook/w2v-bert-2.0"))'),
            ('bigvgan.BigVGAN.from_pretrained(bigvgan_name,', 'bigvgan.BigVGAN.from_pretrained(local_model(bigvgan_name),')]
        edits['indextts/utils/maskgct_utils.py'] = [
            ('from huggingface_hub import hf_hub_download', 'from lab_tools.media_worker import local_file as hf_hub_download, local_model'),
            ('Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")',
             'Wav2Vec2BertModel.from_pretrained(local_model("facebook/w2v-bert-2.0"))')]
    diffs = []
    for name, replacements in edits.items():
        path = source / name
        before = path.read_text(encoding='utf-8')
        after = before
        for old, new in replacements:
            if after.count(old) != 1:
                raise WorkflowError(f'上游适配位置不唯一：{name} / {old}')
            after = after.replace(old, new)
        path.write_text(after, encoding='utf-8')
        diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=name, tofile=name))
    metadata = tomllib.loads((source / 'pyproject.toml').read_text(encoding='utf-8'))
    requirements = [value for value in metadata['project']['dependencies']
                    if not value.split('=', 1)[0].split('>', 1)[0].strip() in ('torch', 'torchaudio', 'transformers')]
    # The pinned Ideogram FP8 checkpoint was saved by Transformers 5.8.0:
    # its tokenizer list and Qwen3-VL rope_parameters are incompatible with 4.x.
    # Gradio 5.49.1 permits the huggingface-hub 1.x required by Transformers 5.
    requirements += ['torch==' + TORCH_VERSION, 'transformers==5.8.0' if app == 'ideogram4' else 'transformers==4.52.1',
                     'gradio==5.49.1' if app == 'ideogram4' else 'gradio==5.45.0', 'soundfile==0.13.1']
    if app == 'indextts2':
        requirements += ['torchaudio==' + TORCH_VERSION]
    # Installing the upstream package without dependencies avoids its cu128 uv source mapping.
    path = source.parent / 'runtime-requirements.txt'
    path.write_text('\n'.join(requirements) + '\n', encoding='utf-8')
    patch = source.parent / 'local-model-io.patch'
    patch.write_text(''.join(diffs), encoding='utf-8')
    return {'path': str(patch), 'sha256': sha256(patch), 'requirements': str(path),
            'requirements_sha256': sha256(path), 'torch_override': {'upstream': '2.8.*' if app == 'indextts2' else '>=2.11',
                                                                  'runtime': TORCH_VERSION}}


def runtime_files(root: Path, runtime: Path) -> dict:
    package = runtime / 'lab_tools'
    package.mkdir(parents=True)
    for source in (root / 'lab_tools').glob('*'):
        if source.is_file() and source.suffix in ('.py', '.json'):
            shutil.copyfile(source, package / source.name)
    return {str(path): sha256(path) for path in package.glob('*') if path.is_file()}


def plan_media(root: Path, app: str, directory: Path, args) -> dict:
    source, runtime = directory / 'source', directory / 'runtime'
    python = directory / 'venv/bin/python'
    worker = runtime / 'lab_tools/media_worker.py'
    package_index = default_package_index(root)
    wheels = wheel_sources(app)
    # --torch-backend overrides package indexes with official overseas URLs.
    # Direct wheel requirements select only the GPU packages; PyPI uses the project index.
    # uv verbose hides download bars; reserve it for the editable source build.
    uv = ['uv', 'pip', 'install', '--python', str(python), '--default-index', package_index]
    if args.offline:
        uv += ['--offline']
    command = [['git', 'init', str(source)],
               ['git', '-C', str(source), 'remote', 'add', 'origin', APPS[app]['repository']],
               ['git', '-C', str(source), 'fetch', '--progress', '--depth=1', 'origin', APPS[app]['commit']],
               ['git', '-C', str(source), 'checkout', '--detach', APPS[app]['commit']],
               ['uv', 'venv', '--python', APPS[app]['python'], *(['--offline'] if args.offline else []), str(directory / 'venv')],
               [*uv, '-r', str(directory / 'runtime-requirements.txt'), *wheel_requirements(wheels)],
               [*uv, '--verbose', '--no-deps', '-e', str(source)],
               [str(python), '-I', str(worker), 'probe', '--app', app, '--source', str(source), '--output', str(directory / 'provenance.json')]]
    home = rocm_home(args.rocm_home)
    return {'app': app, 'source_commit': APPS[app]['commit'], 'source_repository': APPS[app]['repository'],
            'python_version': APPS[app]['python'], 'torch': TORCH_VERSION, 'backend': 'rocm', 'machine_id': '300',
            'python': str(python), 'source': str(source), 'worker': str(worker), 'model_download_requested': False,
            'model_loading_requested': False, 'commands': command, 'package_default_index': package_index,
            'torch_wheel_mirror': MIRROR, 'rocm_wheel_sources': wheels,
            'language_runtime_resources': language_catalog() if app == 'indextts2' else None,
            'language_runtime_command': ([str(python), '-I', str(runtime / 'lab_tools/nltk_prepare.py'),
                '--directory', str(directory / 'nltk_data'), '--cache-dir', str(root / '.cache/nltk'),
                *(['--offline'] if args.offline else [])] if app == 'indextts2' else None),
            'environment': {**download_environment(),
                            'HIP_VISIBLE_DEVICES': '0', 'ROCM_PATH': str(home), 'HIP_PATH': str(home),
                            'UV_DEFAULT_INDEX': package_index, 'PIP_INDEX_URL': package_index,
                            'NINJA_STATUS': '[%f/%t %p | %e s] ',
                            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONNOUSERSITE': '1',
                            'GIT_LFS_SKIP_SMUDGE': '1',
                            'GRADIO_ANALYTICS_ENABLED': 'False', 'NLTK_DATA': str(directory / 'nltk_data')}}


def prepare_media(root: Path, app: str, args) -> tuple[Path, int]:
    if args.resume or args.model or args.frozen_p0 or args.cuda_home or args.update_sources:
        raise WorkflowError('生图/配音 init 只准备固定源码和运行环境；不接受 model/resume/frozen-p0/cuda-home/update-sources。')
    base = root / 'artifacts/init/300'
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + app + '-', dir=base))
    plan = plan_media(root, app, directory, args)
    manifest = {**plan, 'schema_version': 1, 'preparation_id': 'media-runtime', 'run_id': directory.name,
                'parent_commit': _git(root, 'rev-parse', 'HEAD'), 'parent_dirty': bool(_git(root, 'status', '--porcelain')),
                'status': 'planned', 'started_at': datetime.now(timezone.utc).isoformat(), 'commands': [], 'plan': plan,
                'build_timeout_seconds': args.build_timeout, 'gpu_inference_validated': False, 'blockers': []}
    recorder = Recorder(root, directory, manifest)
    recorder.save()
    print(f'{APPS[app]["name"]} 运行环境准备：{directory}；不下载或加载模型。', flush=True)
    print(f'普通 Python 包源：{plan["package_default_index"]}\nROCm wheel 源：{plan["torch_wheel_mirror"]}', flush=True)
    print(f'下载并发：{plan["environment"]["UV_CONCURRENT_DOWNLOADS"]}；'
          f'HTTP 读取超时：{plan["environment"]["UV_HTTP_TIMEOUT"]} 秒；'
          f'请求重试：{plan["environment"]["UV_HTTP_RETRIES"]} 次。', flush=True)
    for wheel in plan['rocm_wheel_sources']:
        print(f'{wheel["package"]}：{wheel["url"]}', flush=True)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return directory, 0
    started = time.monotonic()

    def step(label, command, *, limit=None):
        remaining = None if args.build_timeout is None else args.build_timeout - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            raise WorkflowError('运行环境准备预算耗尽；日志保留，请新建准备。')
        timeout = remaining if limit is None else min(remaining, limit) if remaining is not None else limit
        return recorder.require(label, command, root, timeout, plan['environment'])

    try:
        require_host('300')
        manifest['environment_report'] = doctor(root)
        manifest['environment_report']['rocm'] = rocm_report(rocm_home(args.rocm_home))
        require_rocm(manifest['environment_report']['rocm'])
        if args.check_only:
            raise WorkflowError('check-only：盘点完成；未安装依赖，未检查实际 PyTorch/import。')
        with initialization_lock(root, args.build_timeout):
            manifest['status'] = 'preparing'
            source = Path(plan['source'])
            if args.offline:
                # An offline init may clone a clean, exact source acquired during earlier research.
                local = root / 'artifacts/research/ai300-media' / ('ideogram4' if app == 'ideogram4' else 'index-tts-v2')
                if not (local / '.git').exists() or _git(local, 'rev-parse', 'HEAD') != APPS[app]['commit'] or _git(local, 'status', '--porcelain'):
                    raise WorkflowError('offline：缺少完整、干净的固定源码；先在线准备。')
                step('clone-source', ['git', 'clone', '--progress', '--no-hardlinks', str(local), str(source)])
            else:
                for index, command in enumerate(plan['commands'][:4]):
                    step(f'source-{index}', command)
            print('准备源码适配与运行文件快照…', flush=True)
            recorder.progress('准备源码适配与运行文件快照')
            manifest['source_adapter'] = patch_source(source, app)
            manifest['runtime_files'] = runtime_files(root, directory / 'runtime')
            manifest['source_files'] = {str(path): sha256(path) for path in source.rglob('*.py')}
            manifest['source_files'][str(source / 'pyproject.toml')] = sha256(source / 'pyproject.toml')
            manifest['source_patch_diff'] = _git(source, 'diff')
            print(f'源码快照完成：{len(manifest["source_files"])} 个文件。', flush=True)
            recorder.progress(f'源码快照完成：{len(manifest["source_files"])} 个文件')
            for name, command in zip(('venv', 'runtime-dependencies', 'package-install'), plan['commands'][4:7]):
                step(name, command)
            # g2p-en's small lexicon/POS resources are runtime language data, not TTS checkpoints.
            if app == 'indextts2':
                step('language-runtime-data', plan['language_runtime_command'])
                manifest['language_data_files'] = {str(p): sha256(p) for p in (directory / 'nltk_data').rglob('*') if p.is_file()}
            step('rocm-runtime-import-probe', plan['commands'][7], limit=60)
            manifest['provenance'] = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
            torch_dir = Path(manifest['provenance']['torch_path']).parent
            print('校验 PyTorch 动态库身份，大文件将显示字节进度…', flush=True)
            recorder.progress('校验 PyTorch 动态库身份')
            manifest['runtime_extensions'] = {str(p): sha256(p) for p in torch_dir.glob('_C*.so')}
            manifest['runtime_extensions'].update({str(p): sha256(p) for p in (torch_dir / 'lib').glob('lib*torch*.so')})
            freeze = step('freeze', ['uv', 'pip', 'freeze', '--python', plan['python']], limit=30)
            (directory / 'freeze.txt').write_bytes((directory / freeze['log']).read_bytes())
            manifest['freeze_sha256'] = sha256(directory / 'freeze.txt')
            manifest['status'] = 'prepared'
            print(f'{APPS[app]["name"]} 运行环境初始化完成。', flush=True)
            recorder.progress(f'{APPS[app]["name"]} 运行环境初始化完成')
            return directory, 0
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
        return directory, 130
    except (WorkflowError, OSError, ValueError, subprocess.SubprocessError) as error:
        manifest['status'] = 'blocked'
        manifest['blockers'].append(str(error))
        print(f'环境未就绪：{error}', flush=True)
        return directory, 2
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        manifest['duration_seconds'] = round(time.monotonic() - started, 3)
        recorder.save()
