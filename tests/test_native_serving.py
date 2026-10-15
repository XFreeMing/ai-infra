"""Admission and isolation regressions; these tests never build frameworks or load GPUs."""
import argparse
from dataclasses import replace
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv
from unittest.mock import patch

from lab_tools import native_launcher as launcher, native_model as model, native_prepare as prep
from lab_tools import prepare, serve_sglang, serve_vllm, serve_freetoken, freetoken_model
from lab_tools.cli import WorkflowError
from lab_tools import source_tracking
from lab_tools import freetoken_models
from lab_tools.model_catalog import ModelKey, MODEL_CATALOG, get_model


class NativeServingTests(unittest.TestCase):
    def test_isolated_probe_avoids_preparation_directory_shadowing_editable_package(self):
        environment = self.root / 'isolated-venv'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        site = environment / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        package = self.root / 'source/fixture_engine'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('value = 42\n')
        (site / 'fixture_finder.py').write_text(
            'import importlib.abc, importlib.util, sys\n'
            'class Finder(importlib.abc.MetaPathFinder):\n'
            ' def find_spec(self, fullname, path=None, target=None):\n'
            f'  if fullname == "fixture_engine": return importlib.util.spec_from_file_location(fullname, {str(package / "__init__.py")!r})\n'
            'sys.meta_path.append(Finder())\n')
        (site / 'fixture_editable.pth').write_text('import fixture_finder\n')
        preparation = self.root / 'preparation'
        (preparation / 'fixture_engine').mkdir(parents=True)
        probe = preparation / 'probe.py'
        probe.write_text(f'import fixture_engine\nassert fixture_engine.__file__ == {str(package / "__init__.py")!r}\n')
        python = environment / 'bin/python'
        plain = subprocess.run([str(python), str(probe)], capture_output=True, text=True)
        isolated = subprocess.run([str(python), '-I', str(probe)], capture_output=True, text=True)
        self.assertNotEqual(plain.returncode, 0)
        self.assertEqual(isolated.returncode, 0, isolated.stderr)

    def test_cache_lock_failure_retries_but_compiler_failure_stops(self):
        recorder = prep.Recorder(self.root, self.root, {'commands': []})
        log = self.root / 'lock.log'
        log.write_text('Failed to acquire lock\nTimeout (3600s)\nUV_LOCK_TIMEOUT')
        failure = {'exit_code': 2, 'log': log.name}
        success = {'exit_code': 0}
        with patch.object(recorder, 'command', side_effect=[failure, success]) as command, \
             patch.object(recorder, 'save'), redirect_stdout(io.StringIO()):
            result = prep.run_step(recorder, 'deps', ['uv', 'pip', 'install', 'torch'],
                                   self.root, 0, None)
        self.assertEqual(result, success)
        self.assertEqual(command.call_count, 2)
        self.assertIn('retry_reason', failure)
        log.write_text('fatal error: missing compiler header')
        with patch.object(recorder, 'command', return_value=failure) as command:
            with self.assertRaises(WorkflowError):
                prep.run_step(recorder, 'deps', ['uv', 'pip', 'install', 'torch'], self.root, 0, None)
        self.assertEqual(command.call_count, 1)

    def test_cache_lock_retries_respect_explicit_outer_budget(self):
        recorder = prep.Recorder(self.root, self.root, {'commands': []})
        log = self.root / 'lock.log'
        log.write_text('Failed to acquire lock\nTimeout (300s)\nUV_LOCK_TIMEOUT')
        with patch.object(recorder, 'command', return_value={'exit_code': 2, 'log': log.name}) as command, \
             patch.object(recorder, 'save'), patch.object(prep.time, 'monotonic', side_effect=[1, 11]), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(WorkflowError, '预算'):
                prep.run_step(recorder, 'deps', ['uv', 'pip', 'install', 'torch'], self.root, 0, 10)
        self.assertEqual(command.call_count, 1)

    def test_native_resume_refuses_prepared_environment_before_mutations(self):
        directory = self.root / 'artifacts/init/ready'
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps({'preparation_id': 'native-cuda', 'status': 'prepared'}))
        args = self.parser.parse_args(['--engines', 'sglang', '--resume', str(directory)])
        with patch.object(prep, '_git', side_effect=self.git), patch.object(prep.Recorder, 'command') as command:
            with self.assertRaisesRegex(WorkflowError, 'blocked/interrupted'):
                prep._prepare_native(self.root, args, {'sglang'})
        command.assert_not_called()
        self.assertEqual(json.loads((directory / 'manifest.json').read_text())['status'], 'prepared')

    def test_native_resume_preserves_failure_logs_and_reuses_venv(self):
        directory = self.root / 'artifacts/init/blocked'
        directory.mkdir(parents=True)
        args = self.parser.parse_args(['--engines', 'sglang', '--resume', str(directory), '--cuda-home', '/cuda'])
        with patch.object(prep, '_git', side_effect=self.git):
            manifest = prep.manifest_base(self.root, directory, args, {'sglang'})
        manifest['status'] = 'blocked'
        manifest['commands'] = [{'label': 'original-failure', 'exit_code': 2, 'log': '00-original.log'}]
        manifest['blockers'] = ['original failure']
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        (directory / '00-original.log').write_text('keep failure evidence')
        plan = manifest['plans']['sglang']
        source = Path(plan['source'])
        source.mkdir(parents=True)
        (source / 'python').mkdir()
        (source / 'python/pyproject.toml').write_text('[build-system]\nrequires = ["torch==2.14.1"]\n')
        python = Path(plan['python'])
        python.parent.mkdir(parents=True)
        python.write_text('fixture venv')
        provenance = {'extensions': []}
        (source.parent / 'provenance.json').write_text(json.dumps(provenance))
        def git(root, *args):
            if root == source:
                return plan['commit'] if args == ('rev-parse', 'HEAD') else ''
            return self.git(root, *args)
        is_file = Path.is_file
        with patch.object(prep, '_git', side_effect=git), patch.object(prep, '_reject_hidden_changes'), \
             patch.object(prep.platform, 'system', return_value='Linux'), \
             patch.object(prep.platform, 'machine', return_value='aarch64'), \
             patch.object(prep.shutil, 'which', return_value='/bin/tool'), \
             patch.object(prep.Recorder, 'require'), patch.object(prep, 'run_step') as step, \
             patch.object(prep.Path, 'is_file', lambda path: path.name == 'nvcc' or is_file(path)), \
             redirect_stdout(io.StringIO()):
            _, code = prep._prepare_native(self.root, args, {'sglang'})
        self.assertEqual(code, 0)
        labels = [call.args[1] for call in step.call_args_list]
        self.assertNotIn('sglang-venv', labels)
        self.assertNotIn('sglang-clone', labels)
        self.assertIn('sglang-build-deps', labels)
        self.assertIn('sglang-cuda-import-probe', labels)
        self.assertEqual((directory / '00-original.log').read_text(), 'keep failure evidence')
        snapshot = json.loads(next(directory.glob('manifest-before-resume-*.json')).read_text())
        self.assertEqual(snapshot['blockers'], ['original failure'])
        self.assertEqual(json.loads((directory / 'manifest.json').read_text())['status'], 'prepared')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'model'
        self.directory.mkdir()
        self.config = {'model_type': 'deepseek_v41', 'architectures': ['DeepseekV41ForCausalLM'],
                       'quantization_config': {'quant_method': 'fp8', 'moe_quant_algo': 'NVFP4'},
                       'text_config': {'max_position_embeddings': 1048576}}
        (self.directory / 'config.json').write_text(json.dumps(self.config))
        (self.directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'one.safetensors'}}))
        (self.directory / 'one.safetensors').write_bytes(b'fixture-weights')
        (self.directory / 'tokenizer.json').write_text('{}')
        (self.directory / 'tokenizer_config.json').write_text('{}')
        self.expected = {'model_id': model.MODEL_ID, 'revision': model.MODEL_REVISION,
                         'model_type': 'deepseek_v41', 'moe_quant_algo': 'NVFP4', 'weight_bytes': 15,
                         'files': [{'path': p.name, 'size_bytes': p.stat().st_size,
                                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                   for p in self.directory.iterdir()]}
        path = self.root / model.PROFILE
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.expected))
        (self.root / 'lab_tools').mkdir()
        self.parser = argparse.ArgumentParser()
        prepare.add_init_arguments(self.parser)

    def git(self, root, *args):
        if args[-1].startswith(('HEAD:sources/', ':sources/')):
            return prep.SOURCES[args[-1].split('/')[-1]]['commit']
        return 'parent'

    def test_updated_gitlink_is_snapshotted_without_a_hardcoded_commit_gate(self):
        args = self.parser.parse_args(['--engines', 'sglang', '--dry-run'])
        latest = 'f' * 40
        with patch.object(prep, '_git', side_effect=lambda root, *a: latest if a[-1] == ':sources/sglang' else 'parent'):
            manifest = prep.manifest_base(self.root, self.root / 'run', args, {'sglang'})
        self.assertEqual(manifest['plans']['sglang']['commit'], latest)
        self.assertEqual(manifest['source_pin_refs']['sglang'], ':sources/sglang')

    def test_torch_requirement_comes_from_each_upstream_build_configuration(self):
        source = self.root / 'upstream'
        (source / 'python').mkdir(parents=True)
        (source / 'requirements/build').mkdir(parents=True)
        (source / 'python/pyproject.toml').write_text('[build-system]\nrequires=["wheel", "torch==2.15.2"]\n')
        (source / 'pyproject.toml').write_text('[build-system]\nrequires=["torch>=2.15,<2.16", "setuptools"]\n')
        (source / 'requirements/build/cuda.txt').write_text('# new upstream version\ntorch==2.15.1\nwheel\n')
        for engine, expected in [('sglang','torch==2.15.2'), ('freetoken','torch>=2.15,<2.16'), ('vllm','torch==2.15.1')]:
            self.assertEqual(prep.torch_requirement(engine, source), expected)
        (source / 'requirements/build/cuda.txt').write_text('wheel\n')
        with self.assertRaisesRegex(WorkflowError, '不唯一'):
            prep.torch_requirement('vllm', source)

    def test_branch_update_fast_forwards_and_refuses_local_edits(self):
        def git(path, *argv):
            return subprocess.run(['git', '-C', str(path), *argv], capture_output=True, text=True, check=True).stdout.strip()
        upstream = self.root / 'remote'
        upstream.mkdir()
        git(upstream, 'init', '-b', 'main')
        git(upstream, 'config', 'user.name', 'Fixture')
        git(upstream, 'config', 'user.email', 'fixture@example.invalid')
        (upstream / 'version').write_text('one')
        git(upstream, 'add', '.')
        git(upstream, 'commit', '-m', 'one')
        git(self.root, 'init')
        (self.root / '.gitmodules').write_text('[submodule "sources/fixture"]\npath = sources/fixture\nbranch = main\n')
        local = self.root / 'sources/fixture'
        subprocess.run(['git', 'clone', str(upstream), str(local)], capture_output=True, check=True)
        (upstream / 'version').write_text('two')
        git(upstream, 'commit', '-am', 'two')
        with redirect_stdout(io.StringIO()):
            source_tracking.update_sources(self.root)
        self.assertEqual(git(local, 'rev-parse', 'HEAD'), git(upstream, 'rev-parse', 'HEAD'))
        self.assertEqual(git(local, 'branch', '--show-current'), 'main')
        (local / 'version').write_text('local edit')
        before = git(local, 'rev-parse', 'HEAD')
        with self.assertRaisesRegex(WorkflowError, '本地修改'):
            source_tracking.update_sources(self.root)
        self.assertEqual(git(local, 'rev-parse', 'HEAD'), before)
        self.assertEqual((local / 'version').read_text(), 'local edit')

    def freetoken_fixture(self):
        self.config = {'model_type': 'qwen3_5_moe', 'architectures': ['Qwen3_5MoeForConditionalGeneration'],
                       'text_config': {'dtype': 'bfloat16', 'max_position_embeddings': 262144}}
        (self.directory / 'config.json').write_text(json.dumps(self.config))
        expected = {**self.expected, 'model_id': 'Qwen/Qwen3.6-35B-A3B', 'revision': None,
                    'model_type': 'qwen3_5_moe', 'moe_quant_algo': None}
        expected['files'] = [{'path': p.name, 'size_bytes': p.stat().st_size,
                             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                            for p in self.directory.iterdir()]
        path = self.root / freetoken_model.PROFILE
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(expected))
        return expected

    def test_freetoken_init_before_parent_commit_and_all_plan_never_build_or_download(self):
        self.freetoken_fixture()
        for engines, option, code in [(['freetoken'], '--dry-run', 0),
                                     (['freetoken'], '--check-only', 2), (['all'], '--dry-run', 0)]:
            with self.subTest(engines=engines, option=option), \
                 patch.object(prep, '_git', side_effect=self.git) as git, \
                 patch.dict(freetoken_models.MODELS, {freetoken_models.DEFAULT_MODEL: replace(
                     freetoken_models.selected_model(), model_path_override=self.directory)}), \
                 patch.object(freetoken_model, 'MODEL_DIR', self.directory), \
                 patch.object(prep.Recorder, 'command') as command, \
                 patch.object(prepare, 'prepare_p0') as p0, \
                 patch.object(prepare.urllib.request, 'urlopen') as network, redirect_stdout(io.StringIO()):
                directory, result = prepare.prepare(self.root, self.parser.parse_args(['--engines', *engines, option]))
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertEqual(result, code)
            self.assertIn('freetoken', manifest['plans'])
            self.assertEqual(manifest['models']['freetoken']['profile']['model_id'], 'Qwen/Qwen3.6-35B-A3B')
            self.assertFalse(manifest['model_download_requested'])
            self.assertIsNone(manifest['build_timeout_seconds'])
            self.assertIn(unittest.mock.call(self.root, 'rev-parse', ':sources/freetoken'), git.call_args_list)
            if engines == ['all']:
                self.assertEqual(set(manifest['plans']), {'freetoken', 'sglang', 'vllm'})
            command.assert_not_called()
            p0.assert_not_called()
            network.assert_not_called()

    def test_freetoken_build_uses_own_cuda_kernel_cache_and_accel_without_nightly_wheels(self):
        plan = prep.build_plan('freetoken', self.root, Path('/cuda'), 4, True)
        args = next(x['argv'] for x in plan['commands'] if x['label']=='framework-source-build')
        self.assertIn(str(self.root / 'freetoken/source') + '[accel]', args)
        self.assertIn('--offline', args)
        self.assertEqual(plan['environment']['TVM_FFI_CUDA_ARCH_LIST'], '12.1')
        self.assertEqual(plan['environment']['FREETOKEN_ALLOW_CUDA_MISMATCH'], '0')
        cache = Path(plan['environment']['FREETOKEN_KERNEL_CACHE_DIR'])
        self.assertTrue(cache.is_relative_to(self.root / 'freetoken'))
        self.assertFalse(cache.is_relative_to(Path(plan['source'])))
        kernel = next(x['argv'] for x in plan['commands'] if x['label']=='kernel-source-build')
        self.assertIn(str(cache), kernel)
        self.assertFalse(any('nightly' in arg for x in plan['commands'] for arg in x['argv']))

    def test_freetoken_rejects_dsv41_nvfp4_before_cuda_or_server_execution(self):
        path = self.root / freetoken_model.PROFILE
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.expected))
        with patch.object(launcher, 'resolve_python', return_value=(Path('/bin/true'), None)), \
             patch.object(launcher.subprocess, 'run') as gpu, patch.object(launcher.os, 'execve') as execute, \
             redirect_stderr(io.StringIO()):
            code = launcher.launch('freetoken', ['--model', str(self.directory)], port=6666, cuda_devices='0',
                                   python=None, preparation_dir=None, configuration_file=Path(__file__),
                                   root=self.root, argv=[])
        self.assertEqual(code, 2)
        gpu.assert_not_called()
        execute.assert_not_called()
        with self.assertRaisesRegex(WorkflowError, 'GGUF'):
            freetoken_model.inspect_freetoken_model(Path('/model.gguf'))

    def test_freetoken_entry_point_foreground_serve_preserves_kernel_cache_and_effective_model(self):
        self.freetoken_fixture()
        probe = {'gpu': 'NVIDIA GB10', 'torch_cuda': '13.0',
                 'module_path': '/source/freetoken/__init__.py', 'cuda_tensor_result': 28}
        prepared = {'framework': {'module_path': probe['module_path'],
                                  'runtime_environment': {'FREETOKEN_KERNEL_CACHE_DIR': '/prepared/cache',
                                                          'CUDA_HOME': '/cuda'}}}
        with patch.object(launcher, 'resolve_python', return_value=(Path('/bin/true'), prepared)), \
             patch.object(launcher.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(probe), '')), \
             patch.object(launcher, '_git', return_value='parent'), \
             patch.object(launcher.os, 'execve') as execute, redirect_stdout(io.StringIO()):
            code = launcher.launch('freetoken', ['--model', '/original', '--max-seq-len-override', '4096'],
                                   port=6666, cuda_devices='0', python=None, preparation_dir=None,
                                   configuration_file=Path(__file__), root=self.root,
                                   argv=['6667', '--', '--model-path', str(self.directory)])
        self.assertEqual(code, 0)
        _, command, environment = execute.call_args.args
        self.assertEqual(command[:5], ['/bin/true', '-I', '-m', 'freetoken.cli', 'serve'])
        self.assertIn('6667', command)
        self.assertEqual(environment['FREETOKEN_KERNEL_CACHE_DIR'], '/prepared/cache')
        self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
        receipt = json.loads(next((self.root / 'artifacts/servers').glob('*/launch.json')).read_text())
        self.assertEqual(receipt['model']['path'], str(self.directory))
        self.assertFalse(receipt['gpu_inference_validated'])

    def test_freetoken_server_help_does_not_require_weights(self):
        self.freetoken_fixture()
        with patch.object(launcher, 'resolve_python', return_value=(Path('/bin/true'), None)), \
             patch.object(launcher.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as command:
            code = launcher.launch('freetoken', ['--model', '/missing'], port=6666, cuda_devices='0',
                                   python=None, preparation_dir=None, configuration_file=Path(__file__),
                                   root=self.root, argv=['--server-help'])
        self.assertEqual(code, 0)
        self.assertEqual(command.call_args.args[0], ['/bin/true', '-I', '-m', 'freetoken.cli', 'serve', '--help'])

    def test_freetoken_dry_run_uses_qwen_config_with_default_port(self):
        self.freetoken_fixture()
        output = io.StringIO()
        with patch.object(launcher, 'resolve_python', side_effect=WorkflowError('not built')), \
             patch.object(launcher.os, 'execve') as execute, redirect_stdout(output):
            code = launcher.launch('freetoken', ['--model', str(self.directory)], port=6666, cuda_devices='0',
                                   python=None, preparation_dir=None, configuration_file=Path(__file__),
                                   root=self.root, argv=['--dry-run'])
        self.assertEqual(code, 0)
        self.assertIn('Qwen/Qwen3.6-35B-A3B', output.getvalue())
        self.assertIn('6666', output.getvalue())
        self.assertIn('freetoken.cli', output.getvalue())
        execute.assert_not_called()

    def test_freetoken_model_profile_is_local_bf16_with_unknown_remote_revision(self):
        root = Path(__file__).resolve().parents[1]
        frozen = freetoken_model.profile(root)
        self.assertEqual(frozen['weight_bytes'], 71903776776)
        self.assertEqual(frozen['dtype'], 'bfloat16')
        self.assertIsNone(frozen['revision'])
        self.assertFalse(frozen['full_weights_sha256_checked'])
        self.assertEqual(freetoken_models.selected_model().model_path, freetoken_model.MODEL_DIR)

    def test_freetoken_default_selection_changes_model_and_kv_geometry_together(self):
        self.freetoken_fixture()
        key = ModelKey.QWEN__QWEN3_8_FLASH_NEXT__FP8_BLOCK128__SAFETENSORS
        config = replace(freetoken_models.MODELS[key], model_path_override=self.directory, context_size=262144)
        output = io.StringIO()
        with patch.object(freetoken_models, 'DEFAULT_MODEL', key), \
             patch.dict(freetoken_models.MODELS, {key: config}), \
             patch.object(launcher, 'resolve_python', side_effect=WorkflowError('not built')), \
             patch.object(launcher.os, 'execve') as execute, redirect_stdout(output):
            code = serve_freetoken.main(['7777', '--dry-run'])
            planned = prep.model_configuration(self.root, 'freetoken')
        self.assertEqual(code, 0)
        record = json.JSONDecoder().raw_decode(output.getvalue())[0]
        argv = record['argv']
        self.assertEqual(argv[argv.index('--model') + 1], str(self.directory))
        self.assertEqual(argv[argv.index('--max-seq-len-override') + 1], '262144')
        self.assertEqual(argv[argv.index('--page-size') + 1], '64')
        self.assertEqual(argv[argv.index('--num-pages') + 1], '4096')
        self.assertEqual(argv[argv.index('--ple-backend') + 1], 'disk')
        self.assertEqual(argv[argv.index('--port') + 1], '7777')
        self.assertEqual(record['default_model_id'], config.model_id)
        self.assertEqual(record['default_model_key'], key.value)
        self.assertIsNone(record['default_weight_bytes'])
        self.assertEqual(planned['path'], str(self.directory))
        self.assertEqual(planned['profile']['model_id'], config.model_id)
        self.assertIsNone(planned['profile_file'])
        execute.assert_not_called()

    def test_freetoken_init_plan_uses_selected_model_without_bf16_profile_label(self):
        key = ModelKey.QWEN__QWEN3_8_FLASH_NEXT__FP8_BLOCK128__SAFETENSORS
        output = io.StringIO()
        with patch.object(freetoken_models, 'DEFAULT_MODEL', key), \
             patch.object(prep, '_git', side_effect=self.git), \
             patch.object(prep.Recorder, 'command') as command, redirect_stdout(output):
            directory, code = prepare.prepare(self.root, self.parser.parse_args(
                ['--engines', 'freetoken', '--dry-run']))
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(code, 0)
        self.assertEqual(manifest['model_dir'], str(freetoken_models.MODELS[key].model_path))
        self.assertEqual(manifest['model_id'], freetoken_models.MODELS[key].model_id)
        self.assertEqual(manifest['model_key'], key.value)
        self.assertIsNone(manifest['model_weight_bytes'])
        self.assertIsNone(manifest['model_profile_sha256'])
        command.assert_not_called()

    def test_unknown_freetoken_default_fails_before_launch(self):
        output = io.StringIO()
        with patch.object(freetoken_models, 'DEFAULT_MODEL', 'missing-model'), \
             patch.object(launcher, 'launch') as launch, redirect_stderr(output):
            code = serve_freetoken.main(['--dry-run'])
        self.assertEqual(code, 2)
        self.assertIn('未知 FreeToken 默认模型', output.getvalue())
        launch.assert_not_called()

    def test_model_catalog_is_shared_by_native_engines_and_matches_frozen_identity(self):
        self.assertEqual(set(MODEL_CATALOG), set(ModelKey))
        for key in ModelKey:
            self.assertIs(ModelKey(key.value), key)
        key = ModelKey.NV_COMMUNITY__DEEPSEEK_V4_1_FLASH__FP8_NVFP4__SAFETENSORS
        spec = get_model(key)
        frozen = model.profile(Path(__file__).resolve().parents[1])
        self.assertEqual(spec.repository_id, frozen['model_id'])
        self.assertEqual(spec.revision, frozen['revision'])
        self.assertEqual(spec.quantization_method, frozen['quant_method'])
        self.assertEqual(spec.model_path, serve_sglang.MODEL_PATH)
        self.assertEqual(spec.model_path, serve_vllm.MODEL_PATH)
        for engine in ('sglang', 'vllm'):
            configuration = prep.model_configuration(Path(__file__).resolve().parents[1], engine)
            self.assertEqual(configuration['model_key'], key.value)
            self.assertEqual(configuration['path'], str(spec.model_path))
        self.assertIn('fp8_e4m3', spec.weight_precisions)
        self.assertIn('nvfp4', spec.weight_precisions)
        with self.assertRaisesRegex(WorkflowError, 'ModelKey'):
            get_model(key.value)

    def test_freetoken_mismatched_registry_key_fails_before_launch(self):
        bf16 = ModelKey.QWEN__QWEN3_6_35B_A3B__BF16__SAFETENSORS
        fp8 = ModelKey.QWEN__QWEN3_8_FLASH_NEXT__FP8_BLOCK128__SAFETENSORS
        with patch.object(freetoken_models, 'DEFAULT_MODEL', bf16), \
             patch.dict(freetoken_models.MODELS, {bf16: freetoken_models.MODELS[fp8]}), \
             patch.object(launcher, 'launch') as launch, redirect_stderr(io.StringIO()):
            code = serve_freetoken.main(['--dry-run'])
        self.assertEqual(code, 2)
        launch.assert_not_called()

    def test_native_plan_and_inventory_never_execute_or_download(self):
        for option, expected_code, status in [('--dry-run', 0, 'planned'), ('--check-only', 2, 'inventory_only')]:
            with self.subTest(option=option), patch.object(prep, '_git', side_effect=self.git), \
                 patch.object(prep.Recorder, 'command') as command, \
                 patch.object(prepare.urllib.request, 'urlopen') as network, redirect_stdout(io.StringIO()):
                directory, code = prepare.prepare(self.root, self.parser.parse_args(['--engines', 'sglang', 'vllm', option]))
                manifest = json.loads((directory / 'manifest.json').read_text())
                self.assertEqual((code, manifest['status']), (expected_code, status))
                self.assertFalse(manifest['model_download_requested'])
                self.assertFalse(manifest['gpu_inference_validated'])
                command.assert_not_called()
                network.assert_not_called()

    def test_frameworks_use_separate_venvs_full_source_builds_and_cuda(self):
        plans = {e: prep.build_plan(e, self.root, Path('/cuda'), 4, False) for e in prep.SOURCES}
        self.assertNotEqual(plans['sglang']['python'], plans['vllm']['python'])
        for engine, plan in plans.items():
            self.assertEqual(plan['environment']['VLLM_USE_PRECOMPILED'], '0')
            self.assertEqual(plan['environment']['TORCH_CUDA_ARCH_LIST'], '12.1a')
            self.assertEqual(plan['environment']['CUDA_VISIBLE_DEVICES'], '0')
            self.assertIn('%f/%t', plan['environment']['NINJA_STATUS'])
            for item in plan['commands']:
                if item['label'] in ('venv', 'build-deps'):
                    self.assertNotIn('--verbose', item['argv'])
                elif item['argv'][:3] == ['uv', 'pip', 'install']:
                    self.assertIn('--verbose', item['argv'])
            command = next(x['argv'] for x in plan['commands'] if x['label']=='framework-source-build')
            self.assertNotIn('--no-build-isolation', command)
            self.assertEqual(command[command.index('--no-build-isolation-package') + 1], engine)
            self.assertIn('cuda-import-probe', [x['label'] for x in plan['commands']])
        self.assertIn('torch==2.14.1', plans['sglang']['commands'][1]['argv'])
        kernel = next(x for x in plans['sglang']['commands'] if x['label'] == 'kernel-source-build')
        self.assertEqual(kernel['environment']['SKBUILD_CMAKE_DEFINE'], 'SGL_KERNEL_COMPILE_THREADS=1')
        self.assertNotIn('SKBUILD_BUILD_DIR', plans['sglang']['environment'])
        self.assertTrue(any('python/sglang/kernels/aot' in arg for item in plans['sglang']['commands'] for arg in item['argv']))
        self.assertTrue(any('requirements/build/cuda.txt' in arg for item in plans['vllm']['commands'] for arg in item['argv']))

    def test_offline_plan_applies_to_venv_and_every_package_install(self):
        plan = prep.build_plan('vllm', self.root, Path('/cuda'), 4, True)
        for item in plan['commands']:
            if item['label'] in ('venv', 'build-deps', 'framework-source-build'):
                self.assertIn('--offline', item['argv'])

    def test_parent_default_index_is_passed_to_every_upstream_clone_install(self):
        mirror = 'https://mirror.example.invalid/simple'
        (self.root / 'pyproject.toml').write_text(
            f'[[tool.uv.index]]\nname="test-mirror"\nurl="{mirror}"\ndefault=true\n')
        with patch.object(prep, '_git', side_effect=self.git), redirect_stdout(io.StringIO()):
            directory, code = prepare.prepare(self.root, self.parser.parse_args(
                ['--engines', 'sglang', 'vllm', '--dry-run', '--offline']))
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(code, 0)
        self.assertEqual(manifest['package_default_index'], mirror)
        for engine, plan in manifest['plans'].items():
            for command in plan['commands']:
                argv = command['argv']
                if argv[:3] == ['uv', 'pip', 'install']:
                    self.assertEqual(argv[argv.index('--default-index') + 1], mirror)
                    self.assertEqual(argv[argv.index('--torch-backend') + 1], 'cu130')
                    self.assertIn('--offline', argv)

    def test_local_model_shards_and_frozen_identity_are_checked(self):
        record = model.inspect_model(self.directory, self.expected)
        self.assertEqual(record['moe_quant_algo'], 'NVFP4')
        self.assertFalse(record['full_weights_sha256_checked'])
        (self.directory / 'one.safetensors').write_bytes(b'truncated')
        with self.assertRaisesRegex(WorkflowError, '大小'):
            model.inspect_model(self.directory, self.expected)

    def test_shard_directory_escape_and_missing_shards_are_rejected(self):
        for name in ('../outside.safetensors', 'missing.safetensors'):
            with self.subTest(name=name):
                (self.directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': name}}))
                with self.assertRaises(WorkflowError):
                    model.inspect_model(self.directory)

    def test_full_download_verification_detects_same_size_corruption(self):
        receipt = model.verify_download(self.directory, self.expected)
        self.assertTrue(receipt['full_weights_sha256_checked'])
        path = self.directory / 'one.safetensors'
        path.write_bytes(b'X' * path.stat().st_size)
        with self.assertRaisesRegex(WorkflowError, '身份'):
            model.verify_download(self.directory, self.expected)

    def test_manual_download_dry_run_never_imports_sdk_or_downloads(self):
        script_path = Path(__file__).resolve().parents[1] / 'scripts/download_deepseek_v41_flash_nvfp4.py'
        spec = importlib.util.spec_from_file_location('manual_download', script_path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with patch.dict('sys.modules', {'modelscope': None}), redirect_stdout(io.StringIO()):
            self.assertEqual(script.main(['--dry-run']), 0)

    def test_dry_run_without_framework_or_weights_and_optional_ports(self):
        for engine, port in [('sglang', '3333'), ('vllm', '5555')]:
            output = io.StringIO()
            flag = '--model-path' if engine == 'sglang' else '--model'
            with patch.object(launcher, 'resolve_python', side_effect=WorkflowError('not built')), \
                 patch.object(launcher.os, 'execve') as execute, redirect_stdout(output):
                code = launcher.launch(engine, [flag, '/missing/model'], port=1, cuda_devices='0',
                                       python=None, preparation_dir=None, configuration_file=Path(__file__),
                                       root=self.root, argv=[port, '--dry-run'])
            self.assertEqual(code, 0)
            self.assertIn(port, output.getvalue())
            self.assertIn('HF_HUB_OFFLINE', output.getvalue())
            execute.assert_not_called()
            self.assertFalse((self.root / 'artifacts/servers').exists())

    def test_model_override_is_effective_local_identity_and_venv_python_not_resolved(self):
        arguments, path, tp = launcher.effective_arguments(['--model', '/original'], 'vllm', 5555,
                                                          ['--model=/new/local', '--tensor-parallel-size=2'])
        self.assertEqual(path, Path('/new/local'))
        self.assertEqual(tp, 2)
        target = self.root / 'base-python'
        target.touch()
        symlink = self.root / 'venv-python'
        symlink.symlink_to(target)
        self.assertEqual(launcher.resolve_python(self.root, 'vllm', None, symlink)[0], symlink)

    def test_gb10_capacity_gate_does_not_treat_cpu_offload_as_extra_unified_memory(self):
        self.assertIsNotNone(launcher.capacity_error('NVIDIA GB10', 491 * 1024**3, 119 * 1024**3))
        self.assertIsNone(launcher.capacity_error('NVIDIA GB10', 80 * 1024**3, 119 * 1024**3))
        self.assertIsNone(launcher.capacity_error('NVIDIA GB300', 491 * 1024**3, 119 * 1024**3))

    def test_failed_cuda_probe_prevents_exec_and_receipt(self):
        with patch.object(launcher, 'resolve_python', return_value=(Path('/bin/true'), None)), \
             patch.object(launcher.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'CUDA missing')), \
             patch.object(launcher.os, 'execve') as execute, redirect_stderr(io.StringIO()):
            code = launcher.launch('vllm', ['--model', str(self.directory)], port=5555, cuda_devices='0',
                                   python=None, preparation_dir=None, configuration_file=Path(__file__),
                                   root=self.root, argv=[])
        self.assertEqual(code, 2)
        execute.assert_not_called()
        self.assertFalse((self.root / 'artifacts/servers').exists())

    def test_launch_records_effective_local_model_and_execs_isolated_python(self):
        probe = {'gpu': 'NVIDIA GB10', 'torch_cuda': '13.0',
                 'module_path': '/source/vllm/__init__.py', 'cuda_tensor_result': 28}
        with patch.object(launcher, 'resolve_python', return_value=(Path('/bin/true'), None)), \
             patch.object(launcher.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(probe), '')), \
             patch.object(launcher, '_git', return_value='parent'), \
             patch.object(launcher.os, 'execve') as execute, redirect_stdout(io.StringIO()):
            code = launcher.launch('vllm', ['--model', '/original', '--max-model-len', '4096'],
                                   port=5555, cuda_devices='0', python=None, preparation_dir=None,
                                   configuration_file=Path(__file__), root=self.root,
                                   argv=['5556', '--', '--model', str(self.directory), '--max-model-len', '8192'])
        self.assertEqual(code, 0)
        command = execute.call_args.args[1]
        self.assertEqual(command[-3:], [str(self.directory), '--max-model-len', '8192'])
        self.assertEqual(execute.call_args.args[2]['HF_HUB_OFFLINE'], '1')
        receipt = json.loads(next((self.root / 'artifacts/servers').glob('*/launch.json')).read_text())
        self.assertEqual(receipt['model']['path'], str(self.directory))
        self.assertFalse(receipt['gpu_inference_validated'])

    def test_prepared_python_preserves_venv_path_and_rejects_modified_source(self):
        directory = self.root / 'artifacts/init/ready'
        source = directory / 'vllm/source'
        source.mkdir(parents=True)
        base = directory / 'base-python'
        base.touch()
        python = directory / 'vllm/venv/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(base)
        pin = prep.SOURCES['vllm']['commit']
        record = {'frameworks': {'vllm': {'prepared': True, 'commit': pin, 'python': str(python),
                                         'source': str(source), 'extensions': []}}}
        (directory / 'manifest.json').write_text(json.dumps(record))
        with patch.object(launcher, '_git', side_effect=[pin, pin, '']), \
             patch.object(launcher, '_reject_hidden_changes'):
            self.assertEqual(launcher.resolve_python(self.root, 'vllm', None, None)[0], python)
        with patch.object(launcher, '_git', side_effect=[pin, pin, ' M model.py']), \
             patch.object(launcher, '_reject_hidden_changes'), self.assertRaisesRegex(WorkflowError, 'tracked'):
            launcher.resolve_python(self.root, 'vllm', None, None)

    def test_requested_repo_is_frozen_and_both_default_launchers_match(self):
        root = Path(__file__).resolve().parents[1]
        frozen = model.profile(root)
        self.assertEqual(frozen['model_id'], 'nv-community/DeepSeek-V4.1-Flash-NVFP4')
        self.assertEqual(frozen['weight_bytes'], 527293384576)
        self.assertEqual(serve_sglang.MODEL_PATH, serve_vllm.MODEL_PATH)
        self.assertEqual(serve_sglang.MODEL_PATH, model.MODEL_DIR)


if __name__ == '__main__':
    unittest.main()
