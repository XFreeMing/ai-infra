"""Contracts for separate runtime init, explicit downloads, and local-only launch."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lab_tools import machines, media_download as download, media_launcher as launch, media_models as models
from lab_tools import media_prepare as prepare, media_worker as worker
from lab_tools.cli import WorkflowError
from lab_tools.prepare import sha256


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git(self.root, 'init', '-q')
        self.git(self.root, 'config', 'user.name', 'Fixture')
        self.git(self.root, 'config', 'user.email', 'fixture@example.invalid')
        (self.root / '.gitignore').write_text('artifacts/\n')
        (self.root / 'pyproject.toml').write_text('[project]\nname="fixture"\nversion="1"\n')
        self.git(self.root, 'add', '.')
        self.git(self.root, 'commit', '-qm', 'Fixture')
        parser = machines.argparse.ArgumentParser()
        machines.add_machine_init_arguments(parser, self.root, '300')
        parser.add_argument('--rocm-home', type=Path)
        self.args = parser.parse_args(['--dry-run'])

    def git(self, root, *argv):
        return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()

    def invoke(self, machine, *argv):
        with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as error:
            code = machines.main(machine, list(argv), root=self.root)
        return code, output.getvalue(), error.getvalue()

    def test_case_insensitive_init_creates_runtime_only_plan(self):
        for app in ('Ideogram4', 'IndexTTS2'):
            with self.subTest(app=app), patch.object(prepare.Recorder, 'command') as execute:
                code, output, error = self.invoke('300', 'init', app, '--dry-run')
                self.assertEqual(code, 0, error)
                execute.assert_not_called()
                manifests = [json.loads(p.read_text()) for p in (self.root / 'artifacts/init/300').glob('*/manifest.json')]
                manifest = next(m for m in manifests if m['app'] == app.casefold())
                self.assertFalse(manifest['model_download_requested'])
                self.assertFalse(manifest['model_loading_requested'])
                commands = manifest['plan']['commands']
                self.assertIn('probe', commands[-1])
                self.assertNotIn('run', commands[-1])
                self.assertFalse(any('snapshot_download' in ' '.join(argv) for argv in commands))
                self.assertNotIn('download_media.py', output)
        self.assertFalse((self.root / 'models').exists())

    def test_multiple_media_targets_are_prepared_serially(self):
        with patch.object(prepare, 'prepare_media', return_value=(self.root, 0)) as init:
            self.assertEqual(self.invoke('300', 'init', 'Ideogram4', 'Indextts2', '--dry-run')[0], 0)
        self.assertEqual([call.args[1] for call in init.call_args_list], ['ideogram4', 'indextts2'])

    def test_runtime_uses_domestic_index_and_keeps_rocm_wheels(self):
        for app in models.APPS:
            plan = prepare.plan_media(self.root, app, self.root / app, self.args)
            mirror = 'https://pypi.tuna.tsinghua.edu.cn/simple'
            self.assertEqual(plan['package_default_index'], mirror)
            self.assertEqual(plan['environment']['UV_DEFAULT_INDEX'], mirror)
            self.assertEqual(plan['environment']['PIP_INDEX_URL'], mirror)
            for command in plan['commands'][5:7]:
                self.assertEqual(command[command.index('--default-index') + 1], mirror)
                self.assertNotIn('--torch-backend', command)
            wheel_args = [arg for arg in plan['commands'][5] if ' @ ' in arg]
            expected = 3 if app == 'indextts2' else 2
            self.assertEqual(len(wheel_args), expected)
            for arg in wheel_args:
                self.assertIn('https://mirror.nju.edu.cn/pytorch/whl/', arg)
                self.assertNotIn('download.pytorch.org', arg)
                if arg.startswith(('torch @ ', 'torchaudio @ ')):
                    self.assertIn('2.11.0%2Brocm7.2', arg)
                    self.assertRegex(arg, r'#sha256=[a-f0-9]{64}$')
            abi = 'cp' + models.APPS[app]['python'].replace('.', '')
            self.assertTrue(all(abi + '-' + abi in arg for arg in wheel_args))

    def test_project_mirror_change_reaches_runtime_and_download_sdk(self):
        mirror = 'https://mirrors.aliyun.com/pypi/simple'
        with (self.root / 'pyproject.toml').open('a') as config:
            config.write('\n[[tool.uv.index]]\nname="mirror"\nurl="' + mirror + '"\ndefault=true\n')
        plan = prepare.plan_media(self.root, 'indextts2', self.root / 'runtime', self.args)
        self.assertEqual(plan['package_default_index'], mirror)
        self.assertEqual(plan['environment']['PIP_INDEX_URL'], mirror)
        with patch.object(download.Recorder, 'command', return_value={'exit_code': 0, 'status': 'completed'}) as execute, \
             redirect_stdout(io.StringIO()):
            code = download.main(['Ideogram4', '--machine', '300'], root=self.root)
        self.assertEqual(code, 0)
        command = execute.call_args.args[1]
        self.assertEqual(command[command.index('--default-index') + 1], mirror)
        self.assertEqual(execute.call_args.args[4]['UV_DEFAULT_INDEX'], mirror)
        self.assertEqual(execute.call_args.args[4]['PIP_INDEX_URL'], mirror)
        manifests = list((self.root / 'artifacts/downloads/300').glob('*/manifest.json'))
        self.assertEqual(json.loads(manifests[0].read_text())['package_default_index'], mirror)

    def test_700_does_not_select_rocm_media_runtime(self):
        with patch.object(prepare, 'prepare_media') as init:
            self.assertEqual(self.invoke('700', 'init', 'Ideogram4', '--dry-run')[0], 2)
        init.assert_not_called()

    def test_new_init_rejects_model_arguments_before_any_preparation(self):
        self.args.model = Path('/missing-model')
        with self.assertRaisesRegex(WorkflowError, '只准备'):
            prepare.prepare_media(self.root, 'ideogram4', self.args)
        self.assertFalse((self.root / 'artifacts').exists())

    def test_download_plans_use_modelscope_and_machine_paths_without_installing_sdk(self):
        for machine, base in [('300', '/home/ops396/models'), ('700', '/data/models')]:
            with self.subTest(machine=machine), patch.object(download.Recorder, 'command') as execute, \
                 redirect_stdout(io.StringIO()) as output:
                code = download.main(['Indextts2', '--machine', machine, '--dry-run'], root=self.root)
            self.assertEqual(code, 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['directory'], base + '/Indextts2')
            self.assertEqual(plan['provider'], 'modelscope')
            self.assertFalse(plan['download_requested'])
            self.assertFalse(plan['huggingface_download_requested'])
            self.assertEqual(len(plan['repositories']), 5)
            execute.assert_not_called()
        self.assertFalse((self.root / 'artifacts').exists())

    def test_profile_download_alias_routes_machine_700(self):
        with redirect_stdout(io.StringIO()):
            code, output, error = self.invoke('700', 'download', 'Ideogram4', '--dry-run')
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(output)['directory'], '/data/models/Ideogram4')

    def test_download_local_dir_override_and_unknown_name(self):
        with redirect_stdout(io.StringIO()) as output:
            code = download.main(['Ideogram4', '--local-dir', str(self.root / 'custom'), '--dry-run'], root=self.root)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())['directory'], str(self.root / 'custom'))
        with redirect_stderr(io.StringIO()):
            self.assertEqual(download.main(['wrong', '--dry-run'], root=self.root), 2)

    def test_catalog_has_pinned_revisions_complete_auxiliary_and_no_training_weights(self):
        for app in models.APPS:
            for repo in models.catalog(app):
                self.assertEqual(len(repo['revision']), 40)
                for file in repo['files']:
                    self.assertEqual(len(file['sha256']), 64)
                    self.assertNotIn('optimizer', file['path'])
                    self.assertNotIn('discriminator', file['path'])
        repos = {repo['model_id'] for repo in models.catalog('indextts2')}
        self.assertIn('nv-community/bigvgan_v2_22khz_80band_256x', repos)
        self.assertIn('iic/speech_campplus_sv_zh-cn_16k-common', repos)
        self.assertNotIn('nvidia/bigvgan_v2_22khz_80band_256x', repos)

    def fixture_files(self):
        payload = b'tiny fixture weights'
        return payload, [{'path': 'weight.bin', 'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}]

    def test_local_receipt_rechecks_changed_file_and_preserves_bad_weights(self):
        payload, files = self.fixture_files()
        directory = self.root / 'model'
        directory.mkdir()
        (directory / 'weight.bin').write_bytes(payload)
        with patch.object(models, 'files_for', return_value=files), patch.object(models, 'catalog', return_value=[]), redirect_stdout(io.StringIO()):
            receipt = models.verify_local('ideogram4', directory)
            (directory / 'download-identity.json').write_text(json.dumps(receipt))
            with patch.object(models, 'hash_progress') as digest:
                self.assertTrue(models.verify_local('ideogram4', directory)['verified'])
                digest.assert_not_called()
            (directory / 'weight.bin').write_bytes(b'X' * len(payload))
            with self.assertRaisesRegex(WorkflowError, 'SHA256'):
                models.verify_local('ideogram4', directory)
        self.assertTrue((directory / 'weight.bin').exists())

    def test_local_file_helpers_reject_unknown_repositories_and_path_escape(self):
        root = self.root / 'models'
        directory = root / 'auxiliary/campplus'
        directory.mkdir(parents=True)
        file = directory / 'campplus_cn_common.bin'
        file.write_bytes(b'fixture')
        with patch.dict(os.environ, {'AI_INFRA_MEDIA_MODEL_DIR': str(root)}):
            self.assertEqual(worker.local_file('funasr/campplus', file.name), str(file))
            with self.assertRaises(ValueError):
                worker.local_model('unknown/model')
            with self.assertRaises(WorkflowError):
                worker.local_file('funasr/campplus', '../../../outside.bin')

    def test_media_launch_plans_default_to_webui_and_disable_network(self):
        for name, port in [('Ideogram4', '7861'), ('Indextts2', '7860')]:
            with self.subTest(name=name), patch.object(launch.os, 'execve') as execute:
                code, output, error = self.invoke('300', name, '--dry-run')
            self.assertEqual(code, 0, error)
            plan = json.loads(output)
            self.assertEqual(plan['model_dir'], '/home/ops396/models/' + name)
            self.assertEqual(plan['argv'][plan['argv'].index('--port') + 1], port)
            self.assertNotIn('--prompt', plan['argv'])
            self.assertFalse(plan['model_download_requested'])
            self.assertEqual(plan['environment']['HF_HUB_OFFLINE'], '1')
            execute.assert_not_called()

    def test_prompt_file_and_reference_are_forwarded_as_literal_arguments(self):
        path = self.root / 'literal $(no-shell).json'
        path.write_text('{"high_level_description":"a red boat"}')
        code, output, error = self.invoke('300', 'Ideogram4', '--dry-run', '--prompt-json', str(path))
        self.assertEqual(code, 0, error)
        args = json.loads(output)['argv']
        self.assertEqual(args[args.index('--prompt') + 1], path.read_text())

    def test_missing_models_never_exec_or_trigger_download(self):
        manifest = {'python': '/python', 'worker': '/worker', 'source': '/source', 'environment': {}}
        with patch.object(launch, 'resolve_runtime', return_value=(manifest, self.root / 'manifest.json')), \
             patch.object(launch, 'require_host'), patch.object(launch.os, 'execve') as execute, \
             patch.object(download.Recorder, 'command') as download_models, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(WorkflowError, 'uv run download'):
                launch.launch_media(self.root, 'indextts2', ['--model-dir', str(self.root / 'missing-model')])
        execute.assert_not_called()
        download_models.assert_not_called()

    def test_worker_probe_does_not_construct_models(self):
        constructor = unittest.mock.Mock()
        module = SimpleNamespace(__file__=str(self.root / 'module.py'), IndexTTS2=constructor)
        with patch.object(worker, 'torch_probe', return_value={'torch': models.TORCH_VERSION}), \
             patch.object(worker, 'import_app', return_value=module), redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main(['probe', '--app', 'indextts2', '--source', str(self.root),
                                          '--output', str(self.root / 'provenance.json')]), 0)
        constructor.assert_not_called()

    def test_index_adapters_change_model_io_and_align_dependency_metadata(self):
        source = self.root / 'source'
        (source / 'indextts/utils').mkdir(parents=True)
        (source / 'pyproject.toml').write_text('[project]\ndependencies=["torch==2.8.*","torchaudio==2.8.*","transformers==4.52.1"]\n')
        (source / 'indextts/infer_v2.py').write_text('from huggingface_hub import hf_hub_download\n'
             'extract = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")\n'
             'vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)\n')
        (source / 'indextts/utils/maskgct_utils.py').write_text('from huggingface_hub import hf_hub_download\n'
             'model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")\n')
        record = prepare.patch_source(source, 'indextts2')
        self.assertIn(models.TORCH_VERSION, (source / 'pyproject.toml').read_text())
        self.assertIn('local_model(bigvgan_name)', (source / 'indextts/infer_v2.py').read_text())
        self.assertEqual(record['sha256'], sha256(Path(record['path'])))
        self.assertNotIn('deepspeed', Path(record['requirements']).read_text())

    def test_download_sdk_receipt_requires_hashes_and_handles_only_selected_files(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/download_media.py'
        spec = importlib.util.spec_from_file_location('test_download_worker', script)
        sdk_worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdk_worker)
        payload, files = self.fixture_files()
        destination = self.root / 'download'
        plan = {'directory': str(destination), 'app': 'indextts2', 'machine_id': '300', 'total_bytes': len(payload),
                'repositories': [{'directory': 'auxiliary/fixture', 'model_id': 'test/fixture', 'revision': 'a' * 40, 'files': files}]}
        calls = []
        def snapshot(**kwargs):
            calls.append(kwargs)
            (Path(kwargs['local_dir']) / 'weight.bin').write_bytes(payload)
        with redirect_stdout(io.StringIO()) as output:
            receipt = sdk_worker.execute(plan, snapshot)
        self.assertTrue(receipt['verified'])
        self.assertEqual(receipt['files'][0]['path'], 'auxiliary/fixture/weight.bin')
        self.assertEqual(calls[0]['allow_file_pattern'], ['weight.bin'])
        self.assertIn('100.0%', output.getvalue())
        self.assertTrue((destination / 'download-identity.json').is_file())
        def corrupt(**kwargs):
            (Path(kwargs['local_dir']) / 'weight.bin').write_bytes(b'X' * len(payload))
        (destination / 'download-identity.json').unlink()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'SHA256'):
            sdk_worker.execute(plan, corrupt)
        self.assertFalse((destination / 'download-identity.json').exists())

    def test_runtime_selection_rejects_changed_frozen_worker(self):
        source = self.root / 'source'
        source.mkdir()
        self.git(source, 'init', '-q')
        self.git(source, 'config', 'user.name', 'Fixture')
        self.git(source, 'config', 'user.email', 'fixture@example.invalid')
        (source / 'main.py').write_text('fixed')
        self.git(source, 'add', '.')
        self.git(source, 'commit', '-qm', 'Source')
        commit = self.git(source, 'rev-parse', 'HEAD')
        directory = self.root / 'artifacts/init/300/ready'
        directory.mkdir(parents=True)
        helper = directory / 'worker.py'
        helper.write_text('frozen worker')
        (directory / 'freeze.txt').write_text('fixed dependencies')
        manifest = {'app': 'ideogram4', 'preparation_id': 'media-runtime', 'machine_id': '300', 'backend': 'rocm',
                    'status': 'prepared', 'source_commit': commit, 'torch': models.TORCH_VERSION,
                    'source': str(source), 'python': sys.executable, 'runtime_files': {str(helper): sha256(helper)},
                    'freeze_sha256': sha256(directory / 'freeze.txt')}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        with patch.dict(models.APPS['ideogram4'], {'commit': commit}):
            self.assertEqual(launch.resolve_runtime(self.root, 'ideogram4', None)[1], directory / 'manifest.json')
            helper.write_text('changed')
            with self.assertRaisesRegex(WorkflowError, '发生变化'):
                launch.resolve_runtime(self.root, 'ideogram4', None)


if __name__ == '__main__':
    unittest.main()
