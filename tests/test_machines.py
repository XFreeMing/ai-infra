"""Machine routing and evidence admission; no frameworks or model inference required."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from lab_tools import cli, machines, machine_prepare as prep, machine_launcher as launcher
from lab_tools.prepare import sha256


class MachineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git(self.root, 'init', '-q')
        self.git(self.root, 'config', 'user.name', 'Fixture')
        self.git(self.root, 'config', 'user.email', 'fixture@example.invalid')
        (self.root / '.gitignore').write_text('artifacts/\n.cache/\n')
        (self.root / '.gitmodules').write_text(''.join(
            f'[submodule "sources/{name}"]\n path = sources/{name}\n url = https://example.invalid/{name}.git\n'
            for name in ('ds4', 'llama.cpp')))
        self.git(self.root, 'add', '.')
        self.git(self.root, 'commit', '-qm', 'Fixture')
        self.revision = patch('lab_tools.machine_prepare.source_revision', return_value=('a' * 40, ':source'))
        self.revision.start()
        self.addCleanup(self.revision.stop)

    def git(self, root, *args):
        return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    def invoke(self, machine, *argv):
        with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as errors:
            code = machines.main(machine, list(argv), root=self.root)
        return code, output.getvalue(), errors.getvalue()

    def manifest(self, machine):
        paths = list((self.root / 'artifacts/init' / machine).glob('*/manifest.json'))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text())

    def test_300_plan_has_hip_source_pins_and_never_executes_install_or_build(self):
        with patch.object(prep.Recorder, 'command') as command:
            code, _, _ = self.invoke('300', 'init', 'ds4', 'llama', '--dry-run', '--rocm-home', '/test-rocm')
        self.assertEqual(code, 0)
        manifest = self.manifest('300')
        self.assertEqual(manifest['status'], 'planned')
        self.assertFalse(manifest['gpu_smoke_passed'])
        self.assertFalse(manifest['gpu_inference_validated'])
        self.assertEqual(manifest['config']['build_timeout'], 900)
        for engine, plan in manifest['plans'].items():
            self.assertEqual(plan['commit'], 'a' * 40)
            self.assertEqual(plan['environment']['HIP_VISIBLE_DEVICES'], '0')
            self.assertNotIn('CUDA_VISIBLE_DEVICES', plan['environment'])
            self.assertIn('%f/%t', plan['environment']['NINJA_STATUS'])
            for item in plan['commands']:
                if item['label'] in ('clone', 'fetch', 'submodules'):
                    self.assertIn('--progress', item['argv'])
                if item['label'] == 'build' and engine == 'llama':
                    self.assertIn('--verbose', item['argv'])
            arguments = [arg for item in plan['commands'] for arg in item['argv']]
            self.assertNotIn('cu130', arguments)
            self.assertIn('strix-halo' if engine == 'ds4' else '-DGGML_HIP=ON', arguments)
        command.assert_not_called()

    def test_100_all_selects_only_cpu_llama_and_multiconfig_release(self):
        code, _, _ = self.invoke('100', 'init', 'all', '--dry-run')
        self.assertEqual(code, 0)
        record = self.manifest('100')
        self.assertEqual(record['selected_engines'], ['llama'])
        plan = record['plans']['llama']
        argv = [arg for item in plan['commands'] for arg in item['argv']]
        self.assertIn('-DGGML_CUDA=OFF', argv)
        self.assertIn('-DGGML_HIP=OFF', argv)
        self.assertIn('-DGGML_VULKAN=OFF', argv)
        self.assertIn('--config', argv)
        self.assertIn('--verbose', next(item['argv'] for item in plan['commands'] if item['label'] == 'build'))
        self.assertEqual(set(plan['environment']), {'NINJA_STATUS'})

    def test_700_positional_engines_delegate_existing_preparation(self):
        with patch('lab_tools.prepare.prepare', return_value=(self.root, 0)) as existing:
            self.assertEqual(self.invoke('700', 'init', 'freetoken', '--dry-run')[0], 0)
        args = existing.call_args.args[1]
        self.assertEqual(args.engines, ['freetoken'])
        self.assertIsNone(args.build_timeout)

    def test_700_service_keeps_port_and_literal_upstream_arguments(self):
        with patch('lab_tools.serve_llama.main', return_value=0) as launch:
            self.assertEqual(self.invoke('700', 'llama', '--dry-run', '--', '-c', '8192')[0], 0)
        launch.assert_called_once_with(['--dry-run', '--', '-c', '8192'])

    def test_unsupported_engines_do_not_start_cuda_preparation(self):
        for machine, engine in [('100', 'ds4'), ('300', 'vllm'), ('300', 'sglang'), ('300', 'freetoken')]:
            with self.subTest(machine=machine, engine=engine), patch('lab_tools.prepare.prepare') as cuda:
                code, _, error = self.invoke(machine, 'init', engine, '--dry-run')
                self.assertEqual(code, 2)
                self.assertIn('本机后端', error)
                cuda.assert_not_called()
        self.assertFalse((self.root / 'artifacts').exists())

    def test_alias_and_positional_engine_options_cannot_be_mixed(self):
        self.assertEqual(self.invoke('300', 'init', 'llama', '--engines', 'ds4', '--dry-run')[0], 2)

    def test_300_permission_failure_preserves_blocked_evidence_before_clone(self):
        report = {'home': '/rocm', 'hipcc_available': True, 'kfd': {'exists': True, 'read_write': False},
                  'rocminfo': {'exit_code': 1, 'error': 'Permission denied'}}
        with patch.object(prep, 'doctor', return_value={'project_commit': 'parent'}), \
             patch('lab_tools.machines.rocm_report', return_value=report), \
             patch.object(prep.Recorder, 'command') as command:
            self.assertEqual(self.invoke('300', 'init', 'llama', '--check-only')[0], 2)
        record = self.manifest('300')
        self.assertEqual(record['status'], 'blocked')
        self.assertIn('/dev/kfd', record['blockers'][0])
        self.assertEqual(record['environment']['rocm'], report)
        self.assertEqual(record['commands'], [])
        command.assert_not_called()

    def test_check_only_never_becomes_prepared_even_when_device_is_accessible(self):
        report = {'hipcc_available': True, 'kfd': {'read_write': True},
                  'rocminfo': {'exit_code': 0, 'output': 'gfx1151'}}
        with patch.object(prep, 'doctor', return_value={}), patch('lab_tools.machines.rocm_report', return_value=report), \
             patch.object(prep.shutil, 'which', return_value='/tools/bin/tool'), patch.object(prep.Recorder, 'command') as command:
            self.assertEqual(self.invoke('300', 'init', '--engines', 'llama', '--check-only')[0], 2)
        self.assertIn('check-only', self.manifest('300')['blockers'][0])
        command.assert_not_called()

    def test_rocm_cpu_only_or_wrong_arch_report_is_rejected(self):
        for code, output in [(1, 'gfx1151'), (0, 'gfx1103'), (0, '')]:
            with self.subTest(code=code, output=output), self.assertRaises(cli.WorkflowError):
                machines.require_rocm({'hipcc_available': True, 'kfd': {'read_write': True},
                                       'rocminfo': {'exit_code': code, 'output': output}})

    def test_wrong_host_blocks_actual_700_build_before_preparation(self):
        with patch.object(machines, 'host_matches', return_value=False), patch('lab_tools.prepare.prepare') as prepare:
            code, _, error = self.invoke('700', 'init', 'llama')
        self.assertEqual(code, 2)
        self.assertIn('aarch64', error)
        prepare.assert_not_called()

    def test_plan_rejects_source_updates_without_mutation(self):
        with patch.object(prep, 'update_sources') as update:
            self.assertEqual(self.invoke('300', 'init', '--update-sources', '--dry-run')[0], 2)
        update.assert_not_called()

    def test_machine_run_records_selected_id_in_manifest_and_child_environment(self):
        directory = self.root / 'experiments/check'
        directory.mkdir(parents=True)
        config = {'id': 'check', 'command': ['{python}', '-c', 'import os; print(os.environ["AI_INFRA_MACHINE_ID"])'],
                  'timeout_seconds': 5, 'required_sources': []}
        (directory / 'experiment.json').write_text(json.dumps(config))
        self.git(self.root, 'add', '.')
        self.git(self.root, 'commit', '-qm', 'Experiment')
        path, code = cli.run_experiment(self.root, 'check', self.root / 'artifacts/runs/300', machine_id='300')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((path / 'manifest.json').read_text())['machine_id'], '300')
        self.assertEqual((path / 'stdout.log').read_text().strip(), '300')

    def ready_build(self, machine='300'):
        directory = self.root / f'artifacts/init/{machine}/ready'
        source = directory / 'llama/source'
        source.mkdir(parents=True)
        self.git(source, 'init', '-q')
        self.git(source, 'config', 'user.name', 'Fixture')
        self.git(source, 'config', 'user.email', 'fixture@example.invalid')
        (source / 'source.txt').write_text('fixed')
        self.git(source, 'add', '.')
        self.git(source, 'commit', '-qm', 'Source')
        commit = self.git(source, 'rev-parse', 'HEAD')
        binary = directory / 'llama/llama-server'
        binary.write_bytes(b'fixture binary')
        library = directory / 'llama/libggml.so'
        library.write_bytes(b'fixture library')
        manifest = {'machine_id': machine, 'backend': machines.PROFILES[machine]['backend'], 'status': 'prepared',
                    'gpu_smoke_passed': True, 'builds': {'llama': {'commit': commit, 'source': str(source),
                    'server': str(binary), 'binaries': [{'path': str(p), 'sha256': sha256(p)} for p in (binary, library)],
                    'runtime_environment': {'ROCM_PATH': '/rocm', 'HIP_VISIBLE_DEVICES': '0'}}}}
        path = directory / 'manifest.json'
        path.write_text(json.dumps(manifest))
        return directory, path, manifest, binary, library

    def test_resolution_rejects_cross_machine_and_wrong_backend(self):
        directory, path, record, _, _ = self.ready_build()
        commit = record['builds']['llama']['commit']
        with patch.object(launcher, 'source_revision', return_value=(commit, 'fixture')):
            for key, value in [('machine_id', '700'), ('backend', 'cuda'), ('status', 'planned')]:
                updated = {**record, key: value}
                path.write_text(json.dumps(updated))
                with self.subTest(key=key), self.assertRaisesRegex(cli.WorkflowError, '尚无匹配'):
                    launcher.resolve_build(self.root, '300', 'llama', directory)

    def test_resolution_rechecks_library_checksum_and_source_modifications(self):
        directory, _, record, binary, library = self.ready_build()
        commit = record['builds']['llama']['commit']
        with patch.object(launcher, 'source_revision', return_value=(commit, 'fixture')):
            self.assertEqual(launcher.resolve_build(self.root, '300', 'llama', None)[0], binary)
            library.write_bytes(b'changed library')
            with self.assertRaisesRegex(cli.WorkflowError, '动态库发生变化'):
                launcher.resolve_build(self.root, '300', 'llama', directory)
            (Path(record['builds']['llama']['source']) / 'source.txt').write_text('changed')
            with self.assertRaisesRegex(cli.WorkflowError, '源码发生变化'):
                launcher.resolve_build(self.root, '300', 'llama', directory)

    def test_dry_run_model_overrides_are_literal_and_do_not_create_launch_receipt(self):
        model = self.root / 'literal $(no-shell).gguf'
        for machine, engine in [('100', 'llama'), ('300', 'llama'), ('300', 'ds4')]:
            with self.subTest(machine=machine, engine=engine), \
                 patch.object(launcher, 'source_revision', return_value=('a' * 40, 'fixture')), \
                 patch.object(launcher.os, 'execve') as execute:
                code, output, error = self.invoke(machine, engine, '--dry-run', '--', '--model', str(model))
                self.assertEqual(code, 0, error)
                plan = json.loads(output)
                self.assertEqual(plan['model_paths']['model'], str(model))
                if machine == '300':
                    self.assertEqual(plan['environment_overrides']['HIP_VISIBLE_DEVICES'], '0')
                if engine == 'ds4':
                    self.assertIn('--rocm', plan['argv'])
                    self.assertNotIn('--cuda', plan['argv'])
                self.assertFalse(plan['gpu_inference_validated'])
                execute.assert_not_called()
        self.assertFalse((self.root / 'artifacts/servers').exists())

    def test_launch_preserves_effective_model_identity_and_machine_receipt(self):
        _, path, record, binary, _ = self.ready_build('100')
        model = self.root / 'replacement.gguf'
        model.write_bytes(b'GGUFfixture')
        with patch.object(launcher, 'source_revision', return_value=(record['builds']['llama']['commit'], 'fixture')), \
             patch.object(launcher, 'require_host'), patch.object(launcher.os, 'execve', side_effect=SystemExit(0)) as execute, \
             patch.object(launcher.os, 'chdir'), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            launcher.launch_machine(self.root, '100', 'llama', ['--', '-m', str(model)])
        receipt_path = next((self.root / 'artifacts/servers/100').glob('*/launch.json'))
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt['machine_id'], '100')
        self.assertEqual(receipt['backend'], 'cpu')
        self.assertEqual(receipt['models']['model']['sha256'], sha256(model))
        self.assertEqual(receipt['preparation_manifest_sha256'], sha256(path))
        self.assertEqual(execute.call_args.args[0], binary)

    def test_old_cuda_selection_cannot_admit_rocm_manifest(self):
        from lab_tools import server_launcher, native_launcher
        study = self.root / 'research/studies/ds4-inference'
        study.mkdir(parents=True)
        (study / 'R0_MODEL.json').write_text(json.dumps({'engines': {'ds4': 'a', 'llama.cpp': 'b'}}))
        prepared = self.root / 'artifacts/init/rocm-fixture'
        prepared.mkdir(parents=True)
        (prepared / 'manifest.json').write_text(json.dumps({'status': 'prepared', 'machine_id': '300', 'backend': 'rocm',
                                                          'builds': {}, 'frameworks': {}}))
        with patch.object(server_launcher, 'source_revision', return_value=('a', 'fixture')), \
             self.assertRaises(cli.WorkflowError):
            server_launcher.resolve_server(self.root, 'llama', None, prepared)
        with patch.object(native_launcher, 'source_revision', return_value=('a', 'fixture')), \
             self.assertRaises(cli.WorkflowError):
            native_launcher.resolve_python(self.root, 'vllm', prepared, None)

    def test_windows_process_cleanup_uses_taskkill_tree(self):
        from unittest.mock import Mock
        process = Mock(pid=12345)
        process.poll.return_value = None
        with patch.object(cli.os, 'name', 'nt'), patch.object(cli.subprocess, 'run') as run:
            cli._stop_process_group(process)
        self.assertEqual(run.call_args.args[0], ['taskkill', '/PID', '12345', '/T', '/F'])
        process.wait.assert_called_once()


if __name__ == '__main__':
    unittest.main()
