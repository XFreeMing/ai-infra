"""Local-only hello contracts; generated artifacts and GPU work are mocked."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab_tools import machines, media_hello as hello
from lab_tools.cli import WorkflowError


class MediaHelloTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.preparation = self.root / 'prepared'
        self.preparation.mkdir()
        self.preparation_manifest = self.preparation / 'manifest.json'
        self.preparation_manifest.write_text('{"status": "prepared"}\n')
        self.runtime = {
            'python': str(self.preparation / 'venv/bin/python'),
            'source': str(self.preparation / 'source'),
            'worker': str(self.preparation / 'runtime/lab_tools/media_worker.py'),
            'environment': {'NLTK_DATA': str(self.preparation / 'nltk_data')},
        }
        self.commands = []
        self.failure = None
        self.validation_result = None
        self.omit_result = False
        self.locked = False
        self.events = []
        self.resolve = self.patching(hello, 'resolve_runtime', return_value=(self.runtime, self.preparation_manifest))
        self.host = self.patching(machines, 'require_host')
        self.verify = self.patching(hello, 'verify_local', side_effect=self.verify_models)
        self.patching(hello, '_git', return_value='fixture-parent-commit')
        self.lock = self.patching(hello, 'initialization_lock', side_effect=self.serial_lock)
        self.execute = self.patching(hello.Recorder, 'command', autospec=True, side_effect=self.command)

    def patching(self, target, name, **kwargs):
        patcher = patch.object(target, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    @contextmanager
    def serial_lock(self, root, timeout):
        self.assertEqual(root, self.root)
        self.assertGreater(timeout, 0)
        self.assertFalse(self.locked, 'hello targets must not overlap')
        self.locked = True
        self.events.append('lock')
        try:
            yield
        finally:
            self.events.append('unlock')
            self.locked = False

    def verify_models(self, app, directory):
        self.assertTrue(self.locked)
        self.events.append(('models', app))
        return {'verified': True, 'app': app, 'directory': str(directory)}

    def command(self, recorder, label, argv, cwd, timeout=None, env=None):
        self.assertTrue(self.locked, 'model operations must hold the shared lock')
        self.assertIsInstance(argv, list)
        self.assertEqual(cwd, self.root)
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 900)
        app = recorder.manifest['app']
        self.events.append((label, app))
        self.commands.append((label, argv, env))
        status, code = ('succeeded', 0)
        if self.failure and label == self.failure[0]:
            status, code = self.failure[1:]
        record = {'label': label, 'argv': argv, 'status': status, 'exit_code': code,
                  'log': label + '.log', 'timeout_seconds': timeout}
        recorder.manifest['commands'].append(record)
        (recorder.directory / record['log']).write_text('fixture subprocess output\n')
        if label == 'validate-reference' and code == 0:
            reference = Path(argv[argv.index('--reference') + 1])
            result = {'path': str(reference), 'sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                      'sample_rate': 24000, 'channels': 1, 'duration_seconds': 4.9}
            Path(argv[argv.index('--result') + 1]).write_text(json.dumps(result))
        if label == 'validate-output' and code == 0 and not self.omit_result:
            output = Path(recorder.manifest['output_dir'])
            output.mkdir()
            artifact = output / ('fixture.png' if app == 'ideogram4' else 'fixture.wav')
            artifact.write_bytes(b'output checked by mocked validator')
            result = self.validation_result
            if result is None:
                result = {'passed': True, 'app': app, 'output_path': str(artifact),
                          'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
            Path(argv[argv.index('--result') + 1]).write_text(json.dumps(result))
        return record

    def invoke(self, *argv, machine='300'):
        with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as error:
            code = machines.main(machine, ['hello', *argv], root=self.root)
        return code, output.getvalue(), error.getvalue()

    def manifests(self):
        return sorted([json.loads(path.read_text()) for path in
                       (self.root / 'artifacts/hello/300').glob('*/manifest.json')],
                      key=lambda item: item['started_at'])

    def assert_failed(self, status='failed'):
        manifest = self.manifests()[-1]
        self.assertEqual(manifest['status'], status)
        self.assertFalse(manifest['gpu_inference_validated'])
        self.assertIn('finished_at', manifest)
        return manifest

    def test_dry_run_is_case_insensitive_and_never_touches_gpu_or_outputs(self):
        before = sorted(str(path) for path in self.root.rglob('*'))
        with patch.object(hello.tempfile, 'mkdtemp') as mkdir:
            for name, app in [('iDEOgRAM4', 'ideogram4'), ('INDextts2', 'indextts2')]:
                with self.subTest(app=app):
                    code, output, error = self.invoke(name, '--dry-run')
                    self.assertEqual(code, 0, error)
                    plan = json.loads(output)
                    self.assertEqual(plan['app'], app)
                    self.assertFalse(plan['model_download_requested'])
                    self.assertFalse(plan['dependency_install_requested'])
                    self.assertEqual(plan['environment']['HF_HUB_OFFLINE'], '1')
                    self.assertEqual(plan['environment']['TRANSFORMERS_OFFLINE'], '1')
                    self.assertEqual(plan['timeout_seconds'], 900)
            mkdir.assert_not_called()
        self.host.assert_not_called()
        self.verify.assert_not_called()
        self.execute.assert_not_called()
        self.lock.assert_not_called()
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))

    def test_dry_run_can_describe_unprepared_runtime(self):
        self.resolve.side_effect = WorkflowError('尚未初始化 Ideogram4；先运行 init。')
        code, output, error = self.invoke('Ideogram4', '--dry-run')
        self.assertEqual(code, 0, error)
        plan = json.loads(output)
        self.assertIn('尚未初始化', plan['runtime_blocker'])
        self.assertIn('<prepared-ideogram4>', plan['argv'][0])
        self.assertIsNone(plan['preparation_manifest'])
        self.assertFalse((self.root / 'artifacts').exists())

    def test_only_machine_300_supports_hello(self):
        for machine in ('100', '700'):
            with self.subTest(machine=machine):
                code, _, error = self.invoke('Ideogram4', '--dry-run', machine=machine)
                self.assertEqual(code, 2)
                self.assertIn('仅支持 AI 300', error)
        self.resolve.assert_not_called()
        self.assertFalse((self.root / 'artifacts').exists())

    def test_plans_use_fixed_small_inputs_and_literal_argv(self):
        code, output, error = self.invoke('Ideogram4', '--dry-run', '--model-dir', 'models $(no-shell)')
        self.assertEqual(code, 0, error)
        plan = json.loads(output)
        argv = plan['argv']
        self.assertEqual(argv[:2], [self.runtime['python'], '-I'])
        self.assertEqual(argv[argv.index('--height') + 1], '256')
        self.assertEqual(argv[argv.index('--width') + 1], '256')
        self.assertEqual(argv[argv.index('--sampler') + 1], 'V4_TURBO_12')
        self.assertEqual(argv[argv.index('--seed') + 1], '0')
        self.assertEqual(json.loads(argv[argv.index('--prompt') + 1]), hello.CAPTION)
        self.assertEqual(argv[argv.index('--model-dir') + 1], str(self.root / 'models $(no-shell)'))
        self.assertNotIn('sh', argv)
        self.assertNotIn('bash', argv)
        code, output, error = self.invoke('Indextts2', '--dry-run', '--reference', 'voice $(no-shell).wav')
        self.assertEqual(code, 0, error)
        plan = json.loads(output)
        self.assertEqual(plan['argv'][plan['argv'].index('--prompt') + 1], hello.TEXT)
        self.assertLess(len(hello.TEXT), 40)
        self.assertEqual(plan['reference'], str(self.root / 'voice $(no-shell).wav'))
        self.assertTrue(plan['argv'][plan['argv'].index('--reference') + 1].endswith('/input/reference.wav'))
        self.assertEqual(plan['environment']['NLTK_DATA'], self.runtime['environment']['NLTK_DATA'])

    def test_packaged_reference_matches_pinned_digest_and_is_nonempty_wav(self):
        data = hello.REFERENCE.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), hello.REFERENCE_SHA256)
        # The upstream sample uses IEEE_FLOAT WAV, unsupported by stdlib wave.
        # Its actual decode is checked by soundfile in the prepared interpreter.
        self.assertEqual(data[:4], b'RIFF')
        self.assertEqual(data[8:12], b'WAVE')
        self.assertGreater(len(data), 44)

    def test_unprepared_runtime_never_runs_generation(self):
        self.resolve.side_effect = WorkflowError('尚未初始化 Indextts2；先运行 init。')
        code, _, _ = self.invoke('Indextts2')
        self.assertEqual(code, 2)
        self.execute.assert_not_called()
        self.verify.assert_not_called()
        self.assert_failed()

    def test_missing_models_never_run_generation_or_download(self):
        self.verify.side_effect = WorkflowError('本地模型文件缺失，请先运行 download。')
        code, _, _ = self.invoke('Ideogram4')
        self.assertEqual(code, 2)
        self.execute.assert_not_called()
        manifest = self.assert_failed()
        self.assertFalse(manifest['model_download_requested'])
        self.assertFalse(manifest['dependency_install_requested'])

    def test_success_requires_both_commands_and_preserves_audio_input_identity(self):
        code, _, error = self.invoke('Indextts2')
        self.assertEqual(code, 0, error)
        manifest = self.manifests()[0]
        self.assertEqual([entry['label'] for entry in manifest['commands']],
                         ['validate-reference', 'generate', 'validate-output'])
        self.assertEqual(manifest['status'], 'passed')
        self.assertTrue(manifest['gpu_inference_validated'])
        self.assertIs(manifest['result']['passed'], True)
        self.assertEqual(manifest['reference_audio']['sha256'], hello.REFERENCE_SHA256)
        copied = self.root / 'artifacts/hello/300' / manifest['run_id'] / 'input/reference.wav'
        self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), hello.REFERENCE_SHA256)
        self.assertEqual(manifest['preparation_manifest_sha256'], hashlib.sha256(self.preparation_manifest.read_bytes()).hexdigest())

    def test_generate_failure_preserves_failure_and_never_validates(self):
        self.failure = ('generate', 'failed', 1)
        code, _, _ = self.invoke('Ideogram4')
        self.assertEqual(code, 2)
        self.assertEqual([command[0] for command in self.commands], ['generate'])
        self.assert_failed()

    def test_validator_failure_never_marks_gpu_validated(self):
        self.failure = ('validate-output', 'failed', 1)
        code, _, _ = self.invoke('Ideogram4')
        self.assertEqual(code, 2)
        self.assertEqual([command[0] for command in self.commands], ['generate', 'validate-output'])
        self.assert_failed()

    def test_subprocess_timeout_is_recorded_and_never_passes(self):
        self.failure = ('generate', 'timed_out', 124)
        code, _, _ = self.invoke('Ideogram4')
        self.assertEqual(code, 124)
        self.assertEqual([command[0] for command in self.commands], ['generate'])
        self.assert_failed('timed_out')

    def test_recorder_interrupt_stops_following_target(self):
        self.failure = ('generate', 'interrupted', 130)
        code, _, _ = self.invoke('Ideogram4', 'Indextts2')
        self.assertEqual(code, 130)
        self.assertEqual(len(self.manifests()), 1)
        self.assertEqual([command[0] for command in self.commands], ['generate'])
        self.assert_failed('interrupted')

    def test_missing_validation_result_never_passes(self):
        self.omit_result = True
        code, _, _ = self.invoke('Ideogram4')
        self.assertEqual(code, 2)
        self.assert_failed()

    def test_rejected_or_wrong_app_result_never_passes(self):
        for result in ({'passed': False, 'app': 'ideogram4'},
                       {'passed': 1, 'app': 'ideogram4'},
                       {'passed': True, 'app': 'indextts2'},
                       {'passed': True, 'app': 'ideogram4'}):
            with self.subTest(result=result):
                self.validation_result = result
                code, _, _ = self.invoke('Ideogram4')
                self.assertEqual(code, 2)
                self.assert_failed()

    def test_multiple_targets_run_generation_and_validation_serially(self):
        code, _, error = self.invoke('Ideogram4', 'Indextts2')
        self.assertEqual(code, 0, error)
        self.assertEqual(self.events, [
            'lock', ('models', 'ideogram4'), ('generate', 'ideogram4'), ('validate-output', 'ideogram4'), 'unlock',
            'lock', ('models', 'indextts2'), ('validate-reference', 'indextts2'),
            ('generate', 'indextts2'), ('validate-output', 'indextts2'), 'unlock',
        ])
        self.assertEqual(self.lock.call_count, 2)
        self.assertTrue(all(manifest['status'] == 'passed' for manifest in self.manifests()))

    def test_reference_path_errors_stop_before_generation(self):
        code, _, _ = self.invoke('Indextts2', '--reference', str(self.root / 'missing.wav'))
        self.assertEqual(code, 2)
        self.execute.assert_not_called()
        self.assert_failed()

    def test_invalid_reference_rejection_stops_before_generation(self):
        broken = self.root / 'not-a-wave.wav'
        broken.write_bytes(b'not audio')
        self.failure = ('validate-reference', 'failed', 1)
        code, _, _ = self.invoke('Indextts2', '--reference', str(broken))
        self.assertEqual(code, 2)
        self.assertEqual([command[0] for command in self.commands], ['validate-reference'])
        self.assert_failed()

    def test_total_budget_expiry_before_generation_is_timeout(self):
        clock = [0.0]

        def verify_past_budget(app, directory):
            clock[0] = 901.0
            return {'verified': True}

        self.verify.side_effect = verify_past_budget
        with patch.object(hello.time, 'monotonic', side_effect=lambda: clock[0]):
            code, _, _ = self.invoke('Ideogram4')
        self.assertEqual(code, 124)
        self.execute.assert_not_called()
        self.assert_failed('timed_out')

    def test_waiting_for_initialization_lock_exhausts_total_budget(self):
        clock = [0.0]

        @contextmanager
        def lock_timeout(root, timeout):
            self.assertEqual(root, self.root)
            self.assertEqual(timeout, 2.0)
            clock[0] = 2.0
            raise WorkflowError('等待其他 native init 完成时耗尽构建预算。')
            yield  # Make the failure occur on context entry, as with the real lock.

        self.lock.side_effect = lock_timeout
        with patch.object(hello.time, 'monotonic', side_effect=lambda: clock[0]):
            code, _, _ = self.invoke('Ideogram4', '--timeout', '2')
        self.assertEqual(code, 124)
        self.verify.assert_not_called()
        self.execute.assert_not_called()
        manifest = self.assert_failed('timed_out')
        self.assertEqual(manifest['commands'], [])
        self.assertEqual(manifest['duration_seconds'], 2.0)
        self.assertIn('等待其他 native init', manifest['error'])

    def test_result_cannot_point_outside_output_directory(self):
        outside = self.root / 'unrelated.png'
        outside.write_bytes(b'existing unrelated artifact')
        self.validation_result = {'passed': True, 'app': 'ideogram4', 'output_path': str(outside),
                                  'sha256': hashlib.sha256(outside.read_bytes()).hexdigest()}
        code, _, _ = self.invoke('Ideogram4')
        self.assertEqual(code, 2)
        self.assert_failed()

    def test_changed_builtin_reference_is_rejected(self):
        code = None
        with patch.object(hello, 'REFERENCE_SHA256', '0' * 64):
            code, _, _ = self.invoke('Indextts2')
        self.assertEqual(code, 2)
        self.execute.assert_not_called()
        self.assertIn('SHA256', self.assert_failed()['error'])

    def test_target_specific_overrides_cannot_ambiguously_cover_two_apps(self):
        for flag in ('--model-dir', '--preparation-dir'):
            with self.subTest(flag=flag):
                code, _, error = self.invoke('Ideogram4', 'Indextts2', flag, 'somewhere')
                self.assertEqual(code, 2)
                self.assertIn('单个应用', error)
        code, _, error = self.invoke('Ideogram4', '--reference', 'voice.wav')
        self.assertEqual(code, 2)
        self.assertIn('只用于', error)
        self.resolve.assert_not_called()
        self.assertFalse((self.root / 'artifacts').exists())


if __name__ == '__main__':
    unittest.main()
