"""Server selection, argv forwarding and launch provenance without model inference."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab_tools import serve_ds4, serve_llama, server_launcher as launcher
from lab_tools.cli import WorkflowError
from lab_tools.prepare import fingerprint


class ServerLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ai-infra-serve-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model with spaces.gguf'
        self.model.write_bytes(b'GGUFfixture')
        self.config = self.root / 'settings.py'
        self.config.write_text('# editable settings\n')
        study = self.root / 'research/studies/ds4-inference'
        study.mkdir(parents=True)
        (study / 'R0_MODEL.json').write_text(json.dumps({'engines': {'ds4': 'a', 'llama.cpp': 'b'}}))
        self.prepared = self.root / 'artifacts/init/20260101-ready'
        self.prepared.mkdir(parents=True)
        self.ds4 = self.prepared / 'ds4-new/ds4-server'
        self.llama = self.prepared / 'llama/build/bin/llama-server'
        for binary in (self.ds4, self.llama):
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b'#!/bin/sh\nexit 0\n')
            binary.chmod(0o755)
        self.manifest = {'status': 'prepared', 'builds': {
            'ds4-new': {'commit': 'a', 'binaries': [self.record(self.ds4)]},
            'llama': {'commit': 'b', 'binaries': [self.record(self.llama)]}},
            'model': {'path': str(self.model), 'stat': fingerprint(self.model.stat()), 'verified': True,
                      'sha256': hashlib.sha256(self.model.read_bytes()).hexdigest()}}
        self.save()

    def record(self, path):
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def save(self):
        (self.prepared / 'manifest.json').write_text(json.dumps(self.manifest))

    def run_launch(self, argv, **kwargs):
        return launcher.launch('ds4', self.model, ['--cuda', '-m', str(self.model), '--port', '1111'],
                               cuda_devices='0', binary=None, preparation_dir=None,
                               configuration_file=self.config, root=self.root,
                               argv=argv, **kwargs)

    def test_incomplete_newer_preparation_does_not_shadow_ready_build(self):
        newer = self.root / 'artifacts/init/20260102-in-progress'
        newer.mkdir()
        (newer / 'manifest.json').write_text(json.dumps({'status': 'preparing'}))
        binary, _, manifest_path = launcher.resolve_server(self.root, 'ds4', None, None)
        self.assertEqual(binary, self.ds4)
        self.assertEqual(manifest_path, self.prepared / 'manifest.json')
        self.assertEqual(launcher.resolve_server(self.root, 'llama', None, None)[0], self.llama)

    def test_changed_binary_or_changed_engine_pin_is_rejected(self):
        self.ds4.write_bytes(b'changed')
        with self.assertRaisesRegex(WorkflowError, '发生变化'):
            launcher.resolve_server(self.root, 'ds4', None, None)
        self.manifest['builds']['ds4-new']['commit'] = 'different'
        self.save()
        with self.assertRaisesRegex(WorkflowError, 'uv run init'):
            launcher.resolve_server(self.root, 'ds4', None, None)

    def test_dry_run_does_not_exec_or_create_launch_receipt(self):
        with patch.object(launcher.os, 'execve') as execute, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.run_launch(['--dry-run']), 0)
        self.assertIn('CUDA_VISIBLE_DEVICES=0', output.getvalue())
        self.assertIn("'" + str(self.model) + "'", output.getvalue())
        execute.assert_not_called()
        self.assertFalse((self.root / 'artifacts/servers').exists())

    def test_foreground_exec_receives_cuda_env_and_literal_argv_and_saves_receipt(self):
        extra = ['--', '--trace', 'literal $(no-shell).log']
        with patch.object(launcher.os, 'execve', side_effect=SystemExit(0)) as execute, \
             patch.object(launcher.os, 'chdir') as chdir, \
             patch.object(launcher, '_git', side_effect=['parent', 'dirty']), \
             redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            self.run_launch(extra)
        binary, argv, env = execute.call_args.args
        self.assertEqual(binary, self.ds4)
        self.assertEqual(argv[-2:], ['--trace', 'literal $(no-shell).log'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
        chdir.assert_called_once_with(self.root)
        receipt = json.loads(next((self.root / 'artifacts/servers').glob('*/launch.json')).read_text())
        self.assertEqual(receipt['kind'], 'service_launch_request')
        self.assertFalse(receipt['model_inference_validated'])
        self.assertEqual(receipt['model']['sha256_verified_by_preparation'], self.manifest['model']['sha256'])

    def test_missing_model_blocks_launch_but_server_help_does_not_load_it(self):
        self.model.unlink()
        with patch.object(launcher.os, 'execve') as execute, redirect_stderr(io.StringIO()):
            self.assertEqual(self.run_launch(['--dry-run']), 2)
        execute.assert_not_called()
        with patch.object(launcher.os, 'execve', side_effect=SystemExit(0)) as execute, \
             patch.object(launcher.os, 'chdir'), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            self.run_launch(['--server-help'])
        self.assertEqual(execute.call_args.args[1][-1], '--help')
        self.assertFalse((self.root / 'artifacts/servers').exists())

    def test_model_override_records_its_identity_instead_of_the_original_model(self):
        replacement = self.root / 'replacement.gguf'
        replacement.write_bytes(b'GGUFdifferent-model')
        with patch.object(launcher.os, 'execve', side_effect=SystemExit(0)), \
             patch.object(launcher.os, 'chdir'), patch.object(launcher, '_git', return_value='parent'), \
             redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            self.run_launch(['--', '-m', str(replacement)])
        receipt = json.loads(next((self.root / 'artifacts/servers').glob('*/launch.json')).read_text())
        self.assertEqual(receipt['model']['path'], str(replacement))
        self.assertIsNone(receipt['model']['sha256_verified_by_preparation'])

    def test_top_level_settings_change_model_context_and_optional_projection(self):
        with patch.object(serve_llama, 'MODEL_PATH', self.model), \
             patch.object(serve_llama, 'CONTEXT_SIZE', 8192), \
             patch.object(serve_llama, 'MMPROJ_PATH', self.root / 'mmproj.gguf'):
            argv = serve_llama.server_arguments()
        self.assertEqual(argv[argv.index('-m') + 1], str(self.model))
        self.assertEqual(argv[argv.index('-c') + 1], '8192')
        self.assertEqual(argv[argv.index('--mmproj') + 1], str(self.root / 'mmproj.gguf'))
        self.assertEqual(argv[argv.index('--gpu-layers') + 1], 'all')
        self.assertEqual(argv[argv.index('--port') + 1], '2222')
        ds4 = serve_ds4.server_arguments()
        self.assertIn('--cuda', ds4)
        self.assertEqual(ds4[ds4.index('--port') + 1], '1111')
        self.assertNotIn('--vision', ds4)
        self.assertNotIn('--dspark', ds4)


if __name__ == '__main__':
    unittest.main()
