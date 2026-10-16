"""The known GPU hang is guarded without initializing GPU libraries."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab_tools import media_gpu_guard as guard, media_launcher as launcher
from lab_tools.cli import WorkflowError


class GpuGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.kfd = self.root / 'kfd'
        self.proc = self.root / 'proc'
        self.kfd.mkdir()
        self.proc.mkdir()

    def test_lists_kfd_clients_without_command_lines_or_self(self):
        for pid in (123, os.getpid()):
            (self.kfd / str(pid)).mkdir()
            (self.proc / str(pid)).mkdir()
            (self.proc / str(pid) / 'comm').write_text('python\n')
        (self.kfd / 'not-a-pid').mkdir()
        result = guard.kfd_processes(self.kfd, self.proc)
        self.assertEqual(result['status'], 'observed')
        self.assertEqual(result['processes'], [{'pid': 123, 'comm': 'python', 'comm_status': 'observed'}])
        self.assertEqual(result['errors'], [])

    def test_unreadable_kfd_is_unknown_not_empty_observation(self):
        with patch.object(Path, 'iterdir', side_effect=PermissionError('fixture denied')):
            result = guard.kfd_processes(self.kfd, self.proc)
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(result['errors'])

    def test_unreadable_comm_keeps_kfd_process_and_blocks(self):
        (self.kfd / '123').mkdir()
        snapshot = guard.kfd_processes(self.kfd, self.proc)
        self.assertEqual(snapshot['processes'][0]['comm_status'], 'unknown')
        with patch.object(guard, 'kfd_processes', return_value=snapshot):
            record = guard.assess_gpu_launch('ideogram4', experimental=False)
        self.assertEqual(record['status'], 'blocked')
        self.assertIn('PID 123', ' '.join(record['reasons']))

    def test_unknown_concurrency_is_recorded_and_warned_for_small_trial(self):
        with patch.object(guard, 'kfd_processes', return_value={'status': 'unknown', 'processes': [], 'errors': ['missing']}):
            record = guard.assess_gpu_launch('ideogram4', experimental=False)
        self.assertEqual(record['status'], 'allowed')
        self.assertIn('unknown', ' '.join(record['warnings']))
        self.assertIn('尚未验证稳定', ' '.join(record['warnings']))

    def test_tts_does_not_inspect_gpu_clients(self):
        with patch.object(guard, 'kfd_processes') as inspect:
            record = guard.assess_gpu_launch('indextts2', experimental=False)
        self.assertEqual(record['status'], 'not_applicable')
        inspect.assert_not_called()


class LauncherGpuGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'manifest.json'
        self.path.write_text('{}')
        self.runtime = {'python': '/fixture/python', 'worker': '/fixture/worker',
                        'source': '/fixture/source', 'environment': {}}
        self.resolve = self.patching(launcher, 'resolve_runtime', return_value=(self.runtime, self.path))
        self.patching(launcher, 'require_host')
        self.patching(launcher, '_git', return_value='fixture')
        self.patching(launcher.os, 'chdir')
        self.execute = self.patching(launcher.os, 'execve')
        self.verify = self.patching(launcher, 'verify_local', return_value={'verified': True})
        self.scan = self.patching(guard, 'kfd_processes', return_value={'status': 'observed', 'processes': [], 'errors': []})

    def patching(self, target, name, **kwargs):
        patcher = patch.object(target, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def receipt(self):
        return json.loads(next((self.root / 'artifacts/servers/300').glob('*/launch.json')).read_text())

    def launch(self, *argv, app='ideogram4'):
        with redirect_stdout(io.StringIO()) as output:
            code = launcher.launch_media(self.root, app, list(argv))
        return code, output.getvalue()

    def test_webui_is_blocked_before_weights_or_exec_with_receipt(self):
        with self.assertRaisesRegex(WorkflowError, 'GPU hang/reset'):
            self.launch()
        self.verify.assert_not_called()
        self.execute.assert_not_called()
        self.assertEqual(self.receipt()['status'], 'blocked')
        self.assertEqual(self.receipt()['gpu_guard']['mode'], 'webui')

    def test_high_resolution_single_launch_is_blocked(self):
        with self.assertRaisesRegex(WorkflowError, '1344×2016'):
            self.launch('--prompt', '{}', '--width', '1344', '--height', '2016')
        self.execute.assert_not_called()

    def test_dry_run_describes_block_without_writing_or_executing(self):
        code, output = self.launch('--dry-run')
        self.assertEqual(code, 0)
        plan = json.loads(output)
        self.assertEqual(plan['gpu_guard']['status'], 'blocked')
        self.assertFalse(plan['gpu_guard']['experimental_gpu'])
        self.assertFalse((self.root / 'artifacts').exists())
        self.verify.assert_not_called()
        self.execute.assert_not_called()

    def test_explicit_override_records_risk_and_other_process(self):
        self.scan.return_value = {'status': 'observed', 'processes': [{'pid': 42, 'comm': 'other'}], 'errors': []}
        self.assertEqual(self.launch('--experimental-gpu')[0], 0)
        self.execute.assert_called_once()
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'launch_requested')
        self.assertEqual(receipt['gpu_guard']['status'], 'experimental_override')
        self.assertTrue(receipt['gpu_guard']['experimental_gpu'])
        self.assertIn('PID 42', ' '.join(receipt['gpu_guard']['reasons']))

    def test_small_single_launch_rechecks_concurrency_after_model_validation(self):
        self.scan.side_effect = [
            {'status': 'observed', 'processes': [], 'errors': []},
            {'status': 'observed', 'processes': [{'pid': 42, 'comm': 'other'}], 'errors': []},
        ]
        with self.assertRaisesRegex(WorkflowError, 'PID 42'):
            self.launch('--prompt', '{}')
        self.verify.assert_called_once()
        self.execute.assert_not_called()
        self.assertEqual(self.receipt()['status'], 'blocked')

    def test_small_single_launch_warns_and_preserves_size(self):
        code, output = self.launch('--prompt', '{}', '--width', '336', '--height', '496')
        self.assertEqual(code, 0)
        self.assertIn('尚未验证稳定', output)
        self.assertEqual(self.receipt()['gpu_guard']['width'], 336)
        self.execute.assert_called_once()

    def test_tts_does_not_get_blocked_by_ideogram_guard(self):
        self.assertEqual(self.launch(app='indextts2')[0], 0)
        self.scan.assert_not_called()
        self.execute.assert_called_once()
        self.assertEqual(self.receipt()['gpu_guard']['status'], 'not_applicable')

    def test_tts_rejects_image_dimensions_before_resolving_runtime(self):
        with self.assertRaisesRegex(WorkflowError, '不能用于 Indextts2'):
            self.launch('--width', '512', app='indextts2')
        self.resolve.assert_not_called()
        self.execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
