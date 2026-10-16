"""Progress stays live under logging, including PTY control sequences and failures."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from lab_tools import prepare
from lab_tools.init_progress import TerminalText


class RecordingObserver:
    def __init__(self, *, failing_action=None, print_samples=False):
        self.failing_action = failing_action
        self.print_samples = print_samples
        self.calls = []
        self.pid = None

    def _call(self, action):
        self.calls.append(action)
        if action == self.failing_action:
            raise RuntimeError('monitor fixture failure')

    def start(self, pid):
        self._call('start')
        self.pid = pid

    def poll(self):
        self._call('poll')
        if self.print_samples:
            print('resource sample', flush=True)
        return self.print_samples

    def finish(self):
        self._call('finish')
        return {'samples': self.calls.count('poll'), 'pid': self.pid}


class InitProgressTests(unittest.TestCase):
    def test_observer_lifecycle_summary_and_live_updates_replace_console_heartbeats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = RecordingObserver(print_samples=True)
            recorder = prepare.Recorder(root, root, {'commands': []})
            with patch.object(prepare, 'OUTPUT_POLL_SECONDS', .01), \
                 patch.object(prepare, 'PROGRESS_INTERVAL_SECONDS', .04), redirect_stdout(io.StringIO()) as terminal:
                result = recorder.command('generate', [sys.executable, '-c', 'import time; time.sleep(.15)'],
                                          root, 5, observer=observer)
            self.assertEqual((result['status'], result['exit_code']), ('succeeded', 0))
            self.assertEqual(observer.calls[0], 'start')
            self.assertEqual(observer.calls[-1], 'finish')
            self.assertEqual(observer.calls.count('finish'), 1)
            self.assertGreater(result['resource_summary']['samples'], 1)
            self.assertEqual(result['resource_summary']['pid'], result['child_pid'])
            saved = json.loads((root / 'manifest.json').read_text())['commands'][0]
            self.assertEqual(saved['resource_summary'], result['resource_summary'])
            self.assertIn('resource sample', terminal.getvalue())
            self.assertNotIn('\n进行中：', terminal.getvalue())
            self.assertIn('进行中：', (root / 'progress.log').read_text())

    def test_observer_failures_do_not_change_command_success_and_report_once(self):
        for action in ('start', 'poll', 'finish'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                observer = RecordingObserver(failing_action=action)
                recorder = prepare.Recorder(root, root, {'commands': []})
                with patch.object(prepare, 'OUTPUT_POLL_SECONDS', .01), \
                     patch.object(prepare, 'PROGRESS_INTERVAL_SECONDS', .04), redirect_stdout(io.StringIO()) as terminal:
                    result = recorder.command('generate', [sys.executable, '-c', 'import time; time.sleep(.15)'],
                                              root, 5, observer=observer)
                self.assertEqual((result['status'], result['exit_code']), ('succeeded', 0))
                self.assertEqual(observer.calls.count(action), 1)
                self.assertEqual(observer.calls.count('finish'), 1)
                self.assertEqual(len(result['resource_monitor_errors']), 1)
                self.assertEqual(result['resource_monitor_errors'][0]['action'], action)
                self.assertEqual(terminal.getvalue().count('monitor fixture failure'), 1)
                self.assertIn('\n进行中：', terminal.getvalue())
                if action == 'start':
                    self.assertNotIn('poll', observer.calls)
                if action != 'finish':
                    self.assertIn('resource_summary', result)

    def test_bad_observer_summary_does_not_break_manifest_or_exit_status(self):
        for summary in (None, {'value': object()}, {'value': float('nan')}):
            with self.subTest(summary=summary), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                recorder = prepare.Recorder(root, root, {'commands': []})
                observer = RecordingObserver()
                with patch.object(observer, 'finish', return_value=summary), redirect_stdout(io.StringIO()):
                    result = recorder.command('generate', [sys.executable, '-c', 'pass'], root, 5, observer=observer)
                self.assertEqual(result['exit_code'], 0)
                self.assertNotIn('resource_summary', result)
                self.assertEqual(result['resource_monitor_errors'][0]['action'], 'finish')
                self.assertEqual(json.loads((root / 'manifest.json').read_text())['commands'][0]['exit_code'], 0)

    def test_observer_finishes_without_start_when_process_cannot_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = RecordingObserver()
            recorder = prepare.Recorder(root, root, {'commands': []})
            with redirect_stdout(io.StringIO()):
                result = recorder.command('generate', [str(root / 'missing-executable')], root, 5, observer=observer)
            self.assertEqual((result['status'], result['exit_code']), ('launch_error', 127))
            self.assertEqual(observer.calls, ['finish'])
            self.assertEqual(result['resource_summary'], {'samples': 0, 'pid': None})

    @unittest.skipUnless(os.name == 'posix', 'POSIX process cleanup')
    def test_observer_finishes_after_timeout_kills_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = RecordingObserver()
            recorder = prepare.Recorder(root, root, {'commands': []})
            original_finish = observer.finish

            def finish():
                with self.assertRaises(ProcessLookupError):
                    os.kill(observer.pid, 0)
                return original_finish()

            with patch.object(observer, 'finish', side_effect=finish), \
                 patch.object(prepare, 'OUTPUT_POLL_SECONDS', .01), redirect_stdout(io.StringIO()):
                result = recorder.command('generate', [sys.executable, '-c', 'import time; time.sleep(10)'],
                                          root, .1, observer=observer)
            self.assertEqual((result['status'], result['exit_code']), ('timed_out', 124))
            self.assertEqual(observer.calls[-1], 'finish')
            self.assertEqual(observer.calls.count('finish'), 1)
            self.assertGreater(result['resource_summary']['samples'], 0)
            self.assertNotIn('resource_monitor_errors', result)

    @unittest.skipUnless(os.name == 'posix', 'POSIX process cleanup')
    def test_observer_finishes_after_keyboard_interrupt_kills_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = RecordingObserver()
            recorder = prepare.Recorder(root, root, {'commands': []})
            original_finish = observer.finish

            def finish():
                with self.assertRaises(ProcessLookupError):
                    os.kill(observer.pid, 0)
                return original_finish()

            class InterruptOutput(io.StringIO):
                interrupted = False

                def write(self, text):
                    result = super().write(text)
                    if text == 'interrupt marker\n' and not self.interrupted:
                        self.interrupted = True
                        raise KeyboardInterrupt
                    return result

            with patch.object(observer, 'finish', side_effect=finish), \
                 redirect_stdout(InterruptOutput()), self.assertRaises(KeyboardInterrupt):
                recorder.command('generate', [sys.executable, '-c',
                    'import time; print("interrupt marker", flush=True); time.sleep(10)'], root, 5, observer=observer)
            result = json.loads((root / 'manifest.json').read_text())['commands'][0]
            self.assertEqual((result['status'], result['exit_code']), ('interrupted', 130))
            self.assertEqual(observer.calls[-1], 'finish')
            self.assertEqual(observer.calls.count('finish'), 1)
            self.assertIn('resource_summary', result)
            self.assertNotIn('resource_monitor_errors', result)

    def test_split_escape_sequences_do_not_corrupt_progress_text(self):
        display = TerminalText()
        text = display.feed('\x1b[3') + display.feed('2mtorch 25%\x1b[0m\r')
        text += display.feed('\x1b]0;title\x1b') + display.feed('\\torch 50%\n')
        self.assertEqual(text, 'torch 25%\ntorch 50%\n')
        self.assertEqual(display.latest, 'torch 50%')

    def test_redrawn_uv_rows_do_not_accumulate_in_latest_output(self):
        display = TerminalText()
        for value in (10, 20, 30):
            display.feed('\x1b[1A\r\x1b[2K\x1b[1B\r\x1b[2K\x1b[1A')
            display.feed('Preparing packages... (0/1)' + ' ' * 80 +
                         'torch' + ' ' * 16 + f'--- {value} MiB/5.79 GiB' + ' ' * 40)
        self.assertEqual(display.latest, 'torch --- 30 MiB/5.79 GiB')

    def test_active_output_does_not_get_duplicate_console_heartbeats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = 'import time\nfor i in range(40):\n print(f"download {i}/40",flush=True)\n time.sleep(.01)\n'
            recorder = prepare.Recorder(root, root, {'commands': []})
            with patch.object(prepare, 'OUTPUT_POLL_SECONDS', .005), \
                 patch.object(prepare, 'PROGRESS_INTERVAL_SECONDS', .1), redirect_stdout(io.StringIO()) as terminal:
                result = recorder.command('download', [sys.executable, '-c', code], root, 5)
            self.assertEqual(result['exit_code'], 0)
            self.assertIn('download 39/40', terminal.getvalue())
            self.assertNotIn('\n进行中：', terminal.getvalue())
            self.assertNotIn('最近输出：', terminal.getvalue())
            self.assertIn('进行中：', (root / 'progress.log').read_text())

    @unittest.skipUnless(os.name == 'posix', 'POSIX PTY')
    def test_terminal_progress_is_visible_before_completion_and_raw_log_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / 'release'
            executable = root / 'uv'
            executable.write_text('#!' + sys.executable + '\n'
                'import os,sys,time\nfrom pathlib import Path\n'
                'assert sys.stdout.isatty() and sys.stderr.isatty()\n'
                'assert "RUST_LOG" not in os.environ\n'
                'os.write(1,b"\\x1b[32mtorch 25%\\x1b[0m\\r" + "中".encode()[:2])\n'
                'deadline=time.monotonic()+3\n'
                f'while not Path({str(release)!r}).exists():\n'
                ' if time.monotonic()>deadline: sys.exit(9)\n'
                ' time.sleep(.01)\n'
                'os.write(2,"中".encode()[2:]+b" 100%\\n")\n')
            executable.chmod(0o755)

            class LiveOutput(io.StringIO):
                def write(self, text):
                    result = super().write(text)
                    if 'torch 25%' in self.getvalue():
                        release.touch()
                    return result

            terminal = LiveOutput()
            recorder = prepare.Recorder(root, root, {'commands': []})
            with patch.dict(os.environ, {'RUST_LOG': 'warn'}), redirect_stdout(terminal):
                result = recorder.command('runtime-dependencies', [str(executable), 'pip', 'install', 'fixture'], root, 5)
            self.assertEqual(result['exit_code'], 0)
            self.assertEqual(result['output_mode'], 'pty')
            self.assertEqual(result['environment_removed'], ['RUST_LOG'])
            self.assertNotIn('\x1b', terminal.getvalue())
            self.assertIn('中 100%', terminal.getvalue())
            self.assertEqual((root / result['log']).read_bytes(), b'\x1b[32mtorch 25%\x1b[0m\r' + '中 100%\n'.encode())
            self.assertIn('步骤完成', (root / 'progress.log').read_text())
            self.assertEqual(json.loads((root / 'manifest.json').read_text())['commands'][0]['last_output'], '中 100%')

    @unittest.skipUnless(os.name == 'posix', 'POSIX PTY')
    def test_pty_timeout_preserves_partial_output_and_exit_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / 'uv'
            executable.write_text('#!' + sys.executable + '\n'
                                  'import time\nprint("Building 3/10",flush=True)\ntime.sleep(10)\n')
            executable.chmod(0o755)
            recorder = prepare.Recorder(root, root, {'commands': []})
            with redirect_stdout(io.StringIO()), patch.object(prepare, 'OUTPUT_POLL_SECONDS', .01):
                result = recorder.command('compile', [str(executable), 'pip', 'install', 'fixture'], root, .3)
            self.assertEqual((result['exit_code'], result['status']), (124, 'timed_out'))
            self.assertEqual((root / result['log']).read_bytes(), b'Building 3/10\n')
            self.assertIn('步骤超时', (root / 'progress.log').read_text())

    @unittest.skipUnless(os.name == 'posix', 'POSIX PTY')
    def test_unavailable_pty_falls_back_to_live_file_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / 'uv'
            executable.write_text('#!' + sys.executable + '\nprint("fallback output",flush=True)\n')
            executable.chmod(0o755)
            recorder = prepare.Recorder(root, root, {'commands': []})
            with patch('pty.openpty', side_effect=OSError('unavailable')), redirect_stdout(io.StringIO()) as terminal:
                result = recorder.command('venv', [str(executable), 'venv'], root, 5)
            self.assertEqual(result['exit_code'], 0)
            self.assertEqual(result['output_mode'], 'file')
            self.assertIn('fallback output', terminal.getvalue())
            self.assertIn('PTY 不可用', terminal.getvalue())

    @unittest.skipUnless(os.name == 'posix', 'POSIX PTY')
    def test_pty_interrupt_keeps_output_and_saves_interrupted_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / 'uv'
            executable.write_text('#!' + sys.executable + '\n'
                                  'import time\nprint("interrupt marker",flush=True)\ntime.sleep(10)\n')
            executable.chmod(0o755)

            class InterruptOutput(io.StringIO):
                interrupted = False
                def write(self, text):
                    result = super().write(text)
                    if text == 'interrupt marker\n' and not self.interrupted:
                        self.interrupted = True
                        raise KeyboardInterrupt
                    return result

            recorder = prepare.Recorder(root, root, {'commands': []})
            with redirect_stdout(InterruptOutput()), self.assertRaises(KeyboardInterrupt):
                recorder.command('runtime', [str(executable), 'pip', 'install', 'fixture'], root, 5)
            result = json.loads((root / 'manifest.json').read_text())['commands'][0]
            self.assertEqual((result['exit_code'], result['status']), (130, 'interrupted'))
            self.assertEqual(result['interruption_reason'], 'keyboard_interrupt')
            self.assertEqual(result['interruption_receiver'], 'init_runner')
            self.assertEqual(result['runner_pid'], os.getpid())
            self.assertIsInstance(result['child_pid'], int)
            self.assertIn('child_returncode_before_cleanup', result)
            self.assertIn('child_returncode', result)
            self.assertIn('KeyboardInterrupt', (root / 'progress.log').read_text())
            self.assertEqual((root / result['log']).read_bytes(), b'interrupt marker\n')
            self.assertLess(result['duration_seconds'], 5)

    def test_slow_hash_reports_real_bytes_without_changing_digest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'library.so'
            payload = b'library fixture'
            path.write_bytes(payload)
            with patch.object(prepare.time, 'monotonic', side_effect=[0, 6, 7]), redirect_stdout(io.StringIO()) as terminal:
                digest = prepare.sha256(path)
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            self.assertIn('100.0%', terminal.getvalue())
            self.assertIn('MiB/s', terminal.getvalue())
            self.assertIn('校验完成', terminal.getvalue())
