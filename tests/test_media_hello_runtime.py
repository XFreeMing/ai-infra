"""Runtime timing uses fake workers/allocators; no model or GPU is needed."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lab_tools import media_hello_runtime as runtime


class FakeCUDA:
    def __init__(self):
        self.events = []
        self.allocated, self.reserved = 10, 20
        self.peak_allocated, self.peak_reserved = 10, 20

    def is_available(self):
        return True

    def synchronize(self):
        self.events.append('sync')

    def reset_peak_memory_stats(self):
        self.events.append('reset')
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved

    def memory_allocated(self):
        return self.allocated

    def memory_reserved(self):
        return self.reserved

    def max_memory_allocated(self):
        return self.peak_allocated

    def max_memory_reserved(self):
        return self.peak_reserved


class MediaHelloRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'runtime-performance.json'
        self.cuda = FakeCUDA()
        patcher = patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=self.cuda)})
        patcher.start()
        self.addCleanup(patcher.stop)
        capture = redirect_stdout(io.StringIO())
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)
        self.metrics = runtime.RuntimeMetrics(self.path)

    def data(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def test_phase_is_published_before_work_and_allocator_peak_is_independent(self):
        def load():
            saved = self.data()
            self.assertEqual(saved['active_phase'], 'model_load')
            self.assertEqual(saved['phases']['model_load']['status'], 'running')
            self.assertEqual(self.cuda.events, ['sync', 'reset'])
            self.cuda.allocated, self.cuda.reserved = 100, 200
            self.cuda.peak_allocated, self.cuda.peak_reserved = 150, 250
            return 'model'

        self.assertEqual(self.metrics.call('model_load', load), 'model')
        loaded = self.data()['phases']['model_load']
        self.assertEqual(loaded['torch_start_allocated_bytes'], 10)
        self.assertEqual(loaded['torch_allocated_bytes'], 100)
        self.assertEqual(loaded['torch_peak_allocated_bytes'], 150)
        self.assertEqual(loaded['torch_peak_reserved_bytes'], 250)
        self.metrics.call('generation', lambda: 'image')
        generated = self.data()['phases']['generation']
        self.assertEqual(generated['torch_start_allocated_bytes'], 100)
        self.assertEqual(generated['torch_peak_allocated_bytes'], 100)
        self.assertEqual(generated['status'], 'succeeded')
        self.assertTrue(generated['duration_complete'])
        self.assertGreaterEqual(generated['duration_seconds'], 0)
        self.assertEqual(self.cuda.events, ['sync', 'reset', 'sync'] * 2)
        self.assertIsNone(self.data()['active_phase'])
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_error_preserves_partial_data_without_completion_gpu_wait(self):
        def fail():
            self.cuda.allocated = 123
            raise ValueError('fixture failure')

        with self.assertRaisesRegex(ValueError, 'fixture failure'):
            self.metrics.call('generation', fail)
        phase = self.data()['phases']['generation']
        self.assertEqual(phase['status'], 'failed')
        self.assertFalse(phase['duration_complete'])
        self.assertFalse(phase['gpu_synchronized_at_end'])
        self.assertEqual(phase['torch_allocated_bytes'], 123)
        self.assertEqual(self.cuda.events, ['sync', 'reset'])
        self.assertEqual(self.data()['errors'][0]['type'], 'ValueError')

    def test_keyboard_interrupt_and_sigterm_exit_are_retained(self):
        for exception in (KeyboardInterrupt(), SystemExit(143)):
            with self.subTest(error=exception):
                def interrupted():
                    raise exception
                with self.assertRaises(type(exception)):
                    self.metrics.call('generation', interrupted)
                self.assertEqual(self.data()['phases']['generation']['status'], 'interrupted')
                self.assertIsNone(self.data()['active_phase'])

    def test_unavailable_cuda_has_null_measurements(self):
        with patch.object(self.cuda, 'is_available', return_value=False):
            self.metrics.call('runtime_check', lambda: 28)
        phase = self.data()['phases']['runtime_check']
        self.assertIsNone(phase['torch_allocated_bytes'])
        self.assertFalse(phase['gpu_synchronized_at_start'])
        self.assertFalse(phase['gpu_synchronized_at_end'])

    def test_measurement_error_does_not_mask_worker_result(self):
        with patch.object(self.cuda, 'memory_allocated', side_effect=RuntimeError('allocator unavailable')):
            self.assertEqual(self.metrics.call('generation', lambda: 42), 42)
        phase = self.data()['phases']['generation']
        self.assertEqual(phase['status'], 'succeeded')
        self.assertIsNone(phase['torch_allocated_bytes'])
        self.assertTrue(any('allocator unavailable' in item for item in phase['measurement_warnings']))

    def test_instrumented_image_descriptors_preserve_arguments_return_and_class(self):
        calls = []

        class Pipeline:
            @classmethod
            def from_pretrained(cls, *, config, device):
                calls.append((cls, config, device))
                return cls()

            def __call__(self, caption, **kwargs):
                calls.append((caption, kwargs))
                return ['image']

        module = SimpleNamespace(Ideogram4Pipeline=Pipeline)
        worker = SimpleNamespace(torch_probe=lambda app: {'app': app}, import_app=lambda source, app: module)
        runtime.instrument_worker(worker, self.metrics)
        self.assertEqual(worker.torch_probe('ideogram4'), {'app': 'ideogram4'})
        self.assertIs(worker.import_app(self.root, 'ideogram4'), module)
        # Calling import_app twice must not stack wrappers or repeat a phase.
        worker.import_app(self.root, 'ideogram4')

        class Derived(Pipeline):
            pass

        pipe = Derived.from_pretrained(config='fixed', device='cuda')
        self.assertIsInstance(pipe, Derived)
        self.assertEqual(pipe('literal caption', height=256, width=256, num_steps=12, seed=0), ['image'])
        self.assertEqual(calls[0], (Derived, 'fixed', 'cuda'))
        self.assertEqual(calls[1][0], 'literal caption')
        self.assertEqual(len(self.cuda.events), 9)
        generated = self.data()['phases']['generation']
        self.assertEqual(generated['request'], {'height': 256, 'width': 256, 'num_steps': 12, 'seed': 0})

    def test_static_loader_descriptor_is_preserved(self):
        class Pipeline:
            @staticmethod
            def from_pretrained(*, device):
                return device

        runtime.instrument_method(Pipeline, 'from_pretrained', self.metrics, 'model_load')
        self.assertEqual(Pipeline.from_pretrained(device='cuda'), 'cuda')
        self.assertIsInstance(Pipeline.__dict__['from_pretrained'], staticmethod)

    def test_tts_constructor_and_infer_preserve_side_effects(self):
        class TTS:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def infer(self, *, text, output_path):
                Path(output_path).write_text(text)
                return 123

        module = SimpleNamespace(IndexTTS2=TTS)
        worker = SimpleNamespace(torch_probe=lambda app: {}, import_app=lambda source, app: module)
        runtime.instrument_worker(worker, self.metrics)
        tts = worker.import_app(self.root, 'indextts2').IndexTTS2(device='cuda:0', use_fp16=False)
        output = self.root / 'audio.wav'
        self.assertEqual(tts.infer(text='你好', output_path=output), 123)
        self.assertEqual(output.read_text(), '你好')
        self.assertEqual(tts.kwargs, {'device': 'cuda:0', 'use_fp16': False})
        self.assertEqual(set(self.data()['phases']), {'model_load', 'generation'})

    def test_main_loads_original_worker_and_restores_sigterm_handler(self):
        worker = SimpleNamespace(torch_probe=lambda app: {}, import_app=lambda source, app: None)
        seen = []

        def worker_main(argv):
            seen.append(argv)
            worker.torch_probe('ideogram4')
            return 0

        worker.main = worker_main
        previous_handler = signal.getsignal(signal.SIGTERM)
        with patch.object(runtime, 'load_worker', return_value=worker) as load:
            result = runtime.main(['--worker', str(self.root / 'worker.py'), '--metrics', str(self.path), '--', 'run', '--app', 'ideogram4'])
        self.assertEqual(result, 0)
        load.assert_called_once_with(self.root / 'worker.py')
        self.assertEqual(seen, [['run', '--app', 'ideogram4']])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)
        self.assertEqual(self.data()['status'], 'succeeded')

    def test_main_worker_import_error_writes_failed_metrics(self):
        with self.assertRaises(FileNotFoundError):
            runtime.main(['--worker', str(self.root / 'missing.py'), '--metrics', str(self.path), '--', 'run'])
        self.assertEqual(self.data()['status'], 'failed')
        self.assertEqual(self.data()['errors'][0]['type'], 'FileNotFoundError')

    def test_main_sigterm_saves_interrupted_phase_and_restores_handler(self):
        worker = SimpleNamespace(torch_probe=lambda app: signal.raise_signal(signal.SIGTERM),
                                 import_app=lambda source, app: None)
        worker.main = lambda argv: worker.torch_probe('ideogram4')
        previous_handler = signal.getsignal(signal.SIGTERM)
        with patch.object(runtime, 'load_worker', return_value=worker):
            with self.assertRaises(SystemExit) as caught:
                runtime.main(['--worker', str(self.root / 'worker.py'), '--metrics', str(self.path), '--', 'run'])
        self.assertEqual(caught.exception.code, 143)
        self.assertEqual(self.data()['status'], 'interrupted')
        self.assertEqual(self.data()['phases']['runtime_check']['status'], 'interrupted')
        self.assertFalse(self.data()['phases']['runtime_check']['duration_complete'])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_copied_standalone_wrapper_uses_worker_owned_import_path_under_isolation(self):
        # The wrapper is outside the package, as it will be in a hello artifact.
        wrapper = self.root / 'run' / 'wrapper.py'
        wrapper.parent.mkdir()
        wrapper.write_bytes(Path(runtime.__file__).read_bytes())
        prepared = self.root / 'prepared'
        prepared.mkdir()
        (prepared / 'worker_local_module.py').write_text('VALUE = 28\n')
        worker = prepared / 'worker.py'
        worker.write_text('''from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker_local_module
def torch_probe(app):
    return worker_local_module.VALUE
def import_app(source, app):
    return None
def main(argv):
    assert argv == ['run', '--app', 'ideogram4']
    assert torch_probe('ideogram4') == 28
    return 0
''')
        (prepared / 'torch.py').write_text('''class CUDA:
    def is_available(self):
        return False
cuda = CUDA()
''')
        completed = subprocess.run([sys.executable, '-I', str(wrapper), '--worker', str(worker), '--metrics', str(self.path),
                                    '--', 'run', '--app', 'ideogram4'], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.data()['status'], 'succeeded')
        self.assertEqual(self.data()['phases']['runtime_check']['status'], 'succeeded')


if __name__ == '__main__':
    unittest.main()
