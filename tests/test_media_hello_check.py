"""Image dimensions follow the run configuration without importing GPU libraries."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from lab_tools import media_hello_check as check


class MediaHelloCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'output'
        self.output.mkdir()
        self.result = self.root / 'result.json'

    def image_decoder(self, width, height, *, image_format='PNG', extrema=None):
        (self.output / 'poster.png').write_bytes(b'fixture image bytes')
        image = Mock(format=image_format, size=(width, height), width=width, height=height)
        image.convert.return_value.getextrema.return_value = extrema or ((1, 240), (2, 220), (3, 200))
        manager = Mock()
        manager.__enter__ = Mock(return_value=image)
        manager.__exit__ = Mock(return_value=False)
        decoder = SimpleNamespace(Image=SimpleNamespace(open=Mock(return_value=manager)))
        return patch.dict('sys.modules', {'PIL': decoder})

    def invoke(self, *args):
        argv = ['media_hello_check.py', *args, '--result', str(self.result)]
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return check.main()

    def test_configured_portrait_dimensions_pass(self):
        with self.image_decoder(1344, 2016):
            result = check.validate('ideogram4', self.output,
                                    expected_width=1344, expected_height=2016)
        self.assertTrue(result['passed'])
        self.assertEqual((result['width'], result['height']), (1344, 2016))
        self.assertEqual(result['output_path'], str(self.output / 'poster.png'))
        self.assertEqual(len(result['sha256']), 64)

    def test_old_small_image_fails_new_dimensions(self):
        with self.image_decoder(256, 256):
            with self.assertRaisesRegex(ValueError, '预期 1344×2016，实际 256×256'):
                check.validate('ideogram4', self.output,
                               expected_width=1344, expected_height=2016)

    def test_dimensions_are_required_positive_integers(self):
        for width, height in [(None, None), (1344, None), (None, 2016), (0, 2016),
                              (1344, -1), (True, 2016), (1344, 2016.0)]:
            with self.subTest(width=width, height=height):
                with self.assertRaisesRegex(ValueError, '正整数'):
                    check.validate('ideogram4', self.output,
                                   expected_width=width, expected_height=height)

    def test_cli_checks_the_provided_dimensions(self):
        with self.image_decoder(1344, 2016):
            code = self.invoke('--app', 'ideogram4', '--output', str(self.output),
                               '--expected-width', '1344', '--expected-height', '2016')
        self.assertEqual(code, 0)
        result = json.loads(self.result.read_text())
        self.assertEqual((result['width'], result['height']), (1344, 2016))

    def test_cli_rejects_missing_or_nonpositive_dimensions_before_decode(self):
        cases = [[], ['--expected-width', '1344'], ['--expected-height', '2016'],
                 ['--expected-width', '0', '--expected-height', '2016'],
                 ['--expected-width', '1344', '--expected-height', '-1'],
                 ['--expected-width', '1344.5', '--expected-height', '2016']]
        with patch.object(check, 'validate') as validate:
            for dimensions in cases:
                with self.subTest(dimensions=dimensions), self.assertRaises(SystemExit) as error:
                    self.invoke('--app', 'ideogram4', '--output', str(self.output), *dimensions)
                self.assertEqual(error.exception.code, 2)
            validate.assert_not_called()
        self.assertFalse(self.result.exists())

    def test_tts_and_reference_do_not_require_dimensions(self):
        with patch.object(check, 'validate', return_value={'passed': True}) as validate:
            self.assertEqual(self.invoke('--app', 'indextts2', '--output', str(self.output)), 0)
            validate.assert_called_once_with('indextts2', self.output,
                                             expected_width=None, expected_height=None)
        reference = self.root / 'reference.wav'
        with patch.object(check, 'validate_reference', return_value={'duration_seconds': 1}) as validate_reference:
            self.assertEqual(self.invoke('--app', 'indextts2', '--reference', str(reference)), 0)
            validate_reference.assert_called_once_with(reference)


if __name__ == '__main__':
    unittest.main()
