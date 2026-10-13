"""Admission checks for local model selection and reproducible preparation."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ai_infra import cli, prepare as prep


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ai-infra-prepare-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / prep.STUDY
        self.study.mkdir(parents=True)
        self.models = self.root / 'models'
        self.models.mkdir()
        self.payload = b'GGUFtest-header-tensor-data'
        self.reference = {'engines': {'ds4': 'a' * 40, 'llama.cpp': 'b' * 40},
                          'artifact': {'filename': 'frozen.gguf', 'size_bytes': len(self.payload),
                                       'sha256_reported_by_hf_lfs': hashlib.sha256(self.payload).hexdigest()},
                          'header': {'tensor_directory_end': 15, 'tensor_data_start': 16,
                                     'header_sha256': hashlib.sha256(self.payload[:16]).hexdigest()}}
        (self.study / 'R0_MODEL.json').write_text(json.dumps(self.reference))
        prompt = 'x\r\n\t  '
        case = {'id': 'whitespace', 'prompt': prompt, 'candidate_rendered_prompt': prompt,
                'prompt_utf8_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'rendered_utf8_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        (self.study / 'R0_INPUTS.json').write_text(json.dumps({'cases': [case]}))
        self.model = self.models / 'frozen.gguf'
        self.model.write_bytes(self.payload)

    def args(self, *argv):
        parser = argparse.ArgumentParser()
        prep.add_init_arguments(parser)
        return parser.parse_args(['--models-dir', str(self.models), *argv])

    def test_selection_ignores_partial_and_other_supported_families(self):
        (self.models / 'frozen.gguf.incomplete').write_bytes(self.payload)
        (self.models / 'Qwen.gguf').write_bytes(self.payload)
        records, selected = prep.inventory(self.models, self.reference)
        self.assertEqual(selected, self.model)
        self.assertEqual(sum(item['frozen_p0_candidate'] for item in records), 1)
        self.model.unlink()
        self.assertIsNone(prep.inventory(self.models, self.reference)[1])

    def test_full_hash_required_and_header_is_not_whole_file_verification(self):
        partial = prep.verify_model(self.model, self.reference, False)
        self.assertFalse(partial['verified'])
        self.assertNotIn('sha256', partial)
        full = prep.verify_model(self.model, self.reference, True)
        self.assertTrue(full['verified'])
        self.model.write_bytes(self.payload[:-1] + b'X')
        with self.assertRaisesRegex(cli.WorkflowError, 'SHA256'):
            prep.verify_model(self.model, self.reference, True)

    def test_size_header_magic_and_incomplete_rejected(self):
        for data in (b'GGUF', b'BAD!' + self.payload[4:], b'GGUFmodified' + self.payload[12:]):
            with self.subTest(data=data), self.assertRaises(cli.WorkflowError):
                self.model.write_bytes(data)
                prep.verify_model(self.model, self.reference, True)
        incomplete = self.models / 'frozen.gguf.incomplete'
        incomplete.write_bytes(self.payload)
        with self.assertRaises(cli.WorkflowError):
            prep.verify_model(incomplete, self.reference, True)

    def test_freeze_preserves_crlf_tabs_and_trailing_spaces(self):
        output = self.root / 'output'
        output.mkdir()
        prep.freeze_inputs(self.root, output)
        self.assertEqual((output / 'inputs/whitespace.prompt.txt').read_bytes(), b'x\r\n\t  ')
        self.assertIsNone(json.loads((output / 'inputs/R0_INPUTS.json').read_text())['cases'][0].get('actual_token_ids'))

    def test_check_only_never_fetches_hashes_full_file_or_builds(self):
        with patch.object(prep, 'doctor', return_value={'project_commit': 'parent'}), \
             patch.object(prep, 'prepare_sources') as sources, \
             patch.object(prep, 'prepare_support') as support, \
             patch.object(prep, 'build_all') as build, redirect_stdout(io.StringIO()):
            directory, code = prep.prepare(self.root, self.args('--check-only'))
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(code, 2)
        self.assertEqual(manifest['status'], 'inventory_only')
        self.assertFalse(manifest['ready_for_validation'])
        self.assertFalse(manifest['model']['verified'])
        sources.assert_not_called()
        support.assert_not_called()
        build.assert_not_called()

    def test_wrong_model_blocks_before_fetch_or_build_and_preserves_manifest(self):
        self.model.write_bytes(b'corrupt')
        with patch.object(prep, 'doctor', return_value={'project_commit': 'parent'}), \
             patch.object(prep, 'prepare_sources') as sources, redirect_stdout(io.StringIO()):
            directory, code = prep.prepare(self.root, self.args())
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(code, 2)
        self.assertEqual(manifest['status'], 'blocked')
        self.assertTrue(manifest['blockers'])
        sources.assert_not_called()
        self.assertTrue((directory / 'README.md').exists())

    def test_offline_support_requires_matching_revision_receipt(self):
        self.reference['artifact'].update(repo='owner/model', revision='c' * 40)
        cache = self.root / 'artifacts/models' / ('c' * 40)
        cache.mkdir(parents=True)
        (cache / 'support-identity.json').write_text(json.dumps({'api_url': 'wrong', 'filename': prep.SUPPORT}))
        with patch.object(prep.urllib.request, 'urlopen') as network, self.assertRaises(cli.WorkflowError):
            prep.prepare_support(self.root, self.models, self.reference, None, True)
        network.assert_not_called()

    def test_offline_support_reuses_verified_local_file(self):
        self.reference['artifact'].update(repo='owner/model', revision='c' * 40)
        cache = self.root / 'artifacts/models' / ('c' * 40)
        cache.mkdir(parents=True)
        identity = {'api_url': 'https://huggingface.co/api/models/owner/model/revision/' + 'c' * 40 + '?blobs=true',
                    'filename': prep.SUPPORT, 'size_bytes': len(self.payload),
                    'sha256': hashlib.sha256(self.payload).hexdigest()}
        (cache / 'support-identity.json').write_text(json.dumps(identity))
        (self.models / prep.SUPPORT).write_bytes(self.payload)
        with patch.object(prep.urllib.request, 'urlopen') as network:
            result = prep.prepare_support(self.root, self.models, self.reference, None, True)
        self.assertTrue(result['verified'])
        self.assertEqual(result['path'], str(self.models / prep.SUPPORT))
        network.assert_not_called()

    def test_support_resumes_partial_file_and_checks_full_digest(self):
        self.reference['artifact'].update(repo='owner/model', revision='c' * 40)
        cache = self.root / 'artifacts/models' / ('c' * 40)
        cache.mkdir(parents=True)
        pending = cache / (prep.SUPPORT + '.incomplete')
        pending.write_bytes(self.payload[:8])
        api_data = {'sha': 'c' * 40, 'siblings': [{'rfilename': prep.SUPPORT,
                    'lfs': {'size': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest()}}]}

        class Response(io.BytesIO):
            def __init__(self, data, status=200, headers=None):
                super().__init__(data)
                self.status, self.headers = status, headers or {}

        api = Response(json.dumps(api_data).encode())
        body = Response(self.payload[8:], 206, {'Content-Range': f'bytes 8-{len(self.payload)-1}/{len(self.payload)}'})
        with patch.object(prep.urllib.request, 'urlopen', side_effect=[api, body]) as network, redirect_stdout(io.StringIO()):
            result = prep.prepare_support(self.root, self.models, self.reference, None, False)
        self.assertEqual(network.call_args_list[1].args[0].get_header('Range'), 'bytes=8-')
        self.assertTrue(result['verified'])
        self.assertEqual(Path(result['path']).read_bytes(), self.payload)
        self.assertFalse(pending.exists())

    def test_support_does_not_append_when_server_ignores_range(self):
        self.reference['artifact'].update(repo='owner/model', revision='c' * 40)
        cache = self.root / 'artifacts/models' / ('c' * 40)
        cache.mkdir(parents=True)
        pending = cache / (prep.SUPPORT + '.incomplete')
        pending.write_bytes(self.payload[:8])
        api_data = {'sha': 'c' * 40, 'siblings': [{'rfilename': prep.SUPPORT,
                    'lfs': {'size': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest()}}]}
        class Response(io.BytesIO):
            status = 200
            headers = {}
        with patch.object(prep.urllib.request, 'urlopen', side_effect=[Response(json.dumps(api_data).encode()), Response(self.payload)]), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(cli.WorkflowError, 'Range'):
            prep.prepare_support(self.root, self.models, self.reference, None, False)
        self.assertEqual(pending.read_bytes(), self.payload[:8])

    def test_command_timeout_and_launch_error_saved(self):
        output = self.root / 'logs'
        output.mkdir()
        manifest = {'commands': []}
        recorder = prep.Recorder(self.root, output, manifest)
        with redirect_stdout(io.StringIO()):
            timeout = recorder.command('timeout', [sys.executable, '-c', 'import time; time.sleep(20)'], self.root, .05)
            missing = recorder.command('missing', [str(self.root / 'missing')], self.root)
        self.assertEqual(timeout['exit_code'], 124)
        self.assertEqual(missing['exit_code'], 127)
        self.assertEqual(len(json.loads((output / 'manifest.json').read_text())['commands']), 2)

    def test_scorer_adapter_changes_only_load_api_and_rejects_unknown_source(self):
        source = self.root / 'score_llama.cpp'
        original = 'before\n    model_params.use_mmap = true;\nafter\n'
        source.write_text(original)
        with patch.object(prep, 'SCORE_LLAMA_SHA256', hashlib.sha256(original.encode()).hexdigest()):
            record = prep.adapt_score_llama(source, self.root / 'adapter')
        self.assertEqual(source.read_text(), original)
        self.assertEqual(Path(record['adapted_source']).read_text(),
                         'before\n    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;\nafter\n')
        self.assertIn('-    model_params.use_mmap = true;', Path(record['patch']).read_text())
        with self.assertRaisesRegex(cli.WorkflowError, 'hash'):
            prep.adapt_score_llama(source, self.root / 'rejected')

    def test_prepared_directory_cannot_be_resumed_or_overwritten(self):
        directory = self.root / 'artifacts/init/completed'
        directory.mkdir(parents=True)
        data = json.dumps({'preparation_id': 'ds4-p0', 'schema_version': 1, 'status': 'prepared'})
        (directory / 'manifest.json').write_text(data)
        with self.assertRaisesRegex(cli.WorkflowError, 'prepared'):
            prep.prepare(self.root, self.args('--resume', str(directory)))
        self.assertEqual((directory / 'manifest.json').read_text(), data)
        self.assertFalse(list(directory.glob('attempt-*.json')))

    def test_resume_accepts_repaired_tools_and_preserves_failed_history(self):
        directory = self.root / 'artifacts/init/failed'
        directory.mkdir(parents=True)
        cuda = self.root / 'toolkit'
        (cuda / 'bin').mkdir(parents=True)
        (cuda / 'bin/nvcc').write_text('mock compiler')
        old = {'preparation_id': 'ds4-p0', 'schema_version': 1, 'status': 'blocked',
               'parent_commit': 'parent', 'cuda_home': str(cuda),
               'inputs': {name: {'sha256': hashlib.sha256((self.study / name).read_bytes()).hexdigest()}
                          for name in ('R0_MODEL.json', 'R0_INPUTS.json')},
               'commands': [{'label': 'version-cc', 'exit_code': 127}], 'builds': {}}
        (directory / 'manifest.json').write_text(json.dumps(old))
        # This tests admission orchestration; no real compiler/GPU is exercised.
        with patch.object(prep, 'doctor', return_value={'project_commit': 'parent'}), \
             patch.object(prep, 'prepare_sources', return_value={}), \
             patch.object(prep, 'prepare_support', return_value={'verified': True}), \
             patch.object(prep, 'build_all', return_value={}), \
             patch.object(prep.Recorder, 'command', return_value={'exit_code': 0, 'log': 'mock.log'}), \
             patch.object(prep.platform, 'system', return_value='Linux'), \
             patch.object(prep.platform, 'machine', return_value='aarch64'), \
             patch.object(prep.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024**3, total=40 * 1024**3)), \
             redirect_stdout(io.StringIO()):
            _, code = prep.prepare(self.root, self.args('--resume', str(directory), '--cuda-home', str(cuda)))
        self.assertEqual(code, 0)
        result = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(result['commands'][0]['exit_code'], 127)
        self.assertEqual(json.loads((directory / 'attempt-1.json').read_text()), old)

    def test_builds_use_separate_clones_without_switching_source(self):
        upstream = self.root / 'upstream'
        upstream.mkdir()

        def git(*args):
            return subprocess.run(['git', '-C', str(upstream), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()

        git('init', '-q')
        git('config', 'user.name', 'Test Runner')
        git('config', 'user.email', 'test@example.invalid')
        (upstream / '.gitignore').write_text('ds4\nds4-server\nds4-bench\nbuild-p0/\ngguf-tools/quality-testing/score_*\n')
        (upstream / 'source.txt').write_text('old')
        git('add', '.')
        git('commit', '-qm', 'old')
        old = git('rev-parse', 'HEAD')
        (upstream / 'source.txt').write_text('new')
        git('commit', '-qam', 'new')
        newer = git('rev-parse', 'HEAD')
        output = self.root / 'build-output'
        output.mkdir()
        sources = {name: {'path': str(upstream), 'commit': newer} for name in ('ds4', 'llama.cpp')}
        recorder = prep.Recorder(self.root, output, {'commands': []})

        def fake_build(label, argv, cwd, timeout, env=None):
            if argv[0] == 'git':
                subprocess.run(argv, cwd=cwd, check=True, capture_output=True)
            elif label.startswith('build-ds4-'):
                for name in ('ds4', 'ds4-server', 'ds4-bench'):
                    (cwd / name).write_bytes(b'fake-binary')
            elif label == 'build-llama':
                target = cwd / 'build-p0/bin'
                target.mkdir(parents=True)
                for name in ('llama-cli', 'llama-server', 'llama-bench', 'llama-tokenize'):
                    (target / name).write_bytes(b'fake-binary')
            elif label.startswith('build-score-'):
                target = cwd / 'gguf-tools/quality-testing'
                target.mkdir(parents=True, exist_ok=True)
                name = 'score_llama' if label.endswith('adapted') else label.removeprefix('build-').replace('-', '_')
                (target / name).write_bytes(b'fake-scorer')
            log = label + '.log'
            (output / log).write_text(str(output / 'llama/build-p0/bin/libllama.so'))
            return {'exit_code': 0, 'log': log}

        with patch.object(prep, 'OLD_DS4', old), patch.object(recorder, 'require', side_effect=fake_build), \
             patch.object(prep, 'adapt_score_llama', return_value={'adapted_source': 'fake-adapter.cpp'}):
            result = prep.build_all(self.root, sources, self.root / 'cuda', self.args(), recorder)
        self.assertEqual((output / 'ds4-old/source.txt').read_text(), 'old')
        self.assertEqual((output / 'ds4-new/source.txt').read_text(), 'new')
        self.assertEqual(git('rev-parse', 'HEAD'), newer)
        self.assertEqual(git('status', '--porcelain'), '')
        self.assertEqual(result['ds4-old']['commit'], old)
        self.assertEqual(result['ds4-new']['status'], 'built')
        with patch.object(prep, 'OLD_DS4', old), patch.object(recorder, 'require', side_effect=fake_build) as run, \
             patch.object(prep, 'adapt_score_llama', return_value={'adapted_source': 'fake-adapter.cpp'}), redirect_stdout(io.StringIO()):
            prep.build_all(self.root, sources, self.root / 'cuda', self.args(), recorder)
        self.assertFalse(any(call.args[0].startswith(('clone-', 'checkout-', 'build-ds4-', 'build-llama'))
                             for call in run.call_args_list))
        (output / 'ds4-new/ds4').write_bytes(b'modified')
        with patch.object(prep, 'OLD_DS4', old), redirect_stdout(io.StringIO()), self.assertRaisesRegex(cli.WorkflowError, '二进制身份'):
            prep.build_all(self.root, sources, self.root / 'cuda', self.args(), recorder)

    def test_console_entry_point_forwards_arguments(self):
        with patch.object(sys, 'argv', ['init', '--offline']), patch.object(cli, 'main', return_value=2) as main:
            self.assertEqual(cli.init_main(), 2)
        main.assert_called_once_with(['init', '--offline'])

    def test_budget_flags_reject_nonpositive_values(self):
        for flag in ('--jobs', '--build-timeout', '--download-timeout'):
            with redirect_stdout(io.StringIO()), patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
                self.args(flag, '0')


if __name__ == '__main__':
    unittest.main()
