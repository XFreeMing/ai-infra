"""Pinned language-resource downloads, safe extraction, and offline reuse."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError
from urllib.request import Request
import zipfile

from lab_tools import nltk_prepare as prepare


class Response(io.BytesIO):
    def __init__(self, payload, url):
        super().__init__(payload)
        self.url = url

    def geturl(self):
        return self.url


class NltkPrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / 'cache'
        self.destination = self.root / 'nltk_data'
        self.output = io.StringIO()
        self.capture = redirect_stdout(self.output)
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def fixture(self, members=None):
        members = members or [('lexicon/', b''), ('lexicon/words.txt', b'hello world\n')]
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            for name, content in members:
                archive.writestr(name, content)
        payload = stream.getvalue()
        package = {
            'id': 'lexicon', 'subdir': 'corpora',
            'url': 'https://raw.githubusercontent.com/nltk/nltk_data/' + 'a' * 40 + '/packages/corpora/lexicon.zip',
            'size': len(payload), 'unzipped_size': sum(len(content) for _, content in members),
            'sha256_checksum': hashlib.sha256(payload).hexdigest(),
            'expected_files': ['corpora/lexicon/words.txt'],
        }
        return package, payload

    def cache_path(self, package):
        return self.cache / package['sha256_checksum'] / (package['id'] + '.zip')

    def cached(self, package, payload):
        archive = self.cache_path(package)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(payload)
        return archive

    def test_catalog_pins_official_urls_and_archive_identity(self):
        data = prepare.catalog()
        self.assertRegex(data['commit'], r'^[a-f0-9]{40}$')
        self.assertIn('cmudict', {package['id'] for package in data['packages']})
        for package in data['packages']:
            self.assertEqual(package['url'], 'https://raw.githubusercontent.com/nltk/nltk_data/' +
                             data['commit'] + '/packages/' + package['subdir'] + '/' + package['id'] + '.zip')
            self.assertRegex(package['sha256_checksum'], r'^[a-f0-9]{64}$')
            self.assertGreater(package['size'], 0)
            self.assertTrue(package['expected_files'])

    def test_catalog_rejects_unpinned_download_host(self):
        package, _ = self.fixture()
        package['url'] = 'https://untrusted.invalid/lexicon.zip'
        with patch.object(Path, 'read_text', return_value=json.dumps({'commit': 'a' * 40, 'packages': [package]})):
            with self.assertRaisesRegex(ValueError, '官方仓库 commit'):
                prepare.catalog()

    def test_valid_cache_works_offline_without_opening_network(self):
        package, payload = self.fixture()
        archive = self.cached(package, payload)
        with patch.object(prepare, 'build_opener') as network:
            self.assertEqual(prepare.acquire(package, self.cache, offline=True), archive)
        network.assert_not_called()
        self.assertIn('缓存校验通过', self.output.getvalue())

    def test_offline_rejects_missing_wrong_size_and_wrong_hash_cache(self):
        package, payload = self.fixture()
        for content in (None, payload[:-1], b'x' * len(payload)):
            with self.subTest(content=None if content is None else len(content)):
                archive = self.cache_path(package)
                if content is not None:
                    self.cached(package, content)
                with patch.object(prepare, 'build_opener') as network:
                    with self.assertRaisesRegex(ValueError, 'offline'):
                        prepare.acquire(package, self.cache, offline=True)
                network.assert_not_called()
                if content is not None:
                    self.assertEqual(archive.read_bytes(), content)

    def test_download_checks_identity_and_reports_byte_progress(self):
        package, payload = self.fixture()
        opener = Mock()
        opener.open.return_value = Response(payload, package['url'])
        with patch.object(prepare, 'build_opener', return_value=opener) as build:
            archive = prepare.acquire(package, self.cache)
        self.assertIsInstance(build.call_args.args[0], prepare.NoRedirect)
        opener.open.assert_called_once_with(package['url'], timeout=30)
        self.assertEqual(archive.read_bytes(), payload)
        self.assertTrue(prepare.verified(archive, package))
        self.assertFalse(list(self.cache.rglob('*.partial')))
        self.assertIn('100.0%', self.output.getvalue())
        self.assertIn('MiB', self.output.getvalue())
        self.assertIn('SHA256 校验通过', self.output.getvalue())

    def test_invalid_download_never_becomes_final_cache(self):
        for kind in ('truncated', 'oversized', 'wrong-hash'):
            with self.subTest(kind=kind):
                package, payload = self.fixture()
                content = {'truncated': payload[:-1], 'oversized': payload + b'x',
                           'wrong-hash': b'x' * len(payload)}[kind]
                opener = Mock()
                opener.open.return_value = Response(content, package['url'])
                with patch.object(prepare, 'build_opener', return_value=opener):
                    with self.assertRaises(ValueError):
                        prepare.acquire(package, self.cache)
                self.assertFalse(self.cache_path(package).exists())
                self.assertFalse(list(self.cache.rglob('*.partial')))

    def test_redirect_handler_rejects_all_redirects(self):
        package, _ = self.fixture()
        for url in ('https://untrusted.invalid/a', package['url']):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, '不接受重定向'):
                prepare.NoRedirect().redirect_request(Request(package['url']), None, 302, '', {}, url)

    def test_changed_response_url_cannot_populate_cache(self):
        package, payload = self.fixture()
        opener = Mock()
        opener.open.return_value = Response(payload, 'https://untrusted.invalid/lexicon.zip')
        with patch.object(prepare, 'build_opener', return_value=opener):
            with self.assertRaisesRegex(ValueError, '地址发生变化'):
                prepare.acquire(package, self.cache)
        self.assertFalse(self.cache_path(package).exists())
        self.assertFalse(list(self.cache.rglob('*.partial')))

    def test_network_failures_retry_then_reuse_successful_response(self):
        package, payload = self.fixture()
        opener = Mock()
        opener.open.side_effect = [URLError('temporary failure'), TimeoutError('timeout'), Response(payload, package['url'])]
        with patch.object(prepare, 'build_opener', return_value=opener):
            archive = prepare.acquire(package, self.cache)
        self.assertEqual(opener.open.call_count, 3)
        self.assertEqual(archive.read_bytes(), payload)
        self.assertFalse(list(self.cache.rglob('*.partial')))
        self.assertIn('下载失败 [1/3]', self.output.getvalue())
        self.assertIn('下载 lexicon [3/3]', self.output.getvalue())

    def test_exhausted_retries_leave_no_partial_or_final_archive(self):
        package, _ = self.fixture()
        opener = Mock()
        opener.open.side_effect = URLError('unreachable')
        with patch.object(prepare, 'build_opener', return_value=opener):
            with self.assertRaises(URLError):
                prepare.acquire(package, self.cache)
        self.assertEqual(opener.open.call_count, 3)
        self.assertFalse(self.cache_path(package).exists())
        self.assertFalse(list(self.cache.rglob('*.partial')))

    def test_extract_preserves_zip_and_reports_file_count(self):
        package, payload = self.fixture()
        archive = self.cached(package, payload)
        prepare.extract(package, archive, self.destination)
        self.assertEqual((self.destination / 'corpora/lexicon/words.txt').read_bytes(), b'hello world\n')
        self.assertEqual((self.destination / 'corpora/lexicon.zip').read_bytes(), payload)
        self.assertIn('0/2 文件', self.output.getvalue())
        self.assertIn('2/2 文件', self.output.getvalue())

    def test_extract_rejects_hash_mismatch_before_creating_destination(self):
        package, payload = self.fixture()
        archive = self.cached(package, b'x' * len(payload))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            prepare.extract(package, archive, self.destination)
        self.assertFalse(self.destination.exists())

    def test_extract_rejects_unsafe_archive_members_before_writing(self):
        for name in ('../escape', '/absolute', 'lexicon/../../escape', 'lexicon/..\\escape', 'other/words.txt'):
            with self.subTest(name=name):
                package, payload = self.fixture([('lexicon/words.txt', b'safe'), (name, b'unsafe')])
                archive = self.cached(package, payload)
                with self.assertRaisesRegex(ValueError, '不安全'):
                    prepare.extract(package, archive, self.destination)
                self.assertFalse(self.destination.exists())

    def test_extract_rejects_symbolic_link_members(self):
        member = zipfile.ZipInfo('lexicon/words.txt')
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        package, payload = self.fixture([(member, b'../../escape')])
        archive = self.cached(package, payload)
        with self.assertRaisesRegex(ValueError, '不安全'):
            prepare.extract(package, archive, self.destination)
        self.assertFalse(self.destination.exists())

    def test_extract_rejects_existing_symlink_escape(self):
        package, payload = self.fixture()
        archive = self.cached(package, payload)
        outside = self.root / 'outside'
        outside.mkdir()
        (self.destination / 'corpora').mkdir(parents=True)
        (self.destination / 'corpora/lexicon').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '不安全'):
            prepare.extract(package, archive, self.destination)
        self.assertEqual(list(outside.iterdir()), [])

    def test_extract_rejects_unpacked_size_overrun_and_missing_expected_files(self):
        package, payload = self.fixture()
        archive = self.cached(package, payload)
        with self.assertRaisesRegex(ValueError, '解压大小'):
            prepare.extract({**package, 'unzipped_size': 0}, archive, self.destination)
        self.assertFalse(self.destination.exists())
        with self.assertRaisesRegex(ValueError, '缺少文件'):
            prepare.extract({**package, 'expected_files': ['corpora/lexicon/missing']}, archive, self.destination)

    def test_prepare_offline_writes_verified_identity_receipt(self):
        package, payload = self.fixture()
        self.cached(package, payload)
        data = {'commit': 'a' * 40, 'packages': [package]}
        with patch.object(prepare, 'catalog', return_value=data), patch.object(prepare, 'build_opener') as network:
            receipt = prepare.prepare(self.destination, self.cache, offline=True)
        network.assert_not_called()
        self.assertTrue(receipt['verified'])
        self.assertEqual(receipt['directory'], str(self.destination.resolve()))
        self.assertEqual(json.loads((self.destination / 'resource-identity.json').read_text()), receipt)
        self.assertIn('语言运行资源 [1/1]', self.output.getvalue())


if __name__ == '__main__':
    unittest.main()
