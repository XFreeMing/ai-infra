"""Git progress survives redirected init output; all remotes are local fixtures."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SourceProgressTests(unittest.TestCase):
    def test_update_sources_streams_fetch_progress_when_stderr_is_redirected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, remote = base / 'workspace', base / 'remote'
            root.mkdir()
            remote.mkdir()

            def git(directory, *arguments):
                result = subprocess.run(
                    ['git', '-C', str(directory), *arguments],
                    capture_output=True, text=True, check=True,
                )
                return result.stdout.strip()

            git(remote, 'init', '-b', 'main')
            git(remote, 'config', 'user.name', 'Fixture')
            git(remote, 'config', 'user.email', 'fixture@example.invalid')
            git(remote, 'config', 'commit.gpgsign', 'false')
            (remote / 'version').write_text('one')
            git(remote, 'add', '.')
            git(remote, 'commit', '-m', 'one')
            git(root, 'init')
            (root / '.gitmodules').write_text(
                '[submodule "sources/fixture"]\npath = sources/fixture\nbranch = main\n'
            )
            local = root / 'sources/fixture'
            git(root, 'clone', str(remote), str(local))
            (remote / 'version').write_text('two')
            git(remote, 'commit', '-am', 'two')

            # Pipe stderr deliberately: ordinary `git fetch` suppresses its
            # object counters here unless --progress is passed explicitly.
            result = subprocess.run(
                [sys.executable, '-c',
                 'from pathlib import Path; import sys; '
                 'from lab_tools.source_tracking import update_sources; '
                 'update_sources(Path(sys.argv[1]))', str(root)],
                cwd=Path(__file__).resolve().parents[1], capture_output=True,
                text=True, timeout=20, env={**os.environ, 'LC_ALL': 'C'},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('Counting objects:', result.stderr)
            self.assertIn('Fast-forward', result.stdout)
            self.assertEqual(git(local, 'rev-parse', 'HEAD'), git(remote, 'rev-parse', 'HEAD'))
            self.assertEqual((local / 'version').read_text(), 'two')


if __name__ == '__main__':
    unittest.main()
