"""Behavior checks using disposable local repositories and tiny subprocesses."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from lab_tools import cli


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ai-infra-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "project"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Test Runner")
        self.git("config", "user.email", "test@example.invalid")
        (self.root / ".gitignore").write_text("artifacts/\n__pycache__/\n", encoding="utf-8")
        templates = self.root / "templates"
        templates.mkdir()
        for name in ("reading.md", "experiment.md"):
            (templates / name).write_text("# {{title}}\n{{slug}} {{date}}\n", encoding="utf-8")

    def git(self, *args, root=None):
        return subprocess.run(
            ["git", "-C", str(root or self.root), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def commit(self):
        self.git("add", "--all")
        self.git("commit", "-qm", "Test snapshot")
        return self.git("rev-parse", "HEAD")

    def configure(self, code="print('recorded')", *, timeout=5, sources=None, command=None):
        directory = self.root / "experiments" / "check"
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "id": "check", "command": command or ["{python}", "-c", code],
            "timeout_seconds": timeout, "required_sources": sources or [],
        }
        (directory / "experiment.json").write_text(json.dumps(config), encoding="utf-8")
        return config

    def run_check(self):
        directory, exit_code = cli.run_experiment(self.root, "check", self.root / "artifacts" / "runs")
        return directory, exit_code, json.loads((directory / "manifest.json").read_text())

    def add_submodule(self):
        upstream = self.base / "upstream"
        upstream.mkdir()
        self.git("init", "-q", root=upstream)
        self.git("config", "user.name", "Test Runner", root=upstream)
        self.git("config", "user.email", "test@example.invalid", root=upstream)
        (upstream / "source.txt").write_text("source\n")
        self.git("add", "source.txt", root=upstream)
        self.git("commit", "-qm", "Source", root=upstream)
        self.git("-c", "protocol.file.allow=always", "submodule", "add", str(upstream), "sources/example")
        return self.root / "sources" / "example"

    def test_create_and_refuse_overwrite(self):
        directory = cli.new_item(self.root, "study", "cache-reading")
        self.assertTrue((directory / "README.md").exists())
        self.assertNotIn("{{", (directory / "README.md").read_text())
        with self.assertRaises(cli.WorkflowError):
            cli.new_item(self.root, "study", "cache-reading")
        experiment = cli.new_item(self.root, "experiment", "cache-test")
        config = json.loads((experiment / "experiment.json").read_text())
        self.assertEqual(config["id"], "cache-test")
        self.assertEqual(config["command"], [])

    def test_reject_unsafe_slug_and_symlink_escape(self):
        for slug in ("../escape", "/tmp/escape", "a/b", "", ".", "UPPER", "a" * 65):
            with self.subTest(slug=slug), self.assertRaises(cli.WorkflowError):
                cli.new_item(self.root, "study", slug)
        outside = self.base / "outside"
        outside.mkdir()
        (self.root / "research").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(cli.WorkflowError):
            cli.new_item(self.root, "study", "safe-name")
        self.assertEqual(list(outside.iterdir()), [])

    def test_doctor_reports_unborn_and_committed_repository(self):
        with patch.object(cli.shutil, "which", return_value=None):
            report = cli.doctor(self.root)
            self.assertIsNone(report["project_commit"])
            self.assertFalse(report["nvidia_smi"]["available"])
            expected_commit = self.commit()
            self.assertEqual(cli.doctor(self.root)["project_commit"], expected_commit)

    def test_run_requires_commit_and_clean_tree(self):
        self.configure()
        with self.assertRaises(cli.WorkflowError):
            self.run_check()
        self.commit()
        (self.root / "uncommitted.txt").write_text("local work")
        with self.assertRaises(cli.WorkflowError):
            self.run_check()
        self.assertFalse((self.root / "artifacts").exists())

    def test_success_records_versions_config_and_unique_outputs(self):
        config = self.configure("import sys; print('中文'); sys.stderr.buffer.write(b'raw\\xff')")
        commit = self.commit()
        first, code, manifest = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(manifest["status"], "succeeded")
        self.assertEqual(manifest["project_commit"], commit)
        self.assertEqual(manifest["config"], config)
        self.assertEqual(manifest["command"][0], sys.executable)
        self.assertEqual(manifest["run_environment"]["AI_INFRA_RUN_DIR"], str(first))
        self.assertEqual((first / "stdout.log").read_text(), "中文\n")
        self.assertEqual((first / "stderr.log").read_bytes(), b"raw\xff")
        self.assertIsNotNone(manifest["finished_at"])
        second, second_code, _ = self.run_check()
        self.assertEqual(second_code, 0)
        self.assertNotEqual(first, second)
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_failure_and_launch_error_are_recorded(self):
        self.configure("raise SystemExit(7)")
        self.commit()
        _, code, manifest = self.run_check()
        self.assertEqual((code, manifest["status"]), (7, "failed"))
        self.configure(command=[str(self.base / "missing-program")])
        self.commit()
        _, code, manifest = self.run_check()
        self.assertEqual((code, manifest["status"]), (127, "launch_error"))
        self.assertIn("error", manifest)

    def test_signal_exit_code_is_shell_compatible(self):
        self.configure("import os, signal; os.kill(os.getpid(), signal.SIGTERM)")
        self.commit()
        _, code, manifest = self.run_check()
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertEqual(manifest["process_exit_code"], -signal.SIGTERM)

    def test_relative_executable_uses_project_working_directory(self):
        program = self.root / "measure"
        program.write_text("#!/bin/sh\nprintf 'measured\\n'\n")
        program.chmod(0o755)
        self.configure(command=["./measure"])
        self.commit()
        directory, code, manifest = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(manifest["command_executable"], str(program))
        self.assertEqual(manifest["command"][0], str(program))
        self.assertEqual((directory / "stdout.log").read_text(), "measured\n")

    def test_run_environment_allows_experiment_to_save_results(self):
        self.configure(
            "import os,pathlib; "
            "(pathlib.Path(os.environ['AI_INFRA_RUN_DIR']) / 'result.txt')"
            ".write_text(os.environ['AI_INFRA_RUN_ID'])"
        )
        self.commit()
        directory, code, manifest = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual((directory / "result.txt").read_text(), manifest["run_id"])

    def test_timeout_stops_descendant_processes(self):
        marker = self.base / "leaked-worker"
        child_code = (
            "import pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"time.sleep(0.7); pathlib.Path({str(marker)!r}).write_text('leaked')"
        )
        self.configure(
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); time.sleep(10)",
            timeout=0.2,
        )
        self.commit()
        _, code, manifest = self.run_check()
        self.assertEqual((code, manifest["status"]), (124, "timed_out"))
        time.sleep(0.8)
        self.assertFalse(marker.exists(), "A timed-out experiment left a worker running.")

    def test_successful_parent_cannot_leave_background_workers(self):
        marker = self.base / "background-worker"
        child_code = (
            "import pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"time.sleep(0.6); pathlib.Path({str(marker)!r}).write_text('leaked')"
        )
        self.configure(
            "import subprocess,sys; "
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}])"
        )
        self.commit()
        _, code, manifest = self.run_check()
        self.assertEqual((code, manifest["status"]), (0, "succeeded"))
        time.sleep(0.7)
        self.assertFalse(marker.exists())

    def test_interrupt_records_result_and_stops_process(self):
        self.check_external_signal(signal.SIGINT)

    def test_sigterm_records_result_and_stops_descendant_processes(self):
        self.check_external_signal(signal.SIGTERM)

    def check_external_signal(self, signum):
        marker = self.base / "signal-leaked-worker"
        child_code = (
            "import pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"time.sleep(0.7); pathlib.Path({str(marker)!r}).write_text('leaked')"
        )
        self.configure(
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
            "print('ready', flush=True); time.sleep(30)"
        )
        self.commit()
        module_root = Path(cli.__file__).resolve().parents[1]
        runner = (
            f"import sys; sys.path.insert(0, {str(module_root)!r}); "
            "from pathlib import Path; from lab_tools.cli import main; "
            f"raise SystemExit(main(['run', 'check'], root=Path({str(self.root)!r})))"
        )
        process = subprocess.Popen([sys.executable, "-c", runner], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            logs = list((self.root / "artifacts" / "runs").glob("*/stdout.log"))
            if logs and b"ready" in logs[0].read_bytes():
                break
            time.sleep(0.03)
        else:
            self.fail("Experiment did not start in time.")
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 128 + signum, stderr.decode())
        manifest = json.loads((Path(stdout.decode().strip()) / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertIsNotNone(manifest["process_exit_code"])
        self.assertEqual(manifest["termination_signal"], signum)
        time.sleep(0.8)
        self.assertFalse(marker.exists())

    def test_required_submodule_is_recorded_and_dirty_changes_rejected(self):
        module = self.add_submodule()
        self.configure(sources=["sources/example"])
        self.commit()
        _, code, manifest = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(manifest["required_sources"][0]["commit"], self.git("rev-parse", "HEAD", root=module))
        self.git("config", "submodule.sources/example.ignore", "all")
        (module / "untracked.txt").write_text("local changes")
        with self.assertRaises(cli.WorkflowError):
            self.run_check()

    def test_required_submodule_must_be_initialized_at_pinned_commit(self):
        module = self.add_submodule()
        self.configure(sources=["sources/example"])
        self.commit()
        self.git("config", "user.name", "Test Runner", root=module)
        self.git("config", "user.email", "test@example.invalid", root=module)
        (module / "source.txt").write_text("different source")
        self.git("add", "source.txt", root=module)
        self.git("commit", "-qm", "Unpinned change", root=module)
        with self.assertRaises(cli.WorkflowError):
            self.run_check()
        self.git("submodule", "deinit", "-f", "sources/example")
        with self.assertRaises(cli.WorkflowError):
            self.run_check()

    def test_config_and_output_validation(self):
        self.configure(sources=["../outside"])
        self.commit()
        with self.assertRaises(cli.WorkflowError):
            self.run_check()
        self.configure(timeout=True)
        self.commit()
        with self.assertRaises(cli.WorkflowError):
            self.run_check()
        self.configure()
        self.commit()
        with self.assertRaises(cli.WorkflowError):
            cli.run_experiment(self.root, "check", self.root / "unignored-results")
        self.assertFalse((self.root / "unignored-results").exists())

    def test_ignored_untracked_experiment_configuration_is_rejected(self):
        self.configure()
        with (self.root / ".gitignore").open("a") as ignored:
            ignored.write("experiments/\n")
        self.commit()
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaises(cli.WorkflowError):
            self.run_check()

    def test_main_repository_hidden_change_flags_are_rejected(self):
        self.configure()
        self.commit()
        name = "experiments/check/experiment.json"
        original = (self.root / name).read_text()
        for flag in ("skip-worktree", "assume-unchanged"):
            with self.subTest(flag=flag):
                self.git("update-index", f"--{flag}", name)
                config = json.loads(original)
                config["command"] = ["{python}", "-c", "print('hidden change')"]
                (self.root / name).write_text(json.dumps(config))
                self.assertEqual(self.git("status", "--porcelain"), "")
                with self.assertRaises(cli.WorkflowError):
                    self.run_check()
                (self.root / name).write_text(original)
                self.git("update-index", f"--no-{flag}", name)

    def test_required_source_hidden_change_flags_are_rejected(self):
        module = self.add_submodule()
        self.configure(sources=["sources/example"])
        self.commit()
        for flag in ("skip-worktree", "assume-unchanged"):
            with self.subTest(flag=flag):
                self.git("update-index", f"--{flag}", "source.txt", root=module)
                (module / "source.txt").write_text("hidden source change")
                self.assertEqual(self.git("status", "--porcelain", "--ignore-submodules=none"), "")
                with self.assertRaises(cli.WorkflowError):
                    self.run_check()
                (module / "source.txt").write_text("source\n")
                self.git("update-index", f"--no-{flag}", "source.txt", root=module)

    def test_nested_submodule_must_be_initialized_and_clean(self):
        module = self.add_submodule()
        self.git("config", "user.name", "Test Runner", root=module)
        self.git("config", "user.email", "test@example.invalid", root=module)
        self.git(
            "-c", "protocol.file.allow=always", "submodule", "add", str(self.base / "upstream"),
            "nested", root=module,
        )
        self.git("commit", "-qam", "Add nested source", root=module)
        self.configure(sources=["sources/example"])
        self.commit()
        _, code, manifest = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(
            {item["path"] for item in manifest["required_sources"]},
            {"sources/example", "sources/example/nested"},
        )
        self.git("config", "submodule.nested.ignore", "all", root=module)
        (module / "nested" / "untracked.txt").write_text("nested local changes")
        with self.assertRaises(cli.WorkflowError):
            self.run_check()
        self.git("submodule", "deinit", "-f", "nested", root=module)
        with self.assertRaises(cli.WorkflowError):
            self.run_check()

    def test_cli_returns_nonzero_for_invalid_request(self):
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["new", "study", "../bad"], root=self.root), 2)


if __name__ == "__main__":
    unittest.main()
