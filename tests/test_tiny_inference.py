"""Tiny CPU lab checks; no network or real model is required."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "tiny_lab", Path(__file__).resolve().parents[1] / "experiments" / "tiny-inference" / "lab.py",
)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class TinyInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tiny-inference-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.lock = {
            "source": {"path": "sources/llama2.c", "commit": "expected-commit"},
            "model": {"filename": "small.bin", "bytes": 4,
                      "sha256": hashlib.sha256(b"good").hexdigest(),
                      "url": "https://example.invalid/small.bin"},
            "tokenizer": {"path": "sources/llama2.c/tokenizer.bin", "sha256": "unused"},
        }

    def test_corrupt_cached_model_is_preserved_without_download(self):
        path = lab.model_path(self.root, self.lock)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"evil")
        with patch.object(lab.subprocess, "run") as process:
            with self.assertRaisesRegex(lab.LabError, "SHA256"):
                lab.ensure_model(self.root, self.lock)
        process.assert_not_called()
        self.assertEqual(path.read_bytes(), b"evil")

    def test_source_pin_mismatch_is_rejected(self):
        source = self.root / "sources" / "llama2.c"
        source.mkdir(parents=True)
        (source / "run.c").write_text("source")
        with patch.object(lab, "git", return_value="different-commit"):
            with self.assertRaisesRegex(lab.LabError, "版本不符"):
                lab.verify_source(self.root, self.lock)

    def test_step_bound_prevents_upstream_zero_meaning_unlimited(self):
        for value in ("0", "-1", "257", "1.5", "abc"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                lab.steps_value(value)
        self.assertEqual(lab.steps_value("1"), 1)
        self.assertEqual(lab.steps_value("256"), 256)

    def test_source_hidden_by_assume_unchanged_is_rejected(self):
        source = self.root / "sources" / "llama2.c"
        source.mkdir(parents=True)
        run_c = source / "run.c"
        run_c.write_text("original source\n")
        for command in (
            ["init", "-q"], ["add", "run.c"],
            ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "pin"],
        ):
            subprocess.run(["git", "-C", str(source), *command], check=True, capture_output=True)
        self.lock["source"]["commit"] = lab.git(source, "rev-parse", "HEAD")
        subprocess.run(
            ["git", "-C", str(source), "update-index", "--assume-unchanged", "run.c"],
            check=True, capture_output=True,
        )
        run_c.write_text("hidden local edit\n")
        self.assertEqual(lab.git(source, "status", "--porcelain"), "")
        with self.assertRaisesRegex(lab.LabError, "SHA256"):
            lab.verify_source(self.root, self.lock)

    def prepare_fake_binary(self, code):
        path = lab.model_path(self.root, self.lock)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"good")
        build_dir = self.root / "artifacts" / "tiny-inference"
        build_dir.mkdir(parents=True)
        binary = build_dir / "run-fast"
        binary.write_text(f"#!{sys.executable}\n{code}\n", encoding="utf-8")
        binary.chmod(0o755)
        self.source = {"commit": "expected-commit", "run_c_sha256": "test-source"}
        lab.write_json(build_dir / "build.json", {
            "source": self.source, "model": self.lock["model"],
            "binaries": {"fast": {"path": str(binary), "sha256": lab.sha256(binary)}},
        })
        return binary

    def run_fake(self):
        args = argparse.Namespace(prompt="Once upon a time", steps=32, debug_build=False)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(lab, "git", side_effect=["parent-commit", " M sources/other"]),
            patch.object(lab, "verify_source", return_value=self.source),
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
        ):
            code = lab.run(self.root, self.lock, args)
        directory = next((self.root / "artifacts" / "tiny-inference" / "runs").iterdir())
        return code, directory, json.loads((directory / "result.json").read_text())

    def test_failed_process_preserves_output_and_provenance(self):
        self.prepare_fake_binary("import sys\nprint('partial result')\nprint('failed', file=sys.stderr)\nraise SystemExit(7)")
        code, directory, result = self.run_fake()
        self.assertEqual(code, 7)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["process_exit_code"], 7)
        self.assertGreater(result["inference_process_seconds"], 0)
        self.assertLessEqual(result["inference_process_seconds"], result["wall_seconds"])
        self.assertEqual(result["parent_commit"], "parent-commit")
        self.assertEqual(result["parent_status"], [" M sources/other"])
        self.assertIn("sha256", result["build"]["binaries"]["fast"])
        self.assertEqual(result["command"][-2:], ["-i", "Once upon a time"])
        self.assertEqual((directory / "model.stdout.log").read_text(), "partial result\n")
        self.assertEqual((directory / "model.stderr.log").read_text(), "failed\n")

    def test_changed_binary_is_not_executed_and_failure_is_recorded(self):
        binary = self.prepare_fake_binary("print('original')")
        binary.write_text("changed binary", encoding="utf-8")
        code, directory, result = self.run_fake()
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("SHA256", result["error"])
        self.assertNotIn("command", result)
        self.assertNotIn("inference_process_seconds", result)
        self.assertEqual((directory / "model.stdout.log").read_bytes(), b"")

    def test_timeout_preserves_partial_output(self):
        self.prepare_fake_binary("print('not executed')")

        def timeout(command, **kwargs):
            kwargs["stdout"].write(b"partial before timeout\n")
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        with patch.object(lab.subprocess, "run", side_effect=timeout):
            code, directory, result = self.run_fake()
        self.assertEqual(code, 124)
        self.assertEqual(result["status"], "timed_out")
        self.assertGreater(result["inference_process_seconds"], 0)
        self.assertEqual((directory / "model.stdout.log").read_bytes(), b"partial before timeout\n")

    def test_sigterm_has_prestart_evidence_and_restores_handler(self):
        self.prepare_fake_binary("print('not executed')")
        previous_handler = signal.getsignal(signal.SIGTERM)

        def terminate(command, **kwargs):
            directory = next((self.root / "artifacts" / "tiny-inference" / "runs").iterdir())
            snapshot = json.loads((directory / "result.json").read_text())
            self.assertEqual(snapshot["status"], "running")
            self.assertEqual(snapshot["command"], command)
            for key in ("build", "model_lock", "source", "parent_commit"):
                self.assertIn(key, snapshot)
            kwargs["stdout"].write(b"partial before termination\n")
            handler = signal.getsignal(signal.SIGTERM)
            self.assertIs(handler, lab.terminate_run)
            handler(signal.SIGTERM, None)

        with patch.object(lab.subprocess, "run", side_effect=terminate):
            code, directory, result = self.run_fake()
        self.assertEqual(code, 143)
        self.assertEqual(result["status"], "terminated")
        self.assertIn("inference_process_seconds", result)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous_handler)
        self.assertEqual((directory / "model.stdout.log").read_bytes(), b"partial before termination\n")


if __name__ == "__main__":
    unittest.main()
