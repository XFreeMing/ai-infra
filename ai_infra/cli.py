"""Explicit helpers for studying code and recording local experiments."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


class WorkflowError(Exception):
    """An actionable workflow error, suitable for showing without a traceback."""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        check=False,
    )
    if result.returncode:
        raise WorkflowError(result.stderr.strip() or "Git command failed.")
    return result.stdout.rstrip("\n")


def _safe_path(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise WorkflowError("Path must remain inside the project.")
    return path


def _slug(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise WorkflowError("Use a slug of 1–64 lowercase letters, digits, '-' or '_'.")
    return value


def _submodules(root: Path, prefix: str = "") -> list[dict[str, Any]]:
    records = []
    for entry in _git(root, "ls-files", "--stage", "-z").split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, pinned_commit, _ = metadata.split()
        if mode != "160000":
            continue
        path = _safe_path(root, name)
        initialized = (path / ".git").exists()
        record: dict[str, Any] = {
            "path": prefix + name, "pinned_commit": pinned_commit,
            "initialized": initialized, "commit": None, "dirty": None,
        }
        if initialized:
            record["commit"] = _git(path, "rev-parse", "HEAD")
            record["dirty"] = bool(_git(
                path, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none",
            ))
        records.append(record)
        if initialized:
            records.extend(_submodules(path, prefix=prefix + name + "/"))
    return records


def _environment() -> dict[str, str]:
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "python_version": platform.python_version(), "python_executable": sys.executable,
    }


def doctor(root: Path) -> dict[str, Any]:
    report: dict[str, Any] = _environment()
    report["git"] = _git(root, "--version")
    try:
        report["project_commit"] = _git(root, "rev-parse", "--verify", "HEAD")
    except WorkflowError:
        _git(root, "rev-parse", "--git-dir")
        report["project_commit"] = None
    report["project_dirty"] = bool(_git(
        root, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none",
    ))
    report["submodules"] = _submodules(root)
    nvidia_smi = shutil.which("nvidia-smi")
    report["nvidia_smi"] = {"available": bool(nvidia_smi)}
    if nvidia_smi:
        try:
            probe = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=5, check=False,
            )
            report["nvidia_smi"].update(
                exit_code=probe.returncode, output=probe.stdout.strip(), error=probe.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            report["nvidia_smi"]["error"] = "GPU probe timed out after 5 seconds."
    return report


def new_item(root: Path, kind: str, slug: str) -> Path:
    slug = _slug(slug)
    base = ("research", "studies") if kind == "study" else ("experiments",)
    template_name = "reading.md" if kind == "study" else "experiment.md"
    template = _safe_path(root, "templates", template_name).read_text(encoding="utf-8")
    for key, value in {"title": slug, "slug": slug, "date": date.today().isoformat()}.items():
        template = template.replace("{{" + key + "}}", value)
    destination = _safe_path(root, *base, slug)
    if destination.exists():
        raise WorkflowError(f"Already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Deliberately refuse to reuse an existing directory.
    (destination / "README.md").write_text(template, encoding="utf-8")
    if kind == "experiment":
        config = {"id": slug, "command": [], "timeout_seconds": 300, "required_sources": []}
        (destination / "experiment.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return destination


def _load_experiment(root: Path, slug: str) -> dict[str, Any]:
    config_path = _safe_path(root, "experiments", _slug(slug), "experiment.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != slug:
        raise WorkflowError("experiment.json must be an object with id matching its directory slug.")
    command = config.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item and "\0" not in item for item in command
    ):
        raise WorkflowError("command must be a non-empty argv array of non-empty strings.")
    timeout = config.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise WorkflowError("timeout_seconds must be a positive finite number.")
    sources = config.get("required_sources")
    if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
        raise WorkflowError("required_sources must be an array of submodule paths.")
    if len(sources) != len(set(sources)):
        raise WorkflowError("required_sources must not contain duplicates.")
    for source in sources:
        if Path(source).is_absolute() or ".." in Path(source).parts or source != Path(source).as_posix():
            raise WorkflowError("required_sources must use canonical relative paths inside the project.")
        _safe_path(root, source)
    return config


def _snapshot(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "--verify", "HEAD")
    _reject_hidden_changes(root)
    if _git(root, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"):
        raise WorkflowError("Commit or remove local changes before running an experiment.")
    config_name = f"experiments/{config['id']}/experiment.json"
    tracked = _git(root, "ls-files", "--stage", "--error-unmatch", "--", config_name)
    if not tracked.startswith(("100644 ", "100755 ")):
        raise WorkflowError("experiment.json must be a tracked regular file in the main repository.")
    modules = {item["path"]: item for item in _submodules(root)}
    required = {}
    for source in config["required_sources"]:
        if source not in modules:
            raise WorkflowError(f"Required source is not a tracked submodule: {source}")
        for name, module in modules.items():
            if name != source and not name.startswith(source + "/"):
                continue
            if not module["initialized"]:
                raise WorkflowError(f"Initialize the required submodule at its pinned commit: {name}")
            _reject_hidden_changes(_safe_path(root, name))
            if module["commit"] != module["pinned_commit"] or module["dirty"]:
                raise WorkflowError(f"Required submodule must be clean and match its pinned commit: {name}")
            required[name] = module
    return {"project_commit": commit, "required_sources": list(required.values())}


def _reject_hidden_changes(root: Path) -> None:
    for entry in _git(root, "ls-files", "-v", "-z").split("\0"):
        if entry and (entry[0] == "S" or entry[0].islower()):
            raise WorkflowError(
                f"Clear skip-worktree/assume-unchanged flags before running: {root / entry[2:]}"
            )


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Descendants may have ignored SIGTERM even if the leader already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    pending = directory / "manifest.json.tmp"
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(directory / "manifest.json")


def _resolve_command(root: Path, original: list[str]) -> tuple[list[str], str | None]:
    command = [sys.executable if item == "{python}" else item for item in original]
    if "/" in command[0]:
        executable = os.path.abspath(root / command[0])
    else:
        # Relative PATH entries, like relative argv paths, are resolved from the child's cwd.
        search_path = os.pathsep.join(os.path.abspath(root / entry) for entry in os.get_exec_path())
        executable = shutil.which(command[0], path=search_path)
    if executable:
        command[0] = executable
    return command, executable


def run_experiment(root: Path, slug: str, output_dir: Path) -> tuple[Path, int]:
    if os.name != "posix":
        raise WorkflowError("Experiment execution currently supports macOS and Linux.")
    config = _load_experiment(root, slug)
    snapshot = _snapshot(root, config)
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(root):
        try:
            _git(root, "check-ignore", str(output_dir / ".run-output-probe"))
        except WorkflowError as error:
            raise WorkflowError("Output inside the project must be Git-ignored; use artifacts/runs.") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    prefix = started_at.strftime("%Y%m%dT%H%M%SZ-") + slug + "-"
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=output_dir))
    command, executable = _resolve_command(root, config["command"])
    run_environment = {"AI_INFRA_RUN_DIR": str(directory), "AI_INFRA_RUN_ID": directory.name}
    manifest = {
        "schema_version": 1, "run_id": directory.name, "experiment_id": slug,
        "started_at": started_at.isoformat(), "finished_at": None,
        "status": "running", "exit_code": None, "process_exit_code": None,
        "project_root": str(root), "environment": _environment(), "config": config,
        "command": command, "command_executable": executable, "run_environment": run_environment,
        "termination_signal": None,
        "logs": {"stdout": "stdout.log", "stderr": "stderr.log"}, **snapshot,
    }
    _write_manifest(directory, manifest)
    started = time.monotonic()
    exit_code = 1
    requested_signal: int | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal requested_signal
        requested_signal = signum

    previous_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in previous_handlers:
        signal.signal(signum, request_stop)
    try:
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(
                command, cwd=root, stdout=stdout, stderr=stderr,
                shell=False, start_new_session=True, env={**os.environ, **run_environment},
            )
            try:
                while True:
                    if requested_signal is not None:
                        exit_code = 128 + requested_signal
                        manifest["status"] = "interrupted"
                        manifest["termination_signal"] = requested_signal
                        break
                    remaining = config["timeout_seconds"] - (time.monotonic() - started)
                    if remaining <= 0:
                        exit_code = 124
                        manifest["status"] = "timed_out"
                        break
                    try:
                        return_code = process.wait(timeout=min(remaining, 0.2))
                    except subprocess.TimeoutExpired:
                        continue
                    exit_code = return_code if return_code >= 0 else 128 - return_code
                    manifest["status"] = "succeeded" if return_code == 0 else "failed"
                    break
            finally:
                # A foreground command can exit while leaving workers behind.
                _stop_process_group(process)
            manifest["process_exit_code"] = process.returncode
    except OSError as error:
        manifest["status"] = "launch_error"
        manifest["error"] = str(error)
        exit_code = 127
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        manifest["exit_code"] = exit_code
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["duration_seconds"] = round(time.monotonic() - started, 6)
        _write_manifest(directory, manifest)
    return directory, exit_code


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    parser = argparse.ArgumentParser(description="Local AI Infra research and experiment helpers.")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("doctor", help="Report local tools, platform and pinned source versions.")
    new_parser = commands.add_parser("new", help="Create a study or experiment from a template.")
    new_parser.add_argument("kind", choices=("study", "experiment"))
    new_parser.add_argument("slug")
    run_parser = commands.add_parser("run", help="Record a local experiment from a clean Git commit.")
    run_parser.add_argument("slug")
    run_parser.add_argument("--output-dir", type=Path, default=root / "artifacts" / "runs")
    args = parser.parse_args(argv)
    try:
        if args.action == "doctor":
            print(json.dumps(doctor(root), ensure_ascii=False, indent=2))
        elif args.action == "new":
            print(new_item(root, args.kind, args.slug))
        else:
            directory, exit_code = run_experiment(root, args.slug, args.output_dir)
            print(directory)
            return exit_code
        return 0
    except (WorkflowError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
