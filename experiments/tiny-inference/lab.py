"""Run the pinned llama2.c teaching model on CPU, with local evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import FrameType
from typing import Any, NoReturn
import uuid


ROOT = Path(__file__).resolve().parents[2]
PREPARE = "请先运行：uv run --offline python experiments/tiny-inference/lab.py prepare"


class LabError(Exception):
    """An actionable preparation or provenance error."""


class LabTerminated(Exception):
    """SIGTERM requested a recorded, clean shutdown."""


def terminate_run(signum: int, frame: FrameType | None) -> NoReturn:
    raise LabTerminated("收到 SIGTERM，模型执行已终止")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
        timeout=10,
    ).stdout.rstrip("\n")


def verify_file(path: Path, expected_hash: str, expected_size: int | None = None) -> str:
    if not path.is_file():
        raise LabError(f"缺少文件 {path}。{PREPARE}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise LabError(f"文件大小不符，保留原文件供检查：{path}")
    actual = sha256(path)
    if actual != expected_hash:
        raise LabError(f"SHA256 不符，保留原文件供检查：{path}")
    return actual


def verify_source(root: Path, lock: dict[str, Any]) -> dict[str, str]:
    source = root / lock["source"]["path"]
    if not (source / "run.c").is_file():
        raise LabError("请先初始化 sources/llama2.c 子模块。")
    actual = git(source, "rev-parse", "HEAD")
    if actual != lock["source"]["commit"]:
        raise LabError(f"llama2.c 版本不符：{actual}，需要 {lock['source']['commit']}")
    if git(source, "status", "--porcelain", "--untracked-files=normal"):
        raise LabError("sources/llama2.c 有本地修改；请先保留修改并恢复固定版本。")
    pinned_source = subprocess.run(
        ["git", "-C", str(source), "show", f"{actual}:run.c"],
        check=True, capture_output=True, timeout=10,
    ).stdout
    run_c_hash = verify_file(source / "run.c", hashlib.sha256(pinned_source).hexdigest())
    tokenizer_hash = verify_file(root / lock["tokenizer"]["path"], lock["tokenizer"]["sha256"])
    return {
        "commit": actual, "path": str(source), "run_c_sha256": run_c_hash,
        "tokenizer_sha256": tokenizer_hash,
    }


def model_path(root: Path, lock: dict[str, Any]) -> Path:
    return root / "models" / "tiny-inference" / lock["model"]["filename"]


def ensure_model(root: Path, lock: dict[str, Any]) -> Path:
    model = lock["model"]
    path = model_path(root, lock)
    if path.exists():
        verify_file(path, model["sha256"], model["bytes"])
        return path
    curl = shutil.which("curl")
    if curl is None:
        raise LabError("下载需要 curl；当前环境找不到它。")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        subprocess.run(
            [curl, "--fail", "--location", "--connect-timeout", "15", "--max-time", "180",
             "--output", str(temporary), model["url"]], check=True, timeout=190,
        )
        try:
            verify_file(temporary, model["sha256"], model["bytes"])
        except LabError as error:
            raise LabError("下载文件未通过大小/SHA256 校验；临时文件已删除，请重试 prepare。") from error
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare(root: Path, lock: dict[str, Any]) -> None:
    source = verify_source(root, lock)
    ensure_model(root, lock)
    compiler = shutil.which("clang")
    if compiler is None:
        raise LabError("需要 clang；Mac 可安装 Xcode Command Line Tools。")
    build_dir = root / "artifacts" / "tiny-inference"
    build_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "model": lock["model"], "compiler": compiler,
        "compiler_version": subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip(),
        "environment": {"platform": platform.platform(), "machine": platform.machine()},
        "binaries": {},
    }
    for name, flags in (("fast", ["-O3"]), ("debug", ["-O0", "-g"])):
        binary = build_dir / f"run-{name}"
        command = [compiler, *flags, str(root / lock["source"]["path"] / "run.c"),
                   "-lm", "-o", str(binary)]
        subprocess.run(command, check=True, timeout=60)
        manifest["binaries"][name] = {
            "path": str(binary), "sha256": sha256(binary), "command": command,
        }
    write_json(build_dir / "build.json", manifest)
    print(f"准备完成：{build_dir / 'build.json'}")


def steps_value(value: str) -> int:
    try:
        steps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("steps 必须是 1–256 的整数") from error
    if not 1 <= steps <= 256:
        raise argparse.ArgumentTypeError("steps 必须是 1–256 的整数（包含输入处理）")
    return steps


def run(root: Path, lock: dict[str, Any], args: argparse.Namespace) -> int:
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    inherited = os.environ.get("AI_INFRA_RUN_DIR")
    directory = Path(inherited) if inherited else root / "artifacts" / "tiny-inference" / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=bool(inherited))
    result = {
        "experiment_id": "tiny-inference", "run_id": os.environ.get("AI_INFRA_RUN_ID", run_id),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "CPU learning run; not a GB10 benchmark",
        "configuration": {"prompt": args.prompt, "steps": args.steps, "temperature": 0,
                          "seed": 1, "debug_build": args.debug_build, "timeout_seconds": 30},
        "model_lock": lock,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": sys.version, "python_executable": sys.executable},
        "stdout_file": "model.stdout.log", "stderr_file": "model.stderr.log",
    }
    started = time.perf_counter()
    code = 1
    with (directory / "model.stdout.log").open("wb") as stdout, (directory / "model.stderr.log").open("wb") as stderr:
        previous_sigterm = signal.signal(signal.SIGTERM, terminate_run)
        try:
            result["parent_commit"] = git(root, "rev-parse", "HEAD")
            result["parent_status"] = git(root, "status", "--porcelain", "--untracked-files=normal").splitlines()
            result["source"] = verify_source(root, lock)
            verify_file(model_path(root, lock), lock["model"]["sha256"], lock["model"]["bytes"])
            manifest_path = root / "artifacts" / "tiny-inference" / "build.json"
            if not manifest_path.is_file():
                raise LabError(PREPARE)
            build = read_json(manifest_path)
            if build["source"] != result["source"] or build["model"] != lock["model"]:
                raise LabError(f"构建证据与当前固定输入不符。{PREPARE}")
            binary = build["binaries"]["debug" if args.debug_build else "fast"]
            verify_file(Path(binary["path"]), binary["sha256"])
            result["build"] = build
            command = [binary["path"], str(model_path(root, lock)), "-z", str(root / lock["tokenizer"]["path"]),
                       "-t", "0", "-s", "1", "-n", str(args.steps), "-i", args.prompt]
            result["command"] = command
            result["status"] = "running"
            write_json(directory / "result.json", result)
            inference_started = time.perf_counter()
            try:
                completed = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, timeout=30)
            finally:
                result["inference_process_seconds"] = time.perf_counter() - inference_started
            result["process_exit_code"] = completed.returncode
            code = completed.returncode if completed.returncode >= 0 else 128 - completed.returncode
            result["status"] = "succeeded" if code == 0 else "failed"
        except subprocess.TimeoutExpired as error:
            code = 124
            result.update(status="timed_out", error=str(error))
        except KeyboardInterrupt:
            code = 130
            result.update(status="interrupted", error="用户中断")
        except LabTerminated as error:
            code = 143
            result.update(status="terminated", error=str(error))
        except (LabError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            result.update(status="failed", error=str(error))
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            result.update(exit_code=code, wall_seconds=time.perf_counter() - started,
                          finished_at=datetime.now(timezone.utc).isoformat())
            write_json(directory / "result.json", result)
    for filename, stream in (("model.stdout.log", sys.stdout), ("model.stderr.log", sys.stderr)):
        print((directory / filename).read_text(encoding="utf-8", errors="replace"), end="", file=stream)
    if "error" in result:
        print(result["error"], file=sys.stderr)
    print(f"\n执行记录：{directory / 'result.json'}")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="固定版本的 tinyllamas / llama2.c CPU 学习实验")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="校验/下载 61 MB 模型，编译快速版和调试版")
    run_parser = commands.add_parser("run", help="运行最多 30 秒，并保存成功或失败证据")
    run_parser.add_argument("--steps", type=steps_value, default=32)
    run_parser.add_argument("--prompt", default="Once upon a time")
    run_parser.add_argument("--debug-build", action="store_true")
    args = parser.parse_args(argv)
    try:
        lock = read_json(ROOT / "experiments" / "tiny-inference" / "model.lock.json")
        if args.command == "prepare":
            prepare(ROOT, lock)
            return 0
        return run(ROOT, lock, args)
    except (LabError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"无法完成：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
