"""Resolve prepared binaries and launch a foreground server without a shell."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

from lab_tools.cli import WorkflowError, _git
from lab_tools.prepare import fingerprint, sha256
from lab_tools.source_tracking import source_revision


def resolve_server(root: Path, engine: str, binary: Path | None,
                   preparation_dir: Path | None) -> tuple[Path, dict | None, Path | None]:
    if binary is not None:
        return (root / binary).resolve(), None, None
    reference = json.loads((root / "research/studies/ds4-inference/R0_MODEL.json").read_text())
    current = {name: source_revision(root, name, _git)[0] for name in ('ds4', 'llama.cpp')}
    candidates = [(root / preparation_dir).resolve()] if preparation_dir else sorted(
        (root / "artifacts/init").glob("*"), reverse=True)
    key, name = ("ds4-new", "ds4-server") if engine == "ds4" else ("llama", "llama-server")
    for directory in candidates:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('machine_id', '700') != '700' or manifest.get('backend', 'cuda') != 'cuda':
            continue
        if manifest.get("status") != "prepared":
            continue
        builds = manifest.get("builds", {})
        expected = reference['engines'] if preparation_dir and manifest.get('source_mode') != 'current-checkout' else current
        if builds.get("ds4-new", {}).get("commit") != expected["ds4"] or \
           builds.get("llama", {}).get("commit") != expected["llama.cpp"]:
            continue
        build = builds[key]
        record = next((item for item in build["binaries"] if Path(item["path"]).name == name), None)
        if record is None:
            continue
        path = Path(record["path"]).resolve()
        if sha256(path) != record["sha256"]:
            raise WorkflowError(f"已准备的 {name} 二进制发生变化：{path}")
        return path, manifest, manifest_path
    raise WorkflowError("找不到匹配固定版本的 prepared 构建；先运行 uv run init，或设置 SERVER_BINARY。")


def launch(engine: str, model_path: Path, arguments: list[str], *, cuda_devices: str,
           binary: Path | None, preparation_dir: Path | None, configuration_file: Path,
           argv: list[str] | None = None,
           root: Path | None = None) -> int:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    parser = argparse.ArgumentParser(prog=engine, description="使用脚本顶部配置启动 CUDA HTTP 服务。")
    parser.add_argument("--dry-run", action="store_true", help="检查路径并显示 argv，不启动服务或加载模型。")
    parser.add_argument("--server-help", action="store_true", help="检查配置参数并显示上游服务帮助，不加载模型。")
    parser.add_argument("server_arguments", nargs=argparse.REMAINDER, help="在 -- 后追加上游参数。")
    args = parser.parse_args(argv)
    try:
        executable, prepared, manifest_path = resolve_server(root, engine, binary, preparation_dir)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise WorkflowError(f"服务二进制不存在或不可执行：{executable}")
        extra = args.server_arguments
        if extra[:1] == ["--"]:
            extra = extra[1:]
        command = [str(executable), *arguments, *extra]
        paths = {"model": (root / model_path).resolve()}
        file_flags = {"-m": "model", "--model": "model", "--vision": "vision",
                      "--mmproj": "mmproj", "--mtp-model": "mtp_model"}
        # Later argv overrides must also update the recorded model identity.
        for index, item in enumerate(command):
            flag, separator, value = item.partition("=")
            if separator and flag in file_flags:
                path = (root / value).resolve()
                paths[file_flags[flag]] = path
                command[index] = flag + "=" + str(path)
            elif item in file_flags:
                if index + 1 == len(command):
                    raise WorkflowError(f"{item} 缺少模型路径。")
                path = (root / command[index + 1]).resolve()
                paths[file_flags[item]] = path
                command[index + 1] = str(path)
        if not args.server_help:
            for path in paths.values():
                if not path.is_file() or path.suffix != ".gguf":
                    raise WorkflowError(f"模型路径需要完整的 .gguf 文件：{path}")
        if args.server_help:
            command += ["--help"]
        overrides = {"CUDA_VISIBLE_DEVICES": cuda_devices}
        display = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_devices)} {shlex.join(command)}"
        if args.dry_run or args.server_help:
            print(display, flush=True)
        if args.dry_run:
            return 0
        if not args.server_help:
            # This is a launch receipt, not a benchmark or a readiness assertion.
            base = root / "artifacts/servers"
            base.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + engine + "-", dir=base))
            effective_model = paths["model"]
            model = {"path": str(effective_model), "stat": fingerprint(effective_model.stat()),
                     "sha256_verified_by_preparation": None}
            if prepared and prepared.get("model", {}).get("path") == model["path"] and \
               prepared["model"].get("stat") == model["stat"] and prepared["model"].get("verified"):
                model["sha256_verified_by_preparation"] = prepared["model"]["sha256"]
            receipt = {"kind": "service_launch_request", "machine_id": "700", "backend": "cuda", "launch_id": output.name,
                       "requested_at": datetime.now(timezone.utc).isoformat(), "engine": engine,
                       "parent_commit": _git(root, "rev-parse", "HEAD"),
                       "parent_dirty": bool(_git(root, "status", "--porcelain")),
                       "argv": command, "environment_overrides": overrides,
                       "preparation_manifest": str(manifest_path) if manifest_path else None,
                       "binary_sha256": sha256(executable), "model": model,
                       "auxiliary_models": {name: {"path": str(path), "stat": fingerprint(path.stat())}
                                            for name, path in paths.items() if name != "model"},
                       "configuration_file": str(configuration_file),
                       "configuration_sha256": sha256(configuration_file),
                       "model_inference_validated": False}
            (output / "launch.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            print(display, flush=True)
            print(f"启动记录：{output / 'launch.json'}", flush=True)
        os.chdir(root)
        os.execve(executable, command, {**os.environ, **overrides})
    except (WorkflowError, OSError, ValueError, KeyError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        return 2
