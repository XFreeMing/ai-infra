"""Export the pinned llama2.c vocabulary and initial embeddings to SQLite."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import sqlite3
import struct
import sys
import tempfile
import time
from typing import BinaryIO
import uuid

from lab import ROOT, LabError, git, model_path, read_json, sha256, verify_file, verify_source


HEADER = struct.Struct("<7i")
CONFIG_FIELDS = (
    "dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len",
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "tiny-inference" / "token-embeddings.sqlite3"


def read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"{label} 数据截断：需要 {size} 字节，读到 {len(data)} 字节")
    return data


def read_config(stream: BinaryIO) -> dict[str, int]:
    config = dict(zip(CONFIG_FIELDS, HEADER.unpack(read_exact(stream, HEADER.size, "模型头"))))
    config["vocab_size"] = abs(config["vocab_size"])
    if any(value <= 0 for value in config.values()):
        raise ValueError("模型配置必须为正整数")
    if config["dim"] % config["n_heads"] or config["n_heads"] % config["n_kv_heads"]:
        raise ValueError("模型 head 配置不整除")
    return config


def token_kind(token_id: int, text: str) -> str:
    if token_id < 3:
        return ("unknown", "BOS", "EOS")[token_id]
    return "byte" if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", text) else "text"


def write_database(model: Path, tokenizer: Path, output: Path, metadata: dict) -> dict[str, int]:
    """Read legacy little-endian FP32 files; publish only a complete new database."""
    if output.exists():
        raise FileExistsError(f"数据库已存在，保留原文件；请用 --output 指定新路径：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        model.open("rb") as weights,
        tokenizer.open("rb") as vocabulary,
        tempfile.TemporaryDirectory(prefix=".embedding-export-", dir=output.parent) as directory,
    ):
        config = read_config(weights)
        dim, vocab_size = config["dim"], config["vocab_size"]
        if model.stat().st_size < HEADER.size + vocab_size * dim * 4:
            raise ValueError("模型 embedding 数据截断")
        max_length, = struct.unpack("<I", read_exact(vocabulary, 4, "词表头"))
        if not 0 < max_length <= tokenizer.stat().st_size:
            raise ValueError("词表 max_token_length 无效")
        temporary = Path(directory) / "export.sqlite3"
        with closing(sqlite3.connect(temporary)) as database, database:
            if dim > database.getlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT):
                raise ValueError("维度超出本教学导出器的纵向视图限制")
            columns = [f"embedding_{index:03d}" for index in range(dim)]
            database.execute(
                "CREATE TABLE tokens (token_id INTEGER PRIMARY KEY, token_text TEXT NOT NULL, "
                "token_text_escaped TEXT NOT NULL, token_kind TEXT NOT NULL, "
                "token_bytes_hex TEXT NOT NULL, tokenizer_score REAL NOT NULL, "
                + ", ".join(f"{name} REAL NOT NULL" for name in columns) + ")"
            )
            database.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            insert = "INSERT INTO tokens VALUES (" + ",".join("?" for _ in range(6 + dim)) + ")"
            embedding_row = struct.Struct(f"<{dim}f")
            for token_id in range(vocab_size):
                score, length = struct.unpack("<fi", read_exact(vocabulary, 8, f"token {token_id} 头"))
                if not 0 <= length <= max_length:
                    raise ValueError(f"token {token_id} 文本长度无效：{length}")
                raw = read_exact(vocabulary, length, f"token {token_id} 文本")
                text = raw.decode("utf-8")
                values = embedding_row.unpack(read_exact(weights, embedding_row.size, f"token {token_id} embedding"))
                if not math.isfinite(score) or not all(map(math.isfinite, values)):
                    raise ValueError(f"token {token_id} 含非有限浮点值")
                database.execute(insert, (
                    token_id, text, json.dumps(text, ensure_ascii=False), token_kind(token_id, text),
                    raw.hex(), score, *values,
                ))
            if vocabulary.read(1):
                raise ValueError("词表有多余数据，可能与模型 vocab_size 不匹配")
            # UNION ALL makes the individual dimensions readable without duplicating weights.
            database.execute("CREATE VIEW embedding_values AS " + " UNION ALL ".join(
                f"SELECT token_id, token_text_escaped, {index} AS dim_index, {name} AS value FROM tokens"
                for index, name in enumerate(columns)
            ))
            database.execute(
                "CREATE VIEW once_embedding AS SELECT * FROM embedding_values "
                "WHERE token_id = 9038 ORDER BY dim_index"
            )
            entries = {
                **metadata,
                "config": config,
                "representation": "初始 token_embedding_table；不是 RMSNorm 输出、Q/K/V 或上下文隐藏状态",
                "float_format": "小端 FP32 解码后精确存为 SQLite REAL（binary64），未人为四舍五入",
                "token_text_semantics": "token_text 保留词表原文；escaped 列用 JSON 引号显示空格和转义符；byte token 保留 <0xXX>，不伪装成完整字符",
                "source_symbols": ["Config", "memory_map_weights", "build_tokenizer", "decode", "forward"],
                "embedding_offset_bytes": HEADER.size,
            }
            database.executemany("INSERT INTO metadata VALUES (?, ?)", (
                (key, json.dumps(value, ensure_ascii=False)) for key, value in entries.items()
            ))
            if database.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("SQLite 完整性检查失败")
        # Same-filesystem hard link refuses to overwrite even if another export raced us.
        os.link(temporary, output)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出固定 TinyStories 模型的全部 token 和初始 embedding 到 SQLite")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="新数据库路径；不会覆盖已有文件")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        lock = read_json(ROOT / "experiments" / "tiny-inference" / "model.lock.json")
        source = verify_source(ROOT, lock)
        model = model_path(ROOT, lock)
        verify_file(model, lock["model"]["sha256"], lock["model"]["bytes"])
        output = args.output.expanduser().resolve()
        metadata = {
            "experiment_id": "tiny-inference-embedding-export",
            "run_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_commit": git(ROOT, "rev-parse", "HEAD"),
            "parent_status": git(ROOT, "status", "--porcelain", "--untracked-files=normal").splitlines(),
            "source": source,
            "source_url": f"{lock['source']['url']}/blob/{source['commit']}/run.c",
            "model_lock": lock,
            "export_script_sha256": sha256(Path(__file__)),
            "environment": {"platform": platform.platform(), "python": sys.version,
                            "python_executable": sys.executable, "sqlite": sqlite3.sqlite_version},
            "command": [sys.executable, str(Path(__file__).resolve()), "--output", str(output)],
            "mode": "CPU 文件解析；未执行推理，不是 GB10 测试",
        }
        config = write_database(model, ROOT / lock["tokenizer"]["path"], output, metadata)
        print(f"已导出：{output}")
        print(f"{config['vocab_size']} 个 token，每个 {config['dim']} 维；{output.stat().st_size / 1024**2:.1f} MiB；{time.perf_counter() - started:.2f} 秒")
        print("打开 tokens 表看全部词表；打开 once_embedding 视图纵向看 9038 的 288 个数。")
        return 0
    except (LabError, OSError, ValueError, sqlite3.Error) as error:
        print(f"导出失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
