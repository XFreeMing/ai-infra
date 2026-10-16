"""Binary layout and lossless SQLite export checks using tiny fixtures."""

from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "tiny-inference" / "export_embeddings.py"
SPEC = importlib.util.spec_from_file_location("embedding_export", SCRIPT)
export = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(export)


class EmbeddingExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model.bin"
        self.tokenizer = self.root / "tokenizer.bin"
        self.output = self.root / "tokens.sqlite3"
        self.words = ["<unk>", "\n<s>\n", "\n</s>\n", "<0x0A>", " Once", "中"]
        # Negative vocab_size encodes an unshared classifier; embedding offset is unchanged.
        self.values = [(index + 0.125, -index - 0.25, index + 0.5) for index in range(len(self.words))]
        self.model.write_bytes(struct.pack("<7i", 3, 6, 1, 1, 1, -len(self.words), 8)
                              + b"".join(struct.pack("<3f", *row) for row in self.values))
        self.tokenizer.write_bytes(struct.pack("<I", 16) + b"".join(
            struct.pack("<fi", -float(index), len(word.encode())) + word.encode()
            for index, word in enumerate(self.words)
        ))

    def test_text_and_float_values_round_trip_with_dimension_view(self):
        config = export.write_database(self.model, self.tokenizer, self.output, {"test": True})
        self.assertEqual(config["vocab_size"], 6)
        with closing(sqlite3.connect(self.output)) as database:
            rows = database.execute("SELECT * FROM tokens ORDER BY token_id").fetchall()
            self.assertEqual(len(rows), len(self.words))
            for index, row in enumerate(rows):
                self.assertEqual(row[0], index)
                self.assertEqual(row[1], self.words[index])
                self.assertEqual(json.loads(row[2]), self.words[index])
                self.assertEqual(bytes.fromhex(row[4]), self.words[index].encode())
                self.assertEqual(row[6:], self.values[index])
            self.assertEqual([row[3] for row in rows], ["unknown", "BOS", "EOS", "byte", "text", "text"])
            self.assertEqual(database.execute(
                "SELECT dim_index, value FROM embedding_values WHERE token_id = 4 ORDER BY dim_index"
            ).fetchall(), list(enumerate(self.values[4])))
            self.assertEqual(database.execute("PRAGMA integrity_check").fetchone(), ("ok",))

    def test_existing_output_is_not_overwritten(self):
        self.output.write_bytes(b"keep this file")
        with self.assertRaises(FileExistsError):
            export.write_database(self.model, self.tokenizer, self.output, {})
        self.assertEqual(self.output.read_bytes(), b"keep this file")

    def test_truncated_embedding_is_rejected_without_output(self):
        self.model.write_bytes(self.model.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "embedding 数据截断"):
            export.write_database(self.model, self.tokenizer, self.output, {})
        self.assertFalse(self.output.exists())

    def test_tokenizer_mismatch_is_rejected_without_output(self):
        original = self.tokenizer.read_bytes()
        for changed in (original[:-1], original + b"extra"):
            with self.subTest(length=len(changed)):
                self.tokenizer.write_bytes(changed)
                with self.assertRaises(ValueError):
                    export.write_database(self.model, self.tokenizer, self.output, {})
                self.assertFalse(self.output.exists())

    def test_non_finite_weights_are_rejected_without_output(self):
        data = bytearray(self.model.read_bytes())
        struct.pack_into("<f", data, 28, float("nan"))
        self.model.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "非有限"):
            export.write_database(self.model, self.tokenizer, self.output, {})
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
