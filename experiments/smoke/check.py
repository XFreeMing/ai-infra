"""Verify the recording workflow without downloading models or requiring a GPU."""

import hashlib
import json
import os
from pathlib import Path
import platform


payload = b"lab_tools recording smoke check\n"
result = {
    "check": "recording-smoke",
    "python_version": platform.python_version(),
    "payload_sha256": hashlib.sha256(payload).hexdigest(),
    "success": True,
}
output = json.dumps(result, indent=2) + "\n"
(Path(os.environ["AI_INFRA_RUN_DIR"]) / "result.json").write_text(output, encoding="utf-8")
print(output, end="")
