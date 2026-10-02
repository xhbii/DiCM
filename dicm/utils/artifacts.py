"""Atomic experiment records and configuration checks when resuming."""
import json
import os
import tempfile
from pathlib import Path

def write_json(path, data):
    path = Path(path)
    # Normalize tuples and integer keys before comparing with an on-disk record.
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if (path.name == "protocol.json" or path.name.startswith("base_fingerprint_")) and path.exists():
        if json.loads(path.read_text()) != json.loads(text):
            raise ValueError(f"Protocol changed in {path.parent}; use a new output directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".record-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def verify_base(directory, digest, dtype):
    """Reject stale training caches if the loaded base weights changed."""
    suffix = str(dtype).replace("torch.", "")
    write_json(Path(directory) / f"base_fingerprint_{suffix}.json", {"sha256": digest})
