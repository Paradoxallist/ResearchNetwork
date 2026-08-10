from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable, Iterator


class JsonlStore:
    """Append-only, single-writer store; incomplete trailing lines are ignored."""
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def records(self) -> Iterator[dict]:
        if not self.path.exists(): return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try: yield json.loads(line)
                except json.JSONDecodeError: continue

    def completed_hashes(self) -> set[str]:
        return {x["config_hash"] for x in self.records() if "config_hash" in x}

    def append(self, record: dict) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n"); handle.flush(); os.fsync(handle.fileno())

    def find(self, identifier: str) -> dict | None:
        for record in self.records():
            if record.get("run_id") == identifier or record.get("config_hash") == identifier:
                return record
        return None
