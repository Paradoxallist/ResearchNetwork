from __future__ import annotations

from collections import Counter, defaultdict
from .persistence import JsonlStore


class ResultAnalysis:
    """Read-only result analysis; it never invokes the simulator."""
    def __init__(self, store: JsonlStore): self.store = store
    def filter(self, **criteria):
        rows = list(self.store.records())
        for path, wanted in criteria.items():
            keys = path.split(".")
            def get(row):
                for key in keys: row = row.get(key) if isinstance(row, dict) else None
                return row
            rows = [row for row in rows if get(row) == wanted]
        return rows
    def group_counts(self, path="classification.primary_regime"):
        keys = path.split(".")
        def get(row):
            for key in keys: row = row.get(key) if isinstance(row, dict) else None
            return row
        return dict(Counter(get(r) for r in self.store.records()))
