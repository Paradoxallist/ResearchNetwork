from __future__ import annotations

import os
import signal
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from time import monotonic
from typing import Callable

from .config import ExperimentConfig
from .persistence import JsonlStore
from .results import run_experiment


@dataclass
class SweepProgress:
    study_id: str; total: int; skipped: int = 0; completed: int = 0; failures: int = 0
    active_workers: int = 0; started: float = field(default_factory=monotonic); recent: list[dict] = field(default_factory=list)
    @property
    def remaining(self): return self.total - self.skipped - self.completed - self.failures
    def to_dict(self):
        elapsed = monotonic() - self.started
        rate = self.completed / elapsed if elapsed else 0
        return {"study_id": self.study_id, "total": self.total, "skipped": self.skipped, "completed": self.completed,
                "remaining": self.remaining, "failures": self.failures, "active_workers": self.active_workers,
                "percentage": 100 * (self.skipped + self.completed + self.failures) / self.total if self.total else 100,
                "runs_per_second": rate, "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": self.remaining / rate if rate else None, "recent": self.recent[-10:]}


def execute(configs: list[ExperimentConfig], store: JsonlStore, workers: int | None = None,
            multiprocessing: bool = True, on_progress: Callable[[SweepProgress], None] | None = None) -> SweepProgress:
    completed = store.completed_hashes(); pending = [c for c in configs if c.config_hash not in completed]
    progress = SweepProgress(configs[0].study_id if configs else "empty", len(configs), len(configs) - len(pending))
    stopped = False
    old_handler = signal.getsignal(signal.SIGINT)
    def stop(*_):
        nonlocal stopped; stopped = True
    signal.signal(signal.SIGINT, stop)
    try:
        if not multiprocessing or (workers == 1):
            for c in pending:
                if stopped: break
                try:
                    record = run_experiment(c); store.append(record); progress.completed += 1; progress.recent.append(record)
                except Exception: progress.failures += 1
                if on_progress: on_progress(progress)
            return progress
        count = workers or max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=count) as pool:
            iterator = iter(pending); futures = {}
            while not stopped and len(futures) < count * 2:
                try: futures[pool.submit(run_experiment, next(iterator))] = True
                except StopIteration: break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED); progress.active_workers = len(futures)
                for future in done:
                    futures.pop(future)
                    try:
                        record = future.result(); store.append(record); progress.completed += 1; progress.recent.append(record)
                    except Exception: progress.failures += 1
                    if not stopped:
                        try: futures[pool.submit(run_experiment, next(iterator))] = True
                        except StopIteration: pass
                if on_progress: on_progress(progress)
            progress.active_workers = 0
            return progress
    finally:
        signal.signal(signal.SIGINT, old_handler)
