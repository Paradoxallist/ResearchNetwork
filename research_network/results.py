from __future__ import annotations

from datetime import datetime, timezone

from .classification import classify
from .config import ExperimentConfig
from .metrics import calculate_metrics
from .simulator import simulate
from .topology import topology_id


def run_experiment(config: ExperimentConfig) -> dict:
    trace = simulate(config)
    metrics = calculate_metrics(trace)
    return {"schema_version": "1.0", "model_version": config.model_version, "run_id": config.run_id, "config_hash": config.config_hash, "study_id": config.study_id,
            "timestamp": datetime.now(timezone.utc).isoformat(), "topology_id": topology_id(config.neuron_count, config.recurrent_edges),
            "config": config.to_dict(), "metrics": metrics, "classification": classify(metrics, config.classification)}
