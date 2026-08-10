from __future__ import annotations

from math import sqrt

from .simulator import SimulationTrace
from .config import ExperimentConfig
from .simulator import simulate


def separability(a: SimulationTrace, b: SimulationTrace, threshold: float = 1e-6) -> dict:
    if a.config.neuron_count != b.config.neuron_count or len(a.ticks) != len(b.ticks):
        raise ValueError("paired traces must have equal shape")
    dv, ds = [], []
    for ta, tb in zip(a.ticks, b.ticks):
        dv.append(sqrt(sum((x - y) ** 2 for x, y in zip(ta.potentials, tb.potentials))))
        ds.append(sum(abs(x - y) for x, y in zip(ta.spikes, tb.spikes)))
    combined = [max(v, float(s > 0)) for v, s in zip(dv, ds)]
    distinguishable = [i for i, x in enumerate(combined) if x > threshold]
    return {"potential_distance": dv, "spike_distance": ds,
            "peak_state_separability": max(combined, default=0.0),
            "final_state_separability": combined[-1] if combined else 0.0,
            "memory_lifetime": max(distinguishable, default=-1) + 1,
            "memory_survived_until_end": bool(distinguishable and distinguishable[-1] == len(combined) - 1),
            "memory_threshold": threshold}


def run_paired(a: ExperimentConfig, b: ExperimentConfig, threshold: float = 1e-6, retain_trace: bool = False) -> dict:
    """Run paired stimuli against an otherwise identical network and initial state."""
    structural = ("neuron_count", "recurrent_edges", "threshold", "retention", "reset_type", "reset_value",
                  "reset_fraction", "simulation_ticks", "initial_potentials", "model_version")
    if any(getattr(a, key) != getattr(b, key) for key in structural):
        raise ValueError("paired trials must share network, dynamics, duration, and initial state")
    metrics = separability(simulate(a), simulate(b), threshold)
    if not retain_trace:
        metrics.pop("potential_distance"); metrics.pop("spike_distance")
    return {"trial_a": a.to_dict(), "trial_b": b.to_dict(), "metrics": metrics}
