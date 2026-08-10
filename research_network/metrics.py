from __future__ import annotations

from math import isclose
from statistics import fmean
from typing import Callable, Sequence

from .simulator import SimulationTrace, TickState


def detect_period(states: Sequence, start: int, maximum_period: int, required_cycles: int, equal: Callable[[object, object], bool] = lambda a, b: a == b) -> tuple[int | None, int | None]:
    for period in range(1, maximum_period + 1):
        span = period * required_cycles
        if len(states) - start < span:
            continue
        cycle_start = len(states) - span
        if cycle_start < start:
            continue
        if all(equal(states[i], states[i - period]) for i in range(cycle_start + period, len(states))):
            while cycle_start - 1 >= start and equal(states[cycle_start - 1], states[cycle_start - 1 + period]):
                cycle_start -= 1
            return period, cycle_start
    return None, None


def calculate_metrics(trace: SimulationTrace) -> dict:
    c, ticks, n = trace.config, trace.ticks, trace.config.neuron_count
    spike_counts = [sum(t.spikes[i] for t in ticks) for i in range(n)]
    active_ticks = [t.tick for t in ticks if any(t.spikes)]
    total = sum(spike_counts)
    forced_end = max((i.protocol.forcing_end(c.simulation_ticks) for i in c.inputs), default=-1)
    start = forced_end + 1
    cc = c.classification
    spike_states = [t.spikes for t in ticks]
    full_states = [(t.spikes, t.potentials) for t in ticks]
    p_spike, p_start = detect_period(spike_states, start, cc.maximum_period, cc.required_cycles)
    eq = lambda a, b: a[0] == b[0] and all(isclose(x, y, abs_tol=cc.state_tolerance, rel_tol=0) for x, y in zip(a[1], b[1]))
    p_full, f_start = detect_period(full_states, start, cc.maximum_period, cc.required_cycles, eq)
    potentials = [[t.potentials[i] for t in ticks] for i in range(n)]
    denom_active = sum(any(t.spikes) for t in ticks)
    synchrony = (sum(sum(t.spikes) >= 2 for t in ticks) / denom_active) if n > 1 and denom_active else (1.0 if n == 1 and denom_active else 0.0)
    periodic_activity = bool((p_spike or p_full) and any(any(t.spikes) for t in ticks[(f_start if f_start is not None else p_start):]))
    recent_activity = any(ticks[-1].spikes)
    return {
        "total_spikes": total, "spikes_per_neuron": spike_counts,
        "spike_rate_network": total / (len(ticks) * n),
        "spike_rate_per_neuron": [x / len(ticks) for x in spike_counts],
        "first_internal_spike_tick": min(active_ticks, default=None),
        "last_internal_spike_tick": max(active_ticks, default=None),
        "fraction_silent_ticks": sum(not any(t.spikes) for t in ticks) / len(ticks),
        "activity_lifetime": (max(active_ticks) - forced_end) if active_ticks and max(active_ticks) > forced_end else 0,
        "activity_survived_until_end": bool(periodic_activity or recent_activity),
        "potential_per_neuron": [{"minimum": min(v), "maximum": max(v), "mean": fmean(v), "final": v[-1]} for v in potentials],
        "spike_period_detected": p_spike is not None, "spike_period": p_spike,
        "full_state_period_detected": p_full is not None, "full_state_period": p_full,
        "period_start_tick": f_start if f_start is not None else p_start,
        "periodic_activity_detected": periodic_activity,
        "synchrony": synchrony, "forcing_end_tick": forced_end,
    }
