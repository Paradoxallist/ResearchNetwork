from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig, ResetType


@dataclass(frozen=True)
class TickState:
    tick: int
    input_spikes: tuple[int, ...]
    spikes: tuple[int, ...]
    potentials: tuple[float, ...]


@dataclass(frozen=True)
class SimulationTrace:
    config: ExperimentConfig
    ticks: tuple[TickState, ...]


def _per_neuron(value: float | tuple[float, ...], n: int) -> tuple[float, ...]:
    return value if isinstance(value, tuple) else (value,) * n


def simulate(config: ExperimentConfig) -> SimulationTrace:
    n = config.neuron_count
    potentials = config.initial_potentials or (0.0,) * n
    old_spikes = (0,) * n
    thresholds = _per_neuron(config.threshold, n)
    retentions = _per_neuron(config.retention, n)
    trace = []
    incoming = [[] for _ in range(n)]
    for edge in config.recurrent_edges:
        incoming[edge.target].append(edge)
    for tick in range(config.simulation_ticks):
        input_spikes = tuple(int(inp.protocol.spikes_at(tick)) for inp in config.inputs)
        candidates = []
        for target in range(n):
            value = potentials[target] * retentions[target]
            value += sum(edge.weight * old_spikes[edge.source] for edge in incoming[target])
            value += sum(inp.weight * input_spikes[i] for i, inp in enumerate(config.inputs) if target in inp.targets)
            candidates.append(value)
        new_spikes = tuple(int(candidates[i] >= thresholds[i]) for i in range(n))
        new_potentials = []
        for i, candidate in enumerate(candidates):
            if not new_spikes[i]: value = candidate
            elif config.reset_type == ResetType.HARD_RESET: value = 0.0
            elif config.reset_type == ResetType.SUBTRACTIVE_RESET: value = candidate - thresholds[i]
            elif config.reset_type == ResetType.FIXED_RESIDUAL_RESET: value = config.reset_value
            else: value = candidate * config.reset_fraction
            new_potentials.append(value)
        potentials, old_spikes = tuple(new_potentials), new_spikes
        trace.append(TickState(tick, input_spikes, new_spikes, potentials))
    return SimulationTrace(config, tuple(trace))
