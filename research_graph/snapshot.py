"""Immutable observable state returned after each completed tick."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputState:
    neuron_id: str
    spike: int


@dataclass(frozen=True)
class NeuronState:
    neuron_id: str
    potential: float
    spike: int
    incoming_signal: float
    candidate: float


@dataclass(frozen=True)
class TickSnapshot:
    """Committed state after one tick, including transition observables."""

    tick: int
    inputs: tuple[InputState, ...]
    neurons: tuple[NeuronState, ...]

    def input(self, neuron_id: str) -> InputState:
        """Return one input state by ID."""
        return next(state for state in self.inputs if state.neuron_id == neuron_id)

    def neuron(self, neuron_id: str) -> NeuronState:
        """Return one stateful neuron state by ID."""
        return next(state for state in self.neurons if state.neuron_id == neuron_id)
