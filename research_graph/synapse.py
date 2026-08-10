"""Immutable directed weighted synapses."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .neuron import InputNeuron, StatefulNeuron


@dataclass(frozen=True)
class Synapse:
    """A fixed directed connection into a stateful neuron."""

    source: InputNeuron | StatefulNeuron
    target: StatefulNeuron
    weight: float
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.source, (InputNeuron, StatefulNeuron)):
            raise TypeError("source must be an InputNeuron or StatefulNeuron")
        if not isinstance(self.target, StatefulNeuron):
            raise TypeError("synapse target must be a StatefulNeuron")
        if not isfinite(self.weight):
            raise ValueError("synapse weight must be finite")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
