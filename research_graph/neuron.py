"""External input and stateful neuron definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from .reset import ResetConfig


def _validate_id(neuron_id: str) -> None:
    if not isinstance(neuron_id, str) or not neuron_id.strip():
        raise ValueError("neuron_id must be a non-empty string")


@dataclass
class InputNeuron:
    """An externally controlled binary spike source with no internal state."""

    neuron_id: str
    spike: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        _validate_id(self.neuron_id)


@dataclass
class StatefulNeuron:
    """A neuron with membrane potential, threshold, retention, and reset policy."""

    neuron_id: str
    threshold: float = 1.0
    retention: float = 0.0
    reset: ResetConfig = field(default_factory=ResetConfig)
    potential: float = field(default=0.0, init=False)
    spike: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        _validate_id(self.neuron_id)
        if not isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("threshold must be finite and greater than zero")
        if not isfinite(self.retention) or not 0.0 <= self.retention <= 1.0:
            raise ValueError("retention must be finite and between 0 and 1")
        if not isinstance(self.reset, ResetConfig):
            raise TypeError("reset must be a ResetConfig")
