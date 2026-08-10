from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ResetType(str, Enum):
    HARD_RESET = "HARD_RESET"
    SUBTRACTIVE_RESET = "SUBTRACTIVE_RESET"
    FIXED_RESIDUAL_RESET = "FIXED_RESIDUAL_RESET"
    PERCENTAGE_RESET = "PERCENTAGE_RESET"


class ProtocolType(str, Enum):
    SINGLE_IMPULSE = "SINGLE_IMPULSE"
    PERIODIC = "PERIODIC"
    FINITE_PERIODIC = "FINITE_PERIODIC"
    MANUAL = "MANUAL"


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    weight: float = 1.0


@dataclass(frozen=True)
class InputProtocol:
    kind: ProtocolType = ProtocolType.SINGLE_IMPULSE
    start_tick: int = 0
    period: int = 1
    spike_count: int | None = 1
    duration: int | None = None
    manual_ticks: tuple[int, ...] = ()

    def spikes_at(self, tick: int) -> bool:
        if self.kind == ProtocolType.MANUAL:
            return tick in self.manual_ticks
        if tick < self.start_tick:
            return False
        if self.kind == ProtocolType.SINGLE_IMPULSE:
            return tick == self.start_tick
        if self.period <= 0 or (tick - self.start_tick) % self.period:
            return False
        ordinal = (tick - self.start_tick) // self.period
        if self.kind == ProtocolType.FINITE_PERIODIC:
            if self.spike_count is not None and ordinal >= self.spike_count:
                return False
            if self.duration is not None and tick >= self.start_tick + self.duration:
                return False
        return True

    def forcing_end(self, simulation_ticks: int) -> int:
        if self.kind == ProtocolType.PERIODIC:
            return simulation_ticks - 1
        ticks = [t for t in range(simulation_ticks) if self.spikes_at(t)]
        return max(ticks, default=-1)


@dataclass(frozen=True)
class InputConfig:
    targets: tuple[int, ...]
    weight: float = 1.0
    protocol: InputProtocol = field(default_factory=InputProtocol)


@dataclass(frozen=True)
class ClassificationConfig:
    zero_epsilon: float = 1e-9
    state_tolerance: float = 1e-8
    maximum_period: int = 32
    required_cycles: int = 3
    high_activity_rate: float = 0.8
    high_synchrony: float = 0.8


@dataclass(frozen=True)
class ExperimentConfig:
    study_id: str
    neuron_count: int
    recurrent_edges: tuple[Edge, ...] = ()
    inputs: tuple[InputConfig, ...] = ()
    threshold: float | tuple[float, ...] = 1.0
    retention: float | tuple[float, ...] = 0.0
    reset_type: ResetType = ResetType.HARD_RESET
    reset_value: float = 0.0
    reset_fraction: float = 0.0
    simulation_ticks: int = 200
    initial_potentials: tuple[float, ...] = ()
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    model_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.neuron_count not in (1, 2, 3):
            raise ValueError("neuron_count must be 1, 2, or 3")
        if not 0 < self.simulation_ticks:
            raise ValueError("simulation_ticks must be positive")
        for name, value in (("retention", self.retention),):
            values = value if isinstance(value, tuple) else (value,)
            if len(values) not in (1, self.neuron_count) or any(not 0 <= x <= 1 for x in values):
                raise ValueError(f"{name} must contain values in [0, 1]")
        if len(self.inputs) > (1 if self.neuron_count == 1 else 2):
            raise ValueError("too many inputs")
        if len({i.targets for i in self.inputs}) != len(self.inputs):
            raise ValueError("input target sets must differ")
        for edge in self.recurrent_edges:
            if edge.source not in range(self.neuron_count) or edge.target not in range(self.neuron_count):
                raise ValueError("edge endpoint outside network")
        for inp in self.inputs:
            if not inp.targets or any(t not in range(self.neuron_count) for t in inp.targets):
                raise ValueError("invalid input targets")

    @staticmethod
    def _normal(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: ExperimentConfig._normal(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [ExperimentConfig._normal(v) for v in value]
        if isinstance(value, float):
            return float(format(value, ".15g"))
        return value

    def to_dict(self) -> dict[str, Any]:
        return self._normal(asdict(self))

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def run_id(self) -> str:
        return f"run_{self.config_hash[:16]}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        d = dict(data)
        d["reset_type"] = ResetType(d["reset_type"])
        d["recurrent_edges"] = tuple(Edge(**x) for x in d.get("recurrent_edges", ()))
        inputs = []
        for x in d.get("inputs", ()):
            p = dict(x.get("protocol", {})); p["kind"] = ProtocolType(p.get("kind", "SINGLE_IMPULSE"))
            p["manual_ticks"] = tuple(p.get("manual_ticks", ()))
            inputs.append(InputConfig(tuple(x["targets"]), x.get("weight", 1.0), InputProtocol(**p)))
        d["inputs"] = tuple(inputs)
        d["initial_potentials"] = tuple(d.get("initial_potentials", ()))
        if isinstance(d.get("threshold"), list): d["threshold"] = tuple(d["threshold"])
        if isinstance(d.get("retention"), list): d["retention"] = tuple(d["retention"])
        d["classification"] = ClassificationConfig(**d.get("classification", {}))
        return cls(**d)
