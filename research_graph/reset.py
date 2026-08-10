"""Spike reset policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ResetMode(str, Enum):
    """Supported post-spike potential reset mechanisms."""

    HARD = "HARD"
    SUBTRACTIVE = "SUBTRACTIVE"
    FIXED_RESIDUAL = "FIXED_RESIDUAL"
    PERCENTAGE = "PERCENTAGE"


@dataclass(frozen=True)
class ResetConfig:
    """Immutable reset mode and its optional parameter."""

    mode: ResetMode = ResetMode.HARD
    reset_value: float = 0.0
    reset_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ResetMode):
            raise TypeError("mode must be a ResetMode")
        if not isfinite(self.reset_value):
            raise ValueError("reset_value must be finite")
        if not isfinite(self.reset_fraction) or not 0.0 <= self.reset_fraction <= 1.0:
            raise ValueError("reset_fraction must be finite and between 0 and 1")


def reset_potential(candidate: float, threshold: float, config: ResetConfig) -> float:
    """Return the potential committed after a firing event."""

    if config.mode is ResetMode.HARD:
        return 0.0
    if config.mode is ResetMode.SUBTRACTIVE:
        return candidate - threshold
    if config.mode is ResetMode.FIXED_RESIDUAL:
        return config.reset_value
    if config.mode is ResetMode.PERCENTAGE:
        return candidate * config.reset_fraction
    raise ValueError(f"unsupported reset mode: {config.mode!r}")
