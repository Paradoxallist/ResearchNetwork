import math

import pytest

from research_graph.reset import ResetConfig, ResetMode, reset_potential


def test_hard_reset():
    assert reset_potential(1.4, 1.0, ResetConfig(ResetMode.HARD)) == 0.0


def test_subtractive_reset_preserves_overshoot():
    assert reset_potential(1.4, 1.0, ResetConfig(ResetMode.SUBTRACTIVE)) == pytest.approx(0.4)


def test_fixed_residual_reset_is_exact_and_unrestricted():
    assert reset_potential(1.4, 1.0, ResetConfig(ResetMode.FIXED_RESIDUAL, reset_value=-2.5)) == -2.5


def test_percentage_reset_uses_candidate():
    assert reset_potential(1.4, 1.0, ResetConfig(ResetMode.PERCENTAGE, reset_fraction=0.5)) == pytest.approx(0.7)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, math.nan, math.inf])
def test_invalid_percentage_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        ResetConfig(ResetMode.PERCENTAGE, reset_fraction=fraction)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_fixed_residual_is_rejected(value):
    with pytest.raises(ValueError):
        ResetConfig(ResetMode.FIXED_RESIDUAL, reset_value=value)
