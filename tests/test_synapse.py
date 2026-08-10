import math

import pytest

from research_graph import InputNeuron, StatefulNeuron, Synapse


def test_positive_and_negative_arbitrary_finite_weights_are_allowed():
    source, target = InputNeuron("input"), StatefulNeuron("A")
    assert Synapse(source, target, 1234.5).weight == 1234.5
    assert Synapse(source, target, -987.25).weight == -987.25


def test_input_neuron_cannot_be_synapse_target():
    a, input_node = StatefulNeuron("A"), InputNeuron("input")
    with pytest.raises(TypeError):
        Synapse(a, input_node, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf])
def test_nonfinite_weight_is_rejected(weight):
    with pytest.raises(ValueError):
        Synapse(InputNeuron("input"), StatefulNeuron("A"), weight)


def test_enabled_must_be_boolean():
    with pytest.raises(TypeError):
        Synapse(InputNeuron("input"), StatefulNeuron("A"), 1.0, enabled=1)  # type: ignore[arg-type]
