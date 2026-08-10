import math

import pytest

from research_graph import InputNeuron, StatefulNeuron


def test_neurons_start_at_zero():
    input_node = InputNeuron("input")
    neuron = StatefulNeuron("A")
    assert input_node.spike == 0
    assert neuron.potential == 0.0
    assert neuron.spike == 0


@pytest.mark.parametrize("threshold", [0.0, -1.0, math.nan, math.inf])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        StatefulNeuron("A", threshold=threshold)


@pytest.mark.parametrize("retention", [-0.01, 1.01, math.nan, math.inf])
def test_invalid_retention_is_rejected(retention):
    with pytest.raises(ValueError):
        StatefulNeuron("A", retention=retention)


def test_empty_neuron_id_is_rejected():
    with pytest.raises(ValueError):
        InputNeuron("  ")
