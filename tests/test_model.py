from dataclasses import FrozenInstanceError

import pytest

from research_graph import InputNeuron, Model, ResetConfig, ResetMode, StatefulNeuron, Synapse


def single_neuron(*, weight=1.0, threshold=1.0, retention=0.0, reset=ResetConfig()):
    input_node = InputNeuron("input")
    neuron = StatefulNeuron("A", threshold=threshold, retention=retention, reset=reset)
    return Model([input_node], [neuron], [Synapse(input_node, neuron, weight)])


def test_no_input_remains_zero_for_many_ticks():
    model = single_neuron()
    for _ in range(20):
        state = model.tick().neuron("A")
        assert (state.potential, state.spike, state.incoming_signal, state.candidate) == (0.0, 0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("reset", "expected"),
    [
        (ResetConfig(ResetMode.HARD), 0.0),
        (ResetConfig(ResetMode.SUBTRACTIVE), 0.2),
        (ResetConfig(ResetMode.FIXED_RESIDUAL, reset_value=-0.3), -0.3),
        (ResetConfig(ResetMode.PERCENTAGE, reset_fraction=0.5), 0.6),
    ],
)
def test_subthreshold_accumulation_then_each_reset_mode(reset, expected):
    model = single_neuron(weight=0.4, retention=1.0, reset=reset)
    first = model.tick({"input": 1}).neuron("A")
    second = model.tick({"input": 1}).neuron("A")
    third = model.tick({"input": 1}).neuron("A")
    assert (first.potential, first.spike) == pytest.approx((0.4, 0))
    assert (second.potential, second.spike) == pytest.approx((0.8, 0))
    assert third.candidate == pytest.approx(1.2)
    assert third.spike == 1
    assert third.potential == pytest.approx(expected)


def test_retention_halves_potential_without_input():
    model = single_neuron(weight=1.0, threshold=10.0, retention=0.5)
    values = [model.tick({"input": 1}).neuron("A").potential]
    values.extend(model.tick().neuron("A").potential for _ in range(3))
    assert values == pytest.approx([1.0, 0.5, 0.25, 0.125])


def test_negative_input_reduces_and_can_make_potential_negative():
    positive = InputNeuron("positive")
    negative = InputNeuron("negative")
    neuron = StatefulNeuron("A", threshold=10.0, retention=1.0)
    model = Model([positive, negative], [neuron], [Synapse(positive, neuron, 0.5), Synapse(negative, neuron, -0.8)])
    assert model.tick({"positive": 1}).neuron("A").potential == pytest.approx(0.5)
    state = model.tick({"negative": 1}).neuron("A")
    assert state.incoming_signal == pytest.approx(-0.8)
    assert state.potential == pytest.approx(-0.3)


def test_one_synapse_per_tick_propagation():
    source, a, b = InputNeuron("input"), StatefulNeuron("A"), StatefulNeuron("B")
    model = Model([source], [a, b], [Synapse(source, a, 1.0), Synapse(a, b, 1.0)])
    tick1 = model.tick({"input": 1})
    tick2 = model.tick()
    assert (tick1.neuron("A").spike, tick1.neuron("B").spike) == (1, 0)
    assert tick1.neuron("B").incoming_signal == 0.0
    assert (tick2.neuron("A").spike, tick2.neuron("B").spike) == (0, 1)


def _ordered_model(reverse=False):
    source, a, b = InputNeuron("input"), StatefulNeuron("A"), StatefulNeuron("B")
    neurons = [b, a] if reverse else [a, b]
    return Model([source], neurons, [Synapse(source, a, 1.0), Synapse(a, b, 1.0), Synapse(b, a, 1.0)])


def test_iteration_order_independence():
    left, right = _ordered_model(False), _ordered_model(True)
    for external in ({"input": 1}, {}, {}, {}, {}):
        a, b = left.tick(external), right.tick(external)
        logical_a = {state.neuron_id: (state.spike, state.potential, state.incoming_signal, state.candidate) for state in a.neurons}
        logical_b = {state.neuron_id: (state.spike, state.potential, state.incoming_signal, state.candidate) for state in b.neurons}
        assert logical_a == logical_b


def test_two_neuron_recurrent_circuit_alternates_exactly():
    model = _ordered_model()
    observed = []
    for external in ({"input": 1}, {}, {}, {}, {}, {}):
        snapshot = model.tick(external)
        observed.append((snapshot.neuron("A").spike, snapshot.neuron("B").spike))
    assert observed == [(1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)]


def test_self_loop_can_sustain_spiking():
    source, a = InputNeuron("input"), StatefulNeuron("A")
    model = Model([source], [a], [Synapse(source, a, 1.0), Synapse(a, a, 1.0)])
    assert [model.tick({"input": int(tick == 0)}).neuron("A").spike for tick in range(5)] == [1, 1, 1, 1, 1]


def test_weak_self_loop_decays_predictably():
    source, a = InputNeuron("input"), StatefulNeuron("A", retention=0.5)
    model = Model([source], [a], [Synapse(source, a, 1.0), Synapse(a, a, 0.5)])
    states = [model.tick({"input": 1}).neuron("A"), model.tick().neuron("A"), model.tick().neuron("A")]
    assert [(s.spike, s.potential) for s in states] == pytest.approx([(1, 0.0), (0, 0.5), (0, 0.25)])


def test_disabled_synapse_transmits_zero():
    source, a = InputNeuron("input"), StatefulNeuron("A")
    model = Model([source], [a], [Synapse(source, a, 100.0, enabled=False)])
    state = model.tick({"input": 1}).neuron("A")
    assert state.incoming_signal == 0.0
    assert (state.spike, state.potential) == (0, 0.0)


@pytest.mark.parametrize("bad", [-1, 2, 0.5, "1", None, True])
def test_nonbinary_external_input_is_rejected(bad):
    with pytest.raises(ValueError):
        single_neuron().tick({"input": bad})


def test_missing_input_is_zero_and_unknown_input_is_rejected():
    model = single_neuron()
    assert model.tick().input("input").spike == 0
    with pytest.raises(ValueError):
        model.tick({"typo": 1})


def test_snapshot_is_immutable_and_contains_committed_transition_state():
    snapshot = single_neuron(weight=0.4, retention=1.0).tick({"input": 1})
    state = snapshot.neuron("A")
    assert (snapshot.tick, state.incoming_signal, state.candidate, state.potential, state.spike) == pytest.approx((1, 0.4, 0.4, 0.4, 0))
    with pytest.raises(FrozenInstanceError):
        state.spike = 1  # type: ignore[misc]


def test_reset_restores_dynamic_state_but_keeps_topology():
    model = single_neuron(weight=0.4, retention=1.0)
    model.tick({"input": 1})
    synapses_before = model.synapses
    model.reset()
    assert model.age == 0
    assert model.inputs[0].spike == 0
    assert (model.neurons[0].potential, model.neurons[0].spike) == (0.0, 0)
    assert model.synapses is synapses_before


def test_duplicate_ids_and_external_synapse_endpoints_are_rejected():
    with pytest.raises(ValueError):
        Model([InputNeuron("A")], [StatefulNeuron("A")], [])
    source, target, outsider = InputNeuron("input"), StatefulNeuron("A"), StatefulNeuron("outside")
    with pytest.raises(ValueError):
        Model([source], [target], [Synapse(source, outsider, 1.0)])
