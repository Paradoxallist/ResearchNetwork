"""Strictly synchronous recurrent network model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .neuron import InputNeuron, StatefulNeuron
from .reset import reset_potential
from .snapshot import InputState, NeuronState, TickSnapshot
from .synapse import Synapse


class Model:
    """A fixed network whose state advances in two synchronous phases."""

    def __init__(self, inputs: Sequence[InputNeuron], neurons: Sequence[StatefulNeuron], synapses: Sequence[Synapse]):
        self.inputs = tuple(inputs)
        self.neurons = tuple(neurons)
        self.synapses = tuple(synapses)
        all_nodes = (*self.inputs, *self.neurons)
        ids = [node.neuron_id for node in all_nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("all neuron IDs in a model must be unique")
        if len({id(node) for node in all_nodes}) != len(all_nodes):
            raise ValueError("the same neuron object cannot occur twice")
        node_identities = {id(node) for node in all_nodes}
        stateful_identities = {id(node) for node in self.neurons}
        for synapse in self.synapses:
            if id(synapse.source) not in node_identities or id(synapse.target) not in stateful_identities:
                raise ValueError("every synapse endpoint must belong to this model")
        self.age = 0

    def tick(self, input_spikes: Mapping[str, int] | None = None) -> TickSnapshot:
        """Advance one tick and return the immutable committed state.

        Missing inputs are zero. Unknown IDs and non-binary values are rejected.
        All stateful next states are calculated before any are committed.
        """

        supplied = dict(input_spikes or {})
        known_inputs = {node.neuron_id for node in self.inputs}
        unknown = set(supplied) - known_inputs
        if unknown:
            raise ValueError(f"unknown input IDs: {sorted(unknown)}")
        for neuron_id, value in supplied.items():
            if type(value) is not int or value not in (0, 1):
                raise ValueError(f"input {neuron_id!r} must be integer 0 or 1")

        # External spikes are current-tick signals. Stateful spikes remain the
        # previous committed values throughout the calculation phase.
        next_inputs = {node.neuron_id: supplied.get(node.neuron_id, 0) for node in self.inputs}
        incoming: dict[int, float] = {id(node): 0.0 for node in self.neurons}
        for synapse in self.synapses:
            if not synapse.enabled:
                continue
            source_spike = (next_inputs[synapse.source.neuron_id]
                            if isinstance(synapse.source, InputNeuron)
                            else synapse.source.spike)
            incoming[id(synapse.target)] += synapse.weight * source_spike

        planned: list[tuple[StatefulNeuron, float, float, int, float]] = []
        for neuron in self.neurons:
            signal = incoming[id(neuron)]
            candidate = neuron.retention * neuron.potential + signal
            spike = int(candidate >= neuron.threshold)
            potential = reset_potential(candidate, neuron.threshold, neuron.reset) if spike else candidate
            planned.append((neuron, signal, candidate, spike, potential))

        for node in self.inputs:
            node.spike = next_inputs[node.neuron_id]
        for neuron, _, _, spike, potential in planned:
            neuron.spike = spike
            neuron.potential = float(potential)
        self.age += 1

        return TickSnapshot(
            tick=self.age,
            inputs=tuple(InputState(node.neuron_id, node.spike) for node in self.inputs),
            neurons=tuple(NeuronState(node.neuron_id, node.potential, node.spike, signal, candidate)
                          for node, signal, candidate, _, _ in planned),
        )

    def reset(self) -> None:
        """Restore tick and all dynamic state while preserving topology and parameters."""

        self.age = 0
        for node in self.inputs:
            node.spike = 0
        for neuron in self.neurons:
            neuron.potential = 0.0
            neuron.spike = 0
