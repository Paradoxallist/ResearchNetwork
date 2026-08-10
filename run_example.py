"""Temporary hand-verification example for the Stage 1 core."""

from research_graph import InputNeuron, Model, StatefulNeuron, Synapse


source = InputNeuron("input")
a = StatefulNeuron("A", threshold=1.0, retention=0.0)
b = StatefulNeuron("B", threshold=1.0, retention=0.0)
model = Model(
    inputs=[source],
    neurons=[a, b],
    synapses=[Synapse(source, a, 1.0), Synapse(a, b, 1.0), Synapse(b, a, 1.0)],
)

print("tick | input | A spike |  A V | B spike |  B V")
print("-----+-------+---------+------+---------+------")
for tick in range(1, 11):
    snapshot = model.tick({"input": int(tick == 1)})
    a_state, b_state = snapshot.neuron("A"), snapshot.neuron("B")
    print(f"{tick:>4} | {snapshot.input('input').spike:^5} | {a_state.spike:^7} | {a_state.potential:>4.1f} | {b_state.spike:^7} | {b_state.potential:>4.1f}")
