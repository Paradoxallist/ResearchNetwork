# ResearchNetwork — Stage 1

This repository currently contains only a small deterministic simulation core for recurrent spiking circuits. Stage 1 is deliberately limited to input neurons, stateful neurons, weighted directed synapses, synchronous ticks, reset behavior, immutable tick snapshots, and tests.

## Dynamics

For stateful neuron `i`, each tick computes:

```text
incoming_i  = Σ weight(j→i) × previous_spike_j
candidate_i = retention_i × previous_potential_i + incoming_i
spike_i     = 1 if candidate_i >= threshold_i, otherwise 0
```

External input values belong to the current tick, so an input can affect its direct targets immediately. A stateful neuron's outgoing signal uses its previously committed spike. All next states are calculated before any stateful neuron is committed. Consequently, a signal crosses at most one synapse per tick and neuron iteration order cannot change the result.

## Reset modes

After a threshold crossing:

- `HARD`: potential becomes `0`.
- `SUBTRACTIVE`: potential becomes `candidate - threshold`.
- `FIXED_RESIDUAL`: potential becomes the configured finite `reset_value`.
- `PERCENTAGE`: potential becomes `candidate × reset_fraction`, where the fraction is in `[0, 1]`.

When no spike occurs, the committed potential is the candidate unchanged. Negative potentials and arbitrary finite positive or negative synaptic weights are supported.

## Run the tests

```powershell
python -m pytest -q
```

## Run the manual example

```powershell
python run_example.py
```

## Intentional Stage 1 exclusions

Web visualization, topology enumeration, parameter sweeps, batch execution, JSON persistence, plots, metrics, classification, analysis, learning, threading, and multiprocessing are intentionally **not implemented yet**. They will only be considered after this core has been verified.
