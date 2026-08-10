# Tiny Recurrent Circuit Laboratory

A standalone, reproducible research environment for studying the dynamics and stimulus memory of **one-, two-, and three-neuron** recurrent spiking circuits. It does not train networks, change weights, or claim mathematical chaos. The priorities are correctness, reproducibility, inspectability, resumability, usability, then performance.

## Quick start

Use Python 3.10 or newer. The project has no runtime dependencies outside the standard library.

```bash
python main.py web
```

Open `http://127.0.0.1:8765`. The main interface is in the browser:

- `/` — research dashboard
- `/simulator` — interactive one-to-three-neuron laboratory
- `/progress` — batch status (research continues if the browser closes)
- `/results` — filtered, paginated saved results
- `/run/<run_id>` — configuration, raw metrics, classification, and deterministic replay
- `/charts` — result-only phase maps and heatmaps

Run or resume a small example study:

```bash
python main.py study studies/tiny_demo.json
python main.py study studies/tiny_demo.json --no-multiprocessing
python main.py study studies/tiny_demo.json --dry-run
python main.py inspect run_0123456789abcdef
python main.py analyze
```

`--dry-run` prints the requested, completed, and pending counts. Process workers default to one fewer than the available CPU cores; use `--workers N`, or disable multiprocessing for debugging.

## Scientific model

Each stateful neuron uses a discrete synchronous update. For neuron `i` at tick `t`:

```text
candidate_i(t) = retention_i × V_i(t-1)
                 + Σ_j weight(j→i) × spike_j(t-1)
                 + Σ_k input_weight(k→i) × input_spike_k(t)

spike_i(t) = 1 if candidate_i(t) ≥ threshold_i, otherwise 0
```

Every candidate is calculated from the same old internal state, and only then are all neurons committed. A recurrent signal therefore crosses exactly one synapse per tick; neuron iteration order cannot affect the answer. Input sources are stateless external spike generators.

After firing, the potential is:

- `HARD_RESET`: `V = 0`
- `SUBTRACTIVE_RESET`: `V = candidate - threshold` (overshoot is preserved)
- `FIXED_RESIDUAL_RESET`: `V = reset_value`
- `PERCENTAGE_RESET`: `V = candidate × reset_fraction`

Retention is multiplicative and constrained to `[0, 1]`. Weights may be positive or negative and remain fixed. A configuration currently shares reset behavior across neurons, while threshold and retention representations already permit later per-neuron values.

## Architecture

The package deliberately separates concerns:

- `config.py` — immutable, serializable experiment objects and protocols
- `topology.py` — deterministic recurrent/input enumeration and canonical IDs
- `simulator.py` — two-phase numerical engine and optional complete trace
- `metrics.py` — raw measurements and tolerant period detection
- `classification.py` — configurable interpretation rules
- `memory.py` — paired-stimulus state and spike separation
- `studies.py` / `sweep.py` — Cartesian study expansion and parallel execution
- `persistence.py` — safe incremental JSON Lines storage and resume identities
- `analysis.py` — result-only filtering/grouping; it never simulates
- `web.py` / `static/` — local multi-page research interface

Configurations use a canonical JSON encoding and SHA-256 hash over every result-affecting field. The first 16 hex characters form the human-scale run ID; the full hash is stored and accepted for lookup.

## Metrics and interpretation

Batch results contain configuration and summaries, never giant traces. Activity measurements include total and per-neuron spikes/rates, first/last spike, silent fraction, post-forcing activity lifetime, and survival through the final tick. Potential summaries contain minimum, maximum, mean, and final value per neuron.

Period detection examines spike tuples and full `(spikes, potentials)` state only after finite forcing ends. Potentials use an absolute tolerance. A period must repeat for `required_cycles` (default 3) and remain within `maximum_period` (default 32). A repeating all-silent suffix is retained as a raw mathematical observation but is not classified as periodic activity. Indefinitely forced periodic protocols intentionally leave no post-forcing window.

Synchrony is:

```text
ticks with at least two simultaneous internal spikes
────────────────────────────────────────────────────
ticks with at least one internal spike
```

For one active neuron it is defined as 1; for no activity it is 0.

Classification is kept outside the simulator. Initial primary regimes are `DEAD`, `QUIESCENT_WITH_STATE`, `TRANSIENT`, `PERIODIC`, `TONIC`, and `ACTIVE_APERIODIC`. An evoked spike with no post-forcing continuation is dead (or quiescent if state remains); transient means activity continued beyond forcing and later stopped. `HIGH_ACTIVITY` and `HIGH_SYNCHRONY` are diagnostic flags. `ACTIVE_APERIODIC` means only that no stable period was detected within the configured search; it is **not** a claim of chaos.

## Paired input memory

Run the same network and initial state under two stimuli and compare:

```text
D_V(t) = √Σ_i (V_i,A(t) - V_i,B(t))²
D_S(t) = Σ_i |spike_i,A(t) - spike_i,B(t)|
```

`separability()` returns the full distances for inspection plus peak/final separability, memory lifetime, and whether memory remains at the end. The threshold is configurable. Activity common to both trials does not count as memory.

## Studies, persistence, and stopping

Studies are explicit JSON files in `studies/`. Scalar values, lists, or inclusive `{start, stop, step}` ranges define Cartesian dimensions. `topologies: "all"` and `input_configurations: "all_valid"` invoke deterministic enumeration. Two-input target sets are always different. Self-connections are opt-in.

Study expansion computes the run count before execution. More than 100,000 runs require an intentional `"allow_large_sweep": true`; this is a warning gate, not a scientific limit.

Each completed run is appended immediately to `results/results.jsonl`, flushed, and synced to disk by the single parent writer. Workers never touch the result file. A small atomically replaced `results/progress.json` sidecar lets a separately running web server show live progress; the sweep does not depend on the page or server. At startup, existing configuration hashes are loaded and skipped. Ctrl+C stops new scheduling, drains completed work, writes it, and exits; a later invocation resumes the missing configurations. Readers ignore an incomplete trailing line, so analysis can safely inspect a file while another process appends to it.

Detailed traces are regenerated, not stored. A saved configuration therefore remains small but is sufficient for exact inspection/replay.

## Adding a study

Copy `studies/tiny_demo.json`, choose a new `study_id`, and narrow the scientific question. Set neuron count, topology policy, inputs, reset types, thresholds, retention/weight ranges, and simulation ticks. First run with `--dry-run`, review the count, then launch it. Individual recurrent edge weights are supported directly by `ExperimentConfig`; study JSON currently offers the common-weight sweep, keeping shared and per-edge weighting conceptually distinct.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers synchronous propagation, all resets, retention, threshold firing, signed weights, topology/input enumeration, target-set uniqueness, deterministic hashes, JSONL durability and resume, period detection, classification, and paired-state separability with analytically predictable circuits.

## Reference-project note

The requested `../RecurringGraphBeta` reference repository was not present when this project was created, so it was neither read nor modified. The new project implements the explicitly requested conceptual patterns independently: directed synapses, old-state/two-phase ticks, clean server/frontend boundaries, and transparent scientific documentation.
