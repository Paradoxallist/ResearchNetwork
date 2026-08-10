from __future__ import annotations

from itertools import combinations

from .config import Edge, InputConfig


def topology_id(neuron_count: int, edges: tuple[Edge, ...]) -> str:
    pairs = sorted((e.source, e.target) for e in edges)
    return f"n{neuron_count}_" + ("none" if not pairs else "-".join(f"{a}{b}" for a, b in pairs))


def enumerate_topologies(neuron_count: int, allow_self: bool = False, include_empty: bool = True) -> list[tuple[str, tuple[Edge, ...]]]:
    if neuron_count not in (1, 2, 3):
        raise ValueError("only 1-3 neurons are supported")
    possible = [(a, b) for a in range(neuron_count) for b in range(neuron_count) if allow_self or a != b]
    result = []
    for mask in range(0 if include_empty else 1, 1 << len(possible)):
        edges = tuple(Edge(a, b) for bit, (a, b) in enumerate(possible) if mask & (1 << bit))
        result.append((topology_id(neuron_count, edges), edges))
    return result


def enumerate_input_topologies(neuron_count: int, input_count: int) -> list[tuple[str, tuple[InputConfig, ...]]]:
    max_inputs = 1 if neuron_count == 1 else 2
    if input_count < 0 or input_count > max_inputs:
        raise ValueError("unsupported input count")
    target_sets = [combo for size in range(1, neuron_count + 1) for combo in combinations(range(neuron_count), size)]
    if input_count == 0:
        return [("i0", ())]
    configs = []
    if input_count == 1:
        candidates = ((a,) for a in target_sets)
    else:
        candidates = ((a, b) for a in target_sets for b in target_sets if a != b)
    for targets in candidates:
        ident = "_".join("i" + str(i + 1) + "-" + "".join(map(str, ts)) for i, ts in enumerate(targets))
        configs.append((ident, tuple(InputConfig(tuple(ts)) for ts in targets)))
    return configs
