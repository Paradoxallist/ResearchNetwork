from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from pathlib import Path

from .config import Edge, ExperimentConfig, InputConfig, InputProtocol, ProtocolType, ResetType
from .topology import enumerate_input_topologies, enumerate_topologies


LARGE_SWEEP = 100_000


def _values(value):
    if isinstance(value, dict) and {"start", "stop", "step"} <= value.keys():
        current, stop, step = map(lambda x: Decimal(str(x)), (value["start"], value["stop"], value["step"]))
        out = []
        while current <= stop: out.append(float(current)); current += step
        return out
    return value if isinstance(value, list) else [value]


@dataclass(frozen=True)
class Study:
    raw: dict

    @property
    def study_id(self): return self.raw["study_id"]

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def configurations(self) -> list[ExperimentConfig]:
        r = self.raw; configs = []
        dimensions = product(_values(r.get("neurons", [2])), _values(r.get("thresholds", [1.0])),
                             _values(r.get("retentions", [0.0])), _values(r.get("reset_types", ["HARD_RESET"])),
                             _values(r.get("recurrent_weights", [1.0])), _values(r.get("input_weights", [1.0])))
        for n, threshold, retention, reset, rw, iw in dimensions:
            tops = enumerate_topologies(n, r.get("allow_self_connections", False), r.get("include_empty_topology", True)) if r.get("topologies", "all") == "all" else []
            input_count = r.get("input_count", 1)
            intops = enumerate_input_topologies(n, input_count) if r.get("input_configurations", "all_valid") == "all_valid" else []
            protocol = InputProtocol(ProtocolType(r.get("input_protocol", "SINGLE_IMPULSE")), r.get("input_start_tick", 0))
            for _, edges in tops:
                weighted = tuple(Edge(e.source, e.target, rw) for e in edges)
                for _, inputs in intops:
                    inputs = tuple(InputConfig(x.targets, iw, protocol) for x in inputs)
                    configs.append(ExperimentConfig(self.study_id, n, weighted, inputs, threshold, retention, ResetType(reset),
                                                    r.get("reset_value", 0.0), r.get("reset_fraction", 0.0), r.get("simulation_ticks", 200)))
        unique = {c.config_hash: c for c in configs}
        configs = list(unique.values())
        if len(configs) > LARGE_SWEEP and not r.get("allow_large_sweep", False):
            raise ValueError(f"study requests {len(configs):,} unique runs; set allow_large_sweep=true after review")
        return configs
