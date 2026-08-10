from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import ExperimentConfig
from .persistence import JsonlStore
from .results import run_experiment
from .simulator import simulate

ROOT = Path(__file__).resolve().parent


def _page(title: str, section: str, body: str) -> bytes:
    nav = "".join(f'<a class="{("active" if section == p else "")}" href="{p}">{label}</a>' for p, label in
                  [("/", "Dashboard"), ("/simulator", "Simulator"), ("/progress", "Progress"), ("/results", "Results"), ("/charts", "Charts")])
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title} · Tiny Recurrent Lab</title><link rel="stylesheet" href="/static/style.css"></head>
<body><aside><div class="brand"><b>TR</b><span>Tiny Recurrent<br>Laboratory</span></div><nav>{nav}</nav><small>1–3 neuron research workspace</small></aside>
<main><header><div><p class="eyebrow">REPRODUCIBLE CIRCUIT DYNAMICS</p><h1>{title}</h1></div><span class="status"><i></i> Local laboratory</span></header>{body}</main>
<script src="/static/app.js"></script></body></html>'''.encode()


PAGES = {
    "/": ("Research dashboard", '''<section class="hero"><div><h2>Small circuits.<br><em>Visible dynamics.</em></h2><p>Explore how retention, recurrent structure, reset behavior and input geometry shape activity and memory.</p><div class="actions"><a class="button" href="/simulator">Open simulator</a><a href="/results">Browse results →</a></div></div><div class="orbit"><span>A</span><span>B</span><span>C</span></div></section><div id="dashboard" class="stats"></div><section class="cards"><article><label>INTERACTIVE</label><h3>Manual network laboratory</h3><p>Configure one to three neurons, trigger inputs, and watch potentials evolve tick by tick.</p><a href="/simulator">Launch laboratory →</a></article><article><label>BATCH RESEARCH</label><h3>Resumable parameter studies</h3><p>Every completed run is persisted immediately with deterministic identity.</p><a href="/progress">View progress →</a></article><article><label>ANALYSIS</label><h3>Phase maps and heatmaps</h3><p>Filter stored measurements without rerunning simulations.</p><a href="/charts">Open charts →</a></article></section>'''),
    "/simulator": ("Interactive simulator", '''<div class="lab"><section class="panel controls"><h2>Network configuration</h2><label>Neurons <select id="neurons"><option>1</option><option selected>2</option><option>3</option></select></label><label>Topology <select id="topology"><option value="cycle">Directed cycle</option><option value="full">All directed</option><option value="self">Self recurrent</option><option value="none">No recurrent edges</option></select></label><label>Individual edge weights <input id="edgeWeights" placeholder="optional, e.g. 0.5,-1"></label><div class="pair"><label>Shared recurrent weight <input id="rw" type="number" step="0.1" value="1"></label><label>Input count <select id="inputCount"><option>1</option><option selected>2</option></select></label></div><div class="pair"><label>Threshold <input id="threshold" type="number" step="0.1" value="1"></label><label>Retention <input id="retention" type="number" min="0" max="1" step="0.01" value="0.5"></label></div><label>Reset <select id="reset"><option>HARD_RESET</option><option>SUBTRACTIVE_RESET</option><option>FIXED_RESIDUAL_RESET</option><option>PERCENTAGE_RESET</option></select></label><div class="pair"><label>Reset value <input id="resetValue" type="number" step="0.1" value="0"></label><label>Reset fraction <input id="resetFraction" type="number" min="0" max="1" step="0.1" value="0.5"></label></div><div class="pair"><label>Input 1 targets <input id="targets" value="0" placeholder="0,1"></label><label>Input 1 weight <input id="iw" type="number" step="0.1" value="1"></label></div><div class="pair"><label>Input 2 targets <input id="targets2" value="1" placeholder="1,2"></label><label>Input 2 weight <input id="iw2" type="number" step="0.1" value="1"></label></div><label>Speed <input id="speed" type="range" min="1" max="20" value="5"></label><p id="validation" class="error"></p></section><section class="panel stage"><div class="toolbar"><button id="play">Play</button><button id="tick">Single tick</button><button id="resetSim">Reset</button><button class="pulse" id="trigger1">Trigger Input 1</button><button class="pulse" id="trigger2">Trigger Input 2</button><span>Tick <b id="tickNo">0</b></span></div><canvas id="network" width="760" height="300"></canvas><h3>Spike raster</h3><canvas id="raster" width="760" height="170"></canvas><h3>Membrane potential</h3><canvas id="potential" width="760" height="220"></canvas></section></div>'''),
    "/progress": ("Batch progress", '''<section class="panel"><div id="progress"><p>No study is active in this server process. Start one from the command line; completed JSONL results remain independent of this page.</p></div><div class="meter"><i style="width:0%"></i></div><div id="recent"></div></section>'''),
    "/results": ("Saved results", '''<section class="panel"><div class="filters"><input id="resultSearch" placeholder="Search run, topology, regime…"><select id="regime"><option value="">All regimes</option><option>DEAD</option><option>TRANSIENT</option><option>PERIODIC</option><option>TONIC</option><option>ACTIVE_APERIODIC</option><option>QUIESCENT_WITH_STATE</option></select><select id="resultN"><option value="">All sizes</option><option>1</option><option>2</option><option>3</option></select></div><div class="tablewrap"><table><thead><tr><th>Run</th><th>n</th><th>Topology</th><th>Reset</th><th>Retention</th><th>Weight</th><th>Regime</th><th>Lifetime</th><th>Rate</th><th>Period</th></tr></thead><tbody id="resultRows"></tbody></table></div><div class="pager"><button id="prev">←</button><span id="page"></span><button id="next">→</button></div></section>'''),
    "/charts": ("Analysis charts", '''<section class="panel"><div class="filters"><label>X <select id="chartX"><option value="retention">Retention</option><option value="weight">Recurrent weight</option><option value="threshold">Threshold</option></select></label><label>Y <select id="chartY"><option value="weight">Recurrent weight</option><option value="retention">Retention</option><option value="threshold">Threshold</option></select></label><label>Metric <select id="chartMetric"><option value="regime">Regime phase map</option><option value="activity_lifetime">Activity lifetime</option><option value="spike_rate_network">Spike rate</option><option value="period">Detected period</option></select></label><button id="drawChart">Draw</button></div><canvas id="chart" width="1000" height="560"></canvas><p class="muted">Charts are computed only from saved JSONL measurements. The analysis view never runs simulations.</p></section>''')
}


class AppHandler(BaseHTTPRequestHandler):
    store: JsonlStore
    progress: dict = {}

    def _json(self, data, status=200):
        payload = json.dumps(data, allow_nan=False).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
    def do_GET(self):
        parsed = urlparse(self.path); path = parsed.path
        if path.startswith("/static/"):
            file = ROOT / "static" / Path(path).name
            if file.exists():
                data = file.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(file)[0] or "application/octet-stream"); self.end_headers(); self.wfile.write(data); return
        if path == "/api/dashboard":
            rows = list(self.store.records()); self._json({"results": len(rows), "studies": len({r.get("study_id") for r in rows}), "regimes": len({r.get("classification", {}).get("primary_regime") for r in rows})}); return
        if path == "/api/results":
            q = parse_qs(parsed.query); page = max(1, int(q.get("page", [1])[0])); size = min(100, int(q.get("size", [25])[0])); rows = list(self.store.records())
            search = q.get("q", [""])[0].lower(); regime = q.get("regime", [""])[0]; n = q.get("n", [""])[0]
            if search: rows = [r for r in rows if search in json.dumps(r).lower()]
            if regime: rows = [r for r in rows if r["classification"]["primary_regime"] == regime]
            if n: rows = [r for r in rows if r["config"]["neuron_count"] == int(n)]
            rows.reverse(); start = (page - 1) * size; self._json({"total": len(rows), "page": page, "rows": rows[start:start+size]}); return
        if path == "/api/progress":
            progress_file = self.store.path.with_name("progress.json")
            try: self._json(json.loads(progress_file.read_text(encoding="utf-8")))
            except (FileNotFoundError, json.JSONDecodeError) as exc: self._json({"status": "unavailable", "progress_path": str(progress_file), "message": str(exc)})
            return
        if path.startswith("/api/run/"):
            record = self.store.find(path.split("/")[-1]); self._json(record or {"error": "not found"}, 200 if record else 404); return
        if path.startswith("/run/"):
            rid = path.split("/")[-1]; record = self.store.find(rid)
            if not record: self.send_error(404); return
            body = f'<section class="panel"><div class="runhead"><div><span class="badge">{record["classification"]["primary_regime"]}</span><h2>{record["run_id"]}</h2><code>{record["config_hash"]}</code></div><button id="inspectRun" data-id="{rid}">Inspect / replay</button></div><div id="runDetail" data-record=\'{json.dumps(record)}\'></div></section>'
            data = _page("Run detail", path, body); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(data); return
        if path in PAGES:
            title, body = PAGES[path]; data = _page(title, path, body); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(data); return
        self.send_error(404)
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0)); raw = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/inspect":
            record = self.store.find(raw.get("id", ""));
            if not record: self._json({"error": "not found"}, 404); return
            config = ExperimentConfig.from_dict(record["config"]); trace = simulate(config)
            self._json({"config": config.to_dict(), "ticks": [{"tick": t.tick, "inputs": t.input_spikes, "spikes": t.spikes, "potentials": t.potentials} for t in trace.ticks]}); return
        self._json({"error": "not found"}, 404)
    def log_message(self, fmt, *args): pass


def serve(host="127.0.0.1", port=8765, results="results/results.jsonl"):
    handler = type("ConfiguredHandler", (AppHandler,), {"store": JsonlStore(results)})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Tiny Recurrent Laboratory: http://{host}:{port}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
