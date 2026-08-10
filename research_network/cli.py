from __future__ import annotations

import argparse, json, os
from pathlib import Path
from .analysis import ResultAnalysis
from .persistence import JsonlStore
from .studies import Study
from .sweep import execute
from .web import serve

def main(argv=None):
    p=argparse.ArgumentParser(prog="tiny-lab",description="Tiny Recurrent Circuit Laboratory"); sub=p.add_subparsers(dest="command",required=True)
    w=sub.add_parser("web",help="start the local browser laboratory"); w.add_argument("--host",default="127.0.0.1");w.add_argument("--port",type=int,default=8765);w.add_argument("--results",default="results/results.jsonl")
    s=sub.add_parser("study",help="run or resume a JSON study");s.add_argument("file");s.add_argument("--results",default="results/results.jsonl");s.add_argument("--workers",type=int);s.add_argument("--no-multiprocessing",action="store_true");s.add_argument("--dry-run",action="store_true")
    i=sub.add_parser("inspect",help="print a deterministic replay trace");i.add_argument("identifier");i.add_argument("--results",default="results/results.jsonl")
    a=sub.add_parser("analyze",help="summarize saved results without simulation");a.add_argument("--results",default="results/results.jsonl")
    args=p.parse_args(argv)
    if args.command=="web": serve(args.host,args.port,args.results);return 0
    store=JsonlStore(args.results)
    if args.command=="study":
        configs=Study.load(args.file).configurations();done=len(store.completed_hashes()&{c.config_hash for c in configs});print(f"Requested: {len(configs):,}; already complete: {done:,}; pending: {len(configs)-done:,}")
        if args.dry_run:return 0
        progress_file=store.path.with_name("progress.json")
        def update(x):
            data=x.to_dict();data["status"]="running";tmp=progress_file.with_suffix(".tmp");tmp.write_text(json.dumps(data,default=str),encoding="utf-8");os.replace(tmp,progress_file)
            print(f"\rCompleted {x.completed}/{x.total-x.skipped}; failures {x.failures}",end="",flush=True)
        progress=execute(configs,store,args.workers,not args.no_multiprocessing,update);final=progress.to_dict();final["status"]="complete";progress_file.write_text(json.dumps(final,default=str),encoding="utf-8");print();print(json.dumps(final,indent=2,default=str));return 0
    if args.command=="inspect":
        from .config import ExperimentConfig
        from .simulator import simulate
        r=store.find(args.identifier)
        if not r: print("Run not found");return 1
        t=simulate(ExperimentConfig.from_dict(r["config"]));print(json.dumps([x.__dict__ for x in t.ticks],indent=2));return 0
    print(json.dumps(ResultAnalysis(store).group_counts(),indent=2));return 0
