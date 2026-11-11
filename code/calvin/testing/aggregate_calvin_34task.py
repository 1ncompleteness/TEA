#!/usr/bin/env python3
"""Aggregate CALVIN 34-task SR from mdt_evaluate.py stdout log.

mdt_evaluate.py prints per-task SR at line 93:
    f"{task}: {cnt_success[task]} / {total[task]} |  SR: {... * 100:.1f}%"

Also prints chain SR at line 77:
    f"{i}: {sr * 100:.1f}%"
And avg seq len at line 74:
    f"Average successful sequence length: {avg_seq_len}"

Usage:
  python aggregate_calvin_34task.py --base base.log --tea tea.log --output aggregate.json
"""
import argparse
import json
import re
import sys
from pathlib import Path


TASK_SR_RE = re.compile(r"^([a-z_]+):\s+(\d+)\s*/\s*(\d+)\s+\|\s+SR:\s+([\d.]+)%")
CHAIN_RE = re.compile(r"^(\d):\s+([\d.]+)%\s*$")
AVG_RE = re.compile(r"Average successful sequence length:\s+([\d.]+)")


def parse_log(path):
    tasks = {}
    chain = {}
    avg = None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                m = TASK_SR_RE.match(line)
                if m:
                    t, succ, tot, sr = m.groups()
                    tasks[t] = {"success": int(succ), "total": int(tot), "sr": float(sr) / 100.0}
                    continue
                m = CHAIN_RE.match(line)
                if m:
                    i, sr = m.groups()
                    chain[int(i)] = float(sr) / 100.0
                    continue
                m = AVG_RE.search(line)
                if m:
                    avg = float(m.group(1))
    except FileNotFoundError:
        print(f"MISSING: {path}", file=sys.stderr)
        return None
    return {"avg_seq_len": avg, "chain_sr": chain, "task_info": tasks}


def summarize(label, parsed):
    if parsed is None or not parsed["task_info"]:
        print(f"{label}: NO DATA")
        return {}
    srs = [t["sr"] for t in parsed["task_info"].values()]
    mean_sr = sum(srs) / len(srs)
    chain = parsed["chain_sr"]
    out = {
        "label": label,
        "avg_seq_len": parsed["avg_seq_len"],
        "chain_sr": {str(k): v for k, v in sorted(chain.items())},
        "n_tasks": len(srs),
        "mean_34_task_sr": mean_sr,
        "per_task_sr": {t: info["sr"] for t, info in parsed["task_info"].items()},
    }
    print(f"{label}:")
    print(f"  avg_seq_len={parsed['avg_seq_len']}")
    c = parsed["chain_sr"]
    print(f"  chain_SR: "
          + " ".join(f"{i}/5={c.get(i,0)*100:.1f}%" for i in range(1, 6)))
    print(f"  34-task mean SR: {mean_sr*100:.2f}% (n={len(srs)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="MDT baseline stdout log")
    ap.add_argument("--tea", required=True, help="MDT+TEA rerank stdout log")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base = parse_log(args.base)
    tea = parse_log(args.tea)

    out = {
        "mdt_baseline": summarize("MDT baseline", base),
        "mdt_tea": summarize("MDT+TEA K=4 rerank", tea),
    }

    if out["mdt_baseline"] and out["mdt_tea"]:
        delta = out["mdt_tea"]["mean_34_task_sr"] - out["mdt_baseline"]["mean_34_task_sr"]
        print(f"\ndelta 34-task SR: {delta*100:+.2f} pp")
        out["delta_mean_34_task_sr"] = delta

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
