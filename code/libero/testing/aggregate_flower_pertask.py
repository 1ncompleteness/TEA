#!/usr/bin/env python3
"""Aggregate per-task FLOWER eval JSONs into single results.json."""
import argparse
import json
import glob
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()

    per_task = {}
    total_ep = 0
    total_succ = 0
    for tf in sorted(glob.glob(os.path.join(args.run_dir, "t*_e*.json"))):
        with open(tf) as f:
            d = json.load(f)
        task = d.get("task_description", "unknown")
        ok = bool(d.get("success", False))
        if task not in per_task:
            per_task[task] = {"success": 0, "total": 0}
        per_task[task]["total"] += 1
        per_task[task]["success"] += int(ok)
        total_ep += 1
        total_succ += int(ok)

    for task, v in per_task.items():
        v["sr"] = v["success"] / v["total"] if v["total"] > 0 else 0.0

    sr = total_succ / total_ep if total_ep > 0 else 0.0
    out = {
        "total_sr": sr,
        "sr": sr,
        "total_successes": total_succ,
        "success_count": total_succ,
        "total_episodes": total_ep,
        "num_episodes": total_ep,
        "per_task": per_task,
    }
    with open(os.path.join(args.run_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"TOTAL: {total_succ}/{total_ep} = {sr:.1%}")
    for task, v in per_task.items():
        print(f"  {task[:50]}: {v['success']}/{v['total']} = {v['sr']:.1%}")


if __name__ == "__main__":
    main()
