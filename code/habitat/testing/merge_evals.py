#!/usr/bin/env python3
"""Merge per-seed eval aggregate.json files into one summary."""
import glob
import json
import sys

import numpy as np


def summarize(pattern):
    eps = []
    for path in sorted(glob.glob(pattern + "/aggregate.json")):
        eps.extend(json.load(open(path))["per_episode"])
    if not eps:
        return None
    S = np.array([e["S"] for e in eps])
    F = np.array([e["raw_stats"][
        "found_human_rate_after_encounter_over_epi"] for e in eps])
    CR = np.array([e["collided"] for e in eps])
    TE = np.array([e["raw_stats"]["frist_ecnounter_steps"] for e in eps])
    n = len(eps)
    return dict(
        n=n,
        S=round(float(S.mean()), 4), S_se=round(float(S.std()/np.sqrt(n)), 4),
        F=round(float(F.mean()), 4), F_se=round(float(F.std()/np.sqrt(n)), 4),
        CR=round(float(CR.mean()), 4),
        CR_se=round(float(CR.std()/np.sqrt(n)), 4),
        T_enc=round(float(TE.mean()), 1),
        T_enc_found=round(float(TE[S > 0].mean()), 1) if S.sum() else None,
    )


if __name__ == "__main__":
    out = {}
    if len(sys.argv) < 5 or "-h" in sys.argv or "--help" in sys.argv:
        print("usage: merge_evals.py K1_GLOB TEAK4_GLOB BLINDK4_GLOB OUT_JSON")
        sys.exit(0)
    for name, pattern in [("K1", sys.argv[1]), ("teaK4", sys.argv[2]),
                          ("blindK4", sys.argv[3])]:
        out[name] = summarize(pattern)
    print(json.dumps(out, indent=1))
    with open(sys.argv[4], "w") as f:
        json.dump(out, f, indent=1)
