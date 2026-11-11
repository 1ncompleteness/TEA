#!/usr/bin/env python3
"""Convert Habitat demos.npz to the chunked-buffer format and run the
standard trust-calibration diagnostics (ECE / dQ / delta / eps) on the
trained Habitat Q-ensemble.

Usage:
    python habitat_trust_calib.py --run_dir results/tea_habitat_q_full_<ts> \
        --out_json results/trust_calib_habitat.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--demos_npz", default=None,
                   help="comma-separated buffers; default run_dir/demos.npz")
    p.add_argument("--out_json", default=None)
    p.add_argument("--label", default="Habitat 3.0 Social-Nav")
    args = p.parse_args()

    if args.demos_npz:
        parts = [np.load(x) for x in args.demos_npz.split(",")]
        demos = {k: np.concatenate([p_[k] for p_ in parts], axis=0)
                 for k in parts[0].files}
    else:
        demos = np.load(os.path.join(args.run_dir, "demos.npz"))
    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    n = demos["states"].shape[0]
    chunk_npz = os.path.join(args.run_dir, "chunk1.npz")
    np.savez(
        chunk_npz,
        obs=demos["states"].astype(np.float32),
        action_chunks=demos["actions"].astype(np.float32)[:, None, :],
        next_obs=demos["next_states"].astype(np.float32),
        reward_chunks=demos["rewards"].astype(np.float32)[:, None],
        done=demos["dones"].astype(np.float32),
    )
    print(f"converted {n} transitions -> {chunk_npz}", flush=True)

    out_json = args.out_json or os.path.join(args.run_dir,
                                             "trust_calib_habitat.json")
    cmd = [
        sys.executable, next(p for p in (
            os.path.join(HERE, "compute_trust_calibration.py"),
            os.path.join(HERE, os.pardir, os.pardir, "trustworthy_calibration",
                         "compute_trust_calibration.py"),
        ) if os.path.exists(p)),
        "--ckpt", os.path.join(args.run_dir, "tea_q.pt"),
        "--chunk_npz", chunk_npz,
        "--state_dim", str(cfg["state_dim"]),
        "--action_dim", str(cfg["action_dim"]),
        "--chunk_size", "1",
        "--num_actions", "4",
        "--use_ensemble", "--n_ensemble", str(cfg.get("n_ensemble", 4)),
        "--hidden_dim", "256",
        "--label", args.label,
        "--out_json", out_json,
    ]
    env = dict(os.environ)
    train_dir = os.path.join(HERE, os.pardir, "training")
    env["PYTHONPATH"] = train_dir + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)
    print(json.dumps(json.load(open(out_json)), indent=1))


if __name__ == "__main__":
    main()
