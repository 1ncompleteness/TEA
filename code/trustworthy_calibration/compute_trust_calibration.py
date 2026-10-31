#!/usr/bin/env python3
"""Compute TEA trust-calibration metrics per task: ECE, Cov, δ, ε.

Loads a Q-network checkpoint (ActionChunkQNetwork) plus a chunked demo cache
(`chunk{H}.npz`) and computes:
  ECE = expected calibration error of Q(s, a^demo) vs Monte-Carlo return-to-go
  Cov = fraction of held-out random-action chunks for which
        Q(s, a^rnd) <= max(Q(s, a^demo), V^MC)
  δ   = 95th percentile of |Q(s,a) - r - γ Q_target(s', a^demo)| (Bellman residual proxy)
  ε   = 2δ + τ log|A_i| with τ = 2.0

Usage:
    python compute_trust_calibration.py \
        --ckpt /workspace/results/tea_libero_spatial_s42_<ts>/tea_q.pt \
        --chunk_npz /workspace/data/libero_spatial_chunk8/chunk8.npz \
        --state_dim 15 --action_dim 7 --chunk_size 8 \
        --num_actions 1 --tau 2.0 --label "LIBERO Spatial"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# pa_rl ships in code/habitat/training/; ActionChunkQNetwork in code/calvin/training/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "habitat", "training"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "calvin", "training"))


def load_action_chunk_q(ckpt_path: str, state_dim: int, action_dim: int,
                        chunk_size: int, hidden_dim: int = 512,
                        num_modes: int = 3, device: str = "cuda"):
    from train_tea_mdt import ActionChunkQNetwork
    q = ActionChunkQNetwork(
        state_dim=state_dim, action_dim=action_dim,
        chunk_size=chunk_size, hidden_dim=hidden_dim, num_modes=num_modes,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get('q_net', ck.get('state_dict', ck))
    miss, unex = q.load_state_dict(sd, strict=False)
    if miss or unex:
        print(f"  load_state_dict missing={len(miss)} unexpected={len(unex)}")
    q.eval()
    return q


def load_q_ensemble(ckpt_path: str, state_dim: int, action_dim: int,
                    chunk_size: int, hidden_dim: int = 512,
                    num_modes: int = 3, n_ensemble: int = 4, device: str = "cuda"):
    from pa_rl import QEnsemble
    q = QEnsemble(
        n_ensemble=n_ensemble, state_dim=state_dim, action_dim=action_dim,
        chunk_size=chunk_size, hidden_dim=hidden_dim, num_modes=num_modes,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get('q_ensemble', ck.get('state_dict', ck))
    miss, unex = q.load_state_dict(sd, strict=False)
    if miss or unex:
        print(f"  load_state_dict missing={len(miss)} unexpected={len(unex)}")
    q.eval()
    return q


def split_holdout(N: int, holdout_frac: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_h = int(N * holdout_frac)
    return idx[n_h:], idx[:n_h]  # train_idx, test_idx


def compute_ece(q_demo: np.ndarray, v_mc: np.ndarray, n_bins: int = 10) -> float:
    """ECE over quantile bins of |q_demo - v_mc| -> mean abs error per bin -> weighted sum."""
    err = np.abs(q_demo - v_mc)
    qbins = np.quantile(q_demo, np.linspace(0, 1, n_bins + 1))
    qbins[0] = -np.inf
    qbins[-1] = np.inf
    ece = 0.0
    N = len(q_demo)
    for b in range(n_bins):
        mask = (q_demo >= qbins[b]) & (q_demo < qbins[b + 1])
        if mask.sum() == 0:
            continue
        w = mask.sum() / N
        ece += w * abs(q_demo[mask].mean() - v_mc[mask].mean())
    return float(ece)


def compute_coverage(q_demo: np.ndarray, q_rand: np.ndarray, v_mc: np.ndarray) -> float:
    """Fraction of (s) for which Q(s,a_rnd) <= max(Q(s,a_demo), V_MC)."""
    upper = np.maximum(q_demo, v_mc)
    return float((q_rand <= upper).mean())


def compute_delta(bellman_residuals: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(np.abs(bellman_residuals), percentile))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--chunk_npz', required=True)
    p.add_argument('--state_dim', type=int, required=True)
    p.add_argument('--action_dim', type=int, required=True)
    p.add_argument('--chunk_size', type=int, required=True)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_modes', type=int, default=3)
    p.add_argument('--num_actions', type=int, default=1,
                   help='|A| for log|A| in ε; 1 for continuous (gives 0 contribution)')
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--label', required=True)
    p.add_argument('--use_ensemble', action='store_true',
                   help='Use QEnsemble loader (PARLQNetwork)')
    p.add_argument('--n_ensemble', type=int, default=4)
    p.add_argument('--n_rand', type=int, default=5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_json', default=None)
    args = p.parse_args()

    print(f"=== Trust calibration: {args.label} ===", flush=True)
    print(f"  ckpt: {args.ckpt}", flush=True)
    print(f"  data: {args.chunk_npz}", flush=True)

    if args.use_ensemble:
        q = load_q_ensemble(
            args.ckpt, state_dim=args.state_dim, action_dim=args.action_dim,
            chunk_size=args.chunk_size, hidden_dim=args.hidden_dim,
            num_modes=args.num_modes, n_ensemble=args.n_ensemble,
            device=args.device,
        )
    else:
        q = load_action_chunk_q(
            args.ckpt, state_dim=args.state_dim, action_dim=args.action_dim,
            chunk_size=args.chunk_size, hidden_dim=args.hidden_dim,
            num_modes=args.num_modes, device=args.device,
        )

    d = np.load(args.chunk_npz)
    obs = d['obs']
    ach_full = d['action_chunks']
    nobs = d['next_obs']
    rc = d['reward_chunks']
    done = d['done']
    N = obs.shape[0]
    data_chunk = ach_full.shape[1]
    if data_chunk != args.chunk_size:
        print(f"  data chunk={data_chunk}, Q-net chunk={args.chunk_size}; "
              f"slicing first {args.chunk_size} action steps", flush=True)
        ach = ach_full[:, :args.chunk_size, :].copy()
    else:
        ach = ach_full
    print(f"  N={N} state={obs.shape[1]} chunk={ach.shape[1]}x{ach.shape[2]}", flush=True)

    train_idx, test_idx = split_holdout(N, holdout_frac=0.1, seed=args.seed)
    NT = len(test_idx)

    s = torch.from_numpy(obs[test_idx]).float().to(args.device)
    a = torch.from_numpy(ach[test_idx]).float().to(args.device)
    s2 = torch.from_numpy(nobs[test_idx]).float().to(args.device)
    r = torch.from_numpy(rc[test_idx]).float().to(args.device).sum(dim=-1, keepdim=True)
    dn = torch.from_numpy(done[test_idx]).float().to(args.device).unsqueeze(-1)

    with torch.no_grad():
        if args.use_ensemble:
            q_demo = q(s, a, reduce='mean').cpu().numpy().squeeze()
            q_next = q(s2, a, reduce='mean').cpu().numpy().squeeze()
        else:
            q_demo = q(s, a).cpu().numpy().squeeze()
            q_next = q(s2, a).cpu().numpy().squeeze()

    v_mc = np.zeros(NT, dtype=np.float32)
    for i in range(NT):
        v_mc[i] = (args.gamma ** (args.chunk_size - 1)) * float(r[i].item())

    bellman = q_demo - (r.cpu().numpy().squeeze() + (args.gamma ** args.chunk_size) *
                        (1.0 - dn.cpu().numpy().squeeze()) * q_next)

    rng = np.random.default_rng(args.seed)
    q_rand_list = []
    for _ in range(args.n_rand):
        a_rnd = torch.from_numpy(
            rng.uniform(-1.0, 1.0, size=ach[test_idx].shape).astype(np.float32)
        ).to(args.device)
        with torch.no_grad():
            if args.use_ensemble:
                q_r = q(s, a_rnd, reduce='mean').cpu().numpy().squeeze()
            else:
                q_r = q(s, a_rnd).cpu().numpy().squeeze()
        q_rand_list.append(q_r)
    q_rand = np.stack(q_rand_list).mean(axis=0)

    ece = compute_ece(q_demo, v_mc, n_bins=10)
    cov = compute_coverage(q_demo, q_rand, v_mc)
    delta = compute_delta(bellman, percentile=95.0)
    log_A = np.log(max(args.num_actions, 2))
    eps = 2.0 * delta + args.tau * log_A
    q_margin = float((q_demo - q_rand).mean())

    print(f"  ECE      = {ece:.4f}")
    print(f"  Cov      = {cov:.4f}")
    print(f"  ΔQ-mar   = {q_margin:.4f}")
    print(f"  δ        = {delta:.4f}")
    print(f"  ε        = {eps:.4f}  (τ={args.tau}, |A|={args.num_actions})")

    out = dict(label=args.label, ece=ece, cov=cov, q_margin=q_margin,
               delta=delta, eps=eps, tau=args.tau,
               num_actions=args.num_actions, N_holdout=NT)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.out_json}", flush=True)

    return out


if __name__ == '__main__':
    main()
