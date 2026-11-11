#!/usr/bin/env python3
"""Q-chunking (QC) critic training — Li, Zhou, Levine ICML 2025 (arxiv:2507.07969).

Trains PARAFAC Q(s, a_{t:t+H}) with Cal-QL penalty + n-step chunked Bellman
backup. Works uniformly across diffusion (CALVIN, H=10), AR VLA (LIBERO,
H=5), and Gaussian (Habitat/LT, H=1) policy regimes by fixing H.

Unified method: same trainer, same loss, only `chunk_size` changes per env.

Not used for any result reported in the paper; the paper's LIBERO critics come
from train_tea_libero.py (see README).

Usage:
    python train_q_chunked.py \\
        --data /workspace/algorithm/data/libero_object_chunk5/chunk5.npz \\
        --chunk 5 --state_dim 15 --action_dim 7 \\
        --steps 20000 --batch 256 --n_ensemble 4 \\
        --gamma 0.99 --cql_alpha 0.01 \\
        --output_dir /workspace/algorithm/results/qchunk_libero_object_<ts>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='chunk{H}.npz path')
    p.add_argument('--chunk', type=int, required=True)
    p.add_argument('--state_dim', type=int, default=15)
    p.add_argument('--action_dim', type=int, default=7)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_modes', type=int, default=3)
    p.add_argument('--n_ensemble', type=int, default=4)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--critic_lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--cql_alpha', type=float, default=0.01)
    p.add_argument('--tau_target', type=float, default=0.005)
    p.add_argument('--warm_init', default=None,
                   help='Optional path to prior chunk=1 tea_q.pt for state/action encoder warm-start')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output_dir', default=None)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'/workspace/algorithm/results/qchunk_s{args.seed}_{ts}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output: {args.output_dir}', flush=True)

    # --- Data ---
    d = np.load(args.data)
    obs = d['obs']                     # [N, S]
    ach = d['action_chunks']           # [N, H, A]
    mk = d['masks']                    # [N, H]
    nobs = d['next_obs']               # [N, S]
    rc = d['reward_chunks']            # [N, H]
    done = d['done']                   # [N]
    N = obs.shape[0]
    assert ach.shape[1] == args.chunk, f"chunk mismatch: data={ach.shape[1]} arg={args.chunk}"
    print(f'N={N} samples, chunk={args.chunk}, state={args.state_dim}, '
          f'action={args.action_dim}', flush=True)

    # Precomputed n-step discounted reward (sum_{k=0..H-1} gamma^k r_{t+k})
    disc = args.gamma ** np.arange(args.chunk, dtype=np.float32)
    r_nstep = (rc * disc[None, :]).sum(axis=1, dtype=np.float32)  # [N]

    # --- Tensors on device ---
    dev = args.device
    s_all = torch.from_numpy(obs).to(dev)
    a_all = torch.from_numpy(ach).to(dev)             # [N, H, A]
    r_all = torch.from_numpy(r_nstep).to(dev)          # [N]
    s2_all = torch.from_numpy(nobs).to(dev)
    d_all = torch.from_numpy(done).to(dev)

    # --- Q ensemble ---
    from pa_rl import QEnsemble, calql_loss, soft_update_
    q_kwargs = dict(state_dim=args.state_dim, action_dim=args.action_dim,
                    chunk_size=args.chunk, hidden_dim=args.hidden_dim,
                    num_modes=args.num_modes)
    q = QEnsemble(n_ensemble=args.n_ensemble, **q_kwargs).to(dev)
    target_q = QEnsemble(n_ensemble=args.n_ensemble, **q_kwargs).to(dev)
    target_q.load_state_dict(q.state_dict())
    for pp in target_q.parameters():
        pp.requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=args.critic_lr)

    # Optional warm start: state encoder from prior chunk=1 Q shares arch
    if args.warm_init and os.path.exists(args.warm_init):
        ck = torch.load(args.warm_init, map_location=dev, weights_only=False)
        sd = ck.get('q_net', ck)
        bcast = {}
        for m_idx in range(args.n_ensemble):
            for k, v in sd.items():
                # Load only state_encoder weights; action encoder has different
                # shape due to chunk_size difference.
                if k.startswith('state_encoder') or k.startswith('mode_encoders'):
                    bcast[f'nets.{m_idx}.{k}'] = v
        miss, unex = q.load_state_dict(bcast, strict=False)
        print(f'Warm-init state+mode encoders: missing={len(miss)} unex={len(unex)}',
              flush=True)
        target_q.load_state_dict(q.state_dict())

    total_p = sum(p.numel() for p in q.parameters())
    print(f'Q-ensemble params: {total_p:,}', flush=True)

    # --- Training ---
    log = []
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch,), device=dev)
        batch = {
            'state_t': s_all[idx],                               # [B, S]
            'action_t': a_all[idx],                              # [B, H, A]
            'reward_t': r_all[idx].unsqueeze(-1),                # [B, 1]
            'state_tp1': s2_all[idx],                            # [B, S]
            'action_tp1': a_all[idx],                            # [B, H, A]  (proxy: same chunk from demo)
            'done': d_all[idx].unsqueeze(-1),                    # [B, 1]
        }
        # gamma^H for chunked Bellman backup
        loss, metrics = calql_loss(q, target_q, batch,
                                   gamma=args.gamma ** args.chunk,
                                   cql_alpha=args.cql_alpha,
                                   use_calql=True)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        opt.step()
        soft_update_(target_q, q, tau=args.tau_target)

        if (step + 1) % 500 == 0:
            with torch.no_grad():
                qd = q(s_all[:512], a_all[:512], reduce='mean').mean().item()
                ar = torch.empty_like(a_all[:512]).uniform_(-1, 1)
                qr = q(s_all[:512], ar, reduce='mean').mean().item()
            log.append({'step': step + 1, 'loss': float(loss),
                        'q_demo': qd, 'q_rand': qr, 'diff': qd - qr,
                        **metrics})
            print(f'  [{step+1}/{args.steps}] loss={loss:.4f} '
                  f'q_d={qd:.3f} q_r={qr:.3f} diff={qd-qr:.3f} '
                  f'cql={metrics.get("cql_pen",0):.4f}',
                  flush=True)

    print(f'Train done in {time.time()-t0:.1f}s', flush=True)

    # --- Save ---
    # Ensemble full
    torch.save({
        'q_ensemble': q.state_dict(),
        'state_dim': args.state_dim, 'action_dim': args.action_dim,
        'chunk_size': args.chunk, 'n_ensemble': args.n_ensemble,
        'hidden_dim': args.hidden_dim, 'num_modes': args.num_modes,
    }, os.path.join(args.output_dir, 'tea_q_ensemble.pt'))
    # Single-net compat (net_0, matches ActionChunkQNetwork loader)
    sd0 = {k.replace('nets.0.', ''): v for k, v in q.state_dict().items()
           if k.startswith('nets.0.')}
    torch.save({
        'q_net': sd0,
        'state_dim': args.state_dim, 'action_dim': args.action_dim,
        'chunk_size': args.chunk,
    }, os.path.join(args.output_dir, 'tea_q.pt'))
    with open(os.path.join(args.output_dir, 'learning_curve.json'), 'w') as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print('Saved.', flush=True)


if __name__ == '__main__':
    main()
