#!/usr/bin/env python3
"""Cache CALVIN per-frame demos into chunk{H}.npz for trust-calibration eval.

Builds (s_t, a_{t:t+H}, next_obs[t+H], reward_chunks[t:t+H], done) tuples
directly from per-frame obs.npy + actions.npy. Uses task annotations from
auto_lang_ann.npy if available to set rewards at task-segment ends; otherwise
falls back to zero rewards (still valid for ECE / ΔQ / Bellman-residual
trust-calibration metrics).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', required=True,
                   help='Dir containing obs.npy, actions.npy (calvin_cache layout)')
    p.add_argument('--demos_root', default=None,
                   help='Optional CALVIN demos root with task_D_D/training/lang_annotations/auto_lang_ann.npy + episode_*.npz')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--chunk', type=int, default=10)
    p.add_argument('--max_samples', type=int, default=200000,
                   help='Cap held-out cache size to bound disk and eval time')
    args = p.parse_args()

    obs = np.load(os.path.join(args.cache_dir, 'obs.npy'))
    actions = np.load(os.path.join(args.cache_dir, 'actions.npy'))
    n = obs.shape[0]
    print(f"loaded obs={obs.shape} actions={actions.shape}")

    H = args.chunk
    reward_per_frame = np.zeros(n, dtype=np.float32)

    if args.demos_root:
        ann_path = os.path.join(args.demos_root, 'task_D_D', 'training',
                                'lang_annotations', 'auto_lang_ann.npy')
        demo_path = os.path.join(args.demos_root, 'task_D_D', 'training')
        if os.path.exists(ann_path):
            ann = np.load(ann_path, allow_pickle=True).item()
            lang_indx = ann['info']['indx']
            files = sorted(glob.glob(os.path.join(demo_path, 'episode_*.npz')))
            frame_ids = [int(os.path.basename(f).replace('episode_', '').replace('.npz', ''))
                         for f in files[:n]]
            fid_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
            mapped = 0
            for sf, ef in lang_indx:
                ei = fid_to_idx.get(ef, -1)
                if ei >= 0:
                    reward_per_frame[ei] = 1.0
                    mapped += 1
            print(f"  mapped {mapped}/{len(lang_indx)} reward frames")
        else:
            print(f"  task annotations not found at {ann_path}; using zero rewards")
    else:
        print("  no demos_root; using zero rewards (ΔQ, ECE, δ still meaningful)")

    valid = np.arange(n - H - 1, dtype=np.int64)
    if len(valid) > args.max_samples:
        rng = np.random.default_rng(42)
        valid = rng.choice(valid, size=args.max_samples, replace=False)
        valid.sort()
    print(f"  {len(valid)} chunk-start indices")

    obs_out = obs[valid].astype(np.float32)
    next_obs_out = obs[valid + H].astype(np.float32)
    a_chunks = np.stack([actions[t:t + H] for t in valid]).astype(np.float32)
    r_chunks = np.stack([reward_per_frame[t:t + H] for t in valid]).astype(np.float32)
    masks = np.ones_like(r_chunks, dtype=np.float32)
    done = (r_chunks.sum(axis=1) > 0.5).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'chunk{H}.npz')
    np.savez(out_path,
             obs=obs_out, action_chunks=a_chunks, masks=masks,
             next_obs=next_obs_out, reward_chunks=r_chunks, done=done,
             episode_ends=np.array([], dtype=np.int64))
    print(f"saved {len(valid)} samples to {out_path}")


if __name__ == '__main__':
    main()
