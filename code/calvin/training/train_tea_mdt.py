#!/usr/bin/env python3
"""
TEA critic training for CALVIN D->D (frozen MDT-V prior, checkpoint mdtv-1-d).

Trains the multilinear Cal-QL critic Q(s, a) (paper Eq. 1, Stage 3(A)) offline
on cached demonstration transitions: s is the 39-D state (15-D proprioception
plus 24-D scene state) and a is a flattened 10-step action chunk (C=10). Each
successful language-annotated segment ends with reward 1 (V_ref = gamma^(T-t));
the loss is TD + 0.5 * Cal-QL penalty with one uniform-random chunk per state.
At inference (Stage 3(B)), MDT-V samples K=4 chunks and the hard Q-argmax picks
one; the CALVIN +TEA evaluation hook lives in a local MDT checkout (see README).
The --mdt_ckpt argument is required by the CLI but not used for training.
"""

import argparse, os, sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime
import time


class ActionChunkQNetwork(nn.Module):
    """Q(state_feat, action_chunk) → scalar return prediction.

    state_feat: [B, state_dim] compact state; the paper's CALVIN critic uses
        39-D (15-D proprioception + 24-D scene state), not image features
    action_chunk: [10, 7] continuous actions (chunk_size × action_dim)
    hidden_dim is the rank R; num_modes is the number M of linear-tanh factor
    projections multiplied in the Hadamard product (M=3 in the paper)
    """

    def __init__(self, state_dim=384, action_dim=7, chunk_size=10,
                 hidden_dim=512, num_modes=3):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(chunk_size * action_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        # PARAFAC-style multilinear: state * action factors → Q
        self.rank = hidden_dim
        self.num_modes = num_modes

        self.mode_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, self.rank), nn.Tanh())
            for _ in range(num_modes)
        ])
        self.q_head = nn.Sequential(
            nn.Linear(self.rank, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state_feat, action_chunk):
        """
        state_feat: [B, state_dim]
        action_chunk: [B, chunk_size, action_dim]
        Returns: [B, 1] Q-value
        """
        B = state_feat.shape[0]
        a_flat = action_chunk.reshape(B, -1)
        a_enc = self.action_encoder(a_flat)
        s_enc = self.state_encoder(state_feat)

        # Multilinear interaction
        fused = s_enc * a_enc  # [B, hidden_dim]
        # PARAFAC: product of mode encodings
        factors = [m(fused) for m in self.mode_encoders]
        joint = factors[0]
        for f in factors[1:]:
            joint = joint * f
        return self.q_head(joint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdt_ckpt", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Dir with obs.npy, actions.npy (from cache_hulc_data)")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005, help="Target update")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"tea_mdt_s{args.seed}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("TEA RL Q-network training for MDT rerank")
    print("=" * 60)

    # Load demo data + task annotations
    print("Loading demo data + task annotations...")
    obs = np.load(os.path.join(args.cache_dir, 'obs.npy'))  # [N, 39]
    actions = np.load(os.path.join(args.cache_dir, 'actions.npy'))  # [N, 7]
    n = len(obs)
    print(f"  {n} frames")

    # Load language annotations for task segment boundaries
    ann_path = os.path.join(
        os.path.dirname(args.cache_dir).replace('hulc_cache', ''),
        'task_D_D/training/lang_annotations/auto_lang_ann.npy')
    if not os.path.exists(ann_path):
        # Try direct path
        ann_path = '/workspace/calvin_dataset/task_D_D/training/lang_annotations/auto_lang_ann.npy'
    ann = np.load(ann_path, allow_pickle=True).item()
    lang_indx = ann['info']['indx']

    # Map frame IDs to array indices (frames cached in order)
    import glob
    demo_path = '/workspace/calvin_dataset/task_D_D/training'
    files = sorted(glob.glob(os.path.join(demo_path, 'episode_*.npz')))
    frame_ids = [int(os.path.basename(f).replace('episode_', '').replace('.npz', '')) for f in files[:n]]
    fid_to_idx = {fid: i for i, fid in enumerate(frame_ids)}

    # Build per-frame reward: r=+1 at segment end, 0 elsewhere
    # Also mark "task-frame-distance" for discounted return computation
    reward_per_frame = np.zeros(n, dtype=np.float32)
    task_end_for_frame = -np.ones(n, dtype=np.int64)  # -1 = no task

    segments_mapped = 0
    for sf, ef in lang_indx:
        si = fid_to_idx.get(sf, -1)
        ei = fid_to_idx.get(ef, -1)
        if si < 0 or ei < 0 or ei <= si:
            continue
        reward_per_frame[ei] = 1.0
        task_end_for_frame[si:ei + 1] = ei
        segments_mapped += 1

    print(f"  {segments_mapped}/{len(lang_indx)} task segments mapped. "
          f"Reward frames: {(reward_per_frame > 0).sum()}")

    C = args.chunk_size
    state_dim = 39
    action_dim = 7

    q_net = ActionChunkQNetwork(
        state_dim=state_dim, action_dim=action_dim, chunk_size=C
    ).to(args.device)
    target_q = ActionChunkQNetwork(
        state_dim=state_dim, action_dim=action_dim, chunk_size=C
    ).to(args.device)
    target_q.load_state_dict(q_net.state_dict())

    opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    total_p = sum(p.numel() for p in q_net.parameters())
    print(f"Q-network: {total_p:,} params")

    curve = {'step': [], 'loss': [], 'q_mean': []}
    t0 = time.time()

    # Only sample indices from within task segments (has real reward signal)
    valid_idx = np.where(task_end_for_frame > 0)[0]
    valid_idx = valid_idx[valid_idx < n - C - 1]
    print(f"  {len(valid_idx)} valid training indices (within task segments)")

    for step in range(args.steps):
        # Sample batch of (s_t, a_{t:t+C}, r, s_{t+C})
        idx = valid_idx[np.random.randint(0, len(valid_idx), args.batch_size)]
        s_t = torch.tensor(obs[idx], dtype=torch.float32, device=args.device)
        s_next = torch.tensor(obs[idx + C], dtype=torch.float32, device=args.device)
        a_chunk = np.stack([actions[i:i + C] for i in idx])
        a_chunk = torch.tensor(a_chunk, dtype=torch.float32, device=args.device)

        # Reward = sum of per-frame rewards within the chunk (usually 0 or 1)
        r_chunk = np.array([
            reward_per_frame[i:i + C].sum() for i in idx
        ], dtype=np.float32)
        r = torch.tensor(r_chunk, device=args.device).unsqueeze(1)

        # Monte Carlo return-to-go for Cal-QL calibration:
        # V_cal[i] = gamma^(task_end - i) * 1.0 (sparse terminal reward)
        dist_to_reward = (task_end_for_frame[idx] - idx).astype(np.float32)
        v_cal = np.where(dist_to_reward >= 0, args.gamma ** dist_to_reward, 0.0)
        v_cal = torch.tensor(v_cal, dtype=torch.float32, device=args.device).unsqueeze(1)

        # Next chunk from demos
        a_next_chunk_np = np.stack([
            actions[i + C:i + 2 * C] for i in idx
        ])
        a_next_chunk = torch.tensor(a_next_chunk_np, dtype=torch.float32, device=args.device)

        # Done mask: if chunk contains reward=1, that's terminal for the task
        done = (r > 0.5).float()

        # Bellman: Q(s,a) ← r + γ (1-done) Q_target(s', a')
        with torch.no_grad():
            q_next = target_q(s_next, a_next_chunk)
            target = r + args.gamma * (1.0 - done) * q_next

        q_pred = q_net(s_t, a_chunk)
        td_loss = ((q_pred - target) ** 2).mean()

        # Cal-QL (Nakamoto 2023): calibrated conservative penalty.
        # Random action Q bounded above by max(Q_demo.detach(), V_cal),
        # preventing over-pessimism while keeping OOD conservatism.
        a_random = torch.empty_like(a_chunk).uniform_(-1, 1)
        q_random = q_net(s_t, a_random)
        upper = torch.maximum(q_pred.detach(), v_cal)
        cql_loss = (q_random - upper).clamp(min=0).mean()

        loss = td_loss + 0.5 * cql_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        opt.step()

        # Soft target update
        with torch.no_grad():
            for p, tp in zip(q_net.parameters(), target_q.parameters()):
                tp.data.mul_(1 - args.tau).add_(p.data, alpha=args.tau)

        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta = (args.steps - step - 1) / rate / 60
            q_mean = q_pred.mean().item()
            q_rnd = q_random.mean().item()
            print(f"Step {step+1:>6d} | td={td_loss.item():.4f} "
                  f"cql={cql_loss.item():.4f} "
                  f"q_demo={q_mean:.2f} q_rand={q_rnd:.2f} "
                  f"diff={q_mean-q_rnd:.2f} | {rate:.0f} steps/s ETA {eta:.1f}min")
            sys.stdout.flush()
            curve['step'].append(step + 1)
            curve['loss'].append(loss.item())
            curve['q_mean'].append(q_mean)

    torch.save({
        'q_net': q_net.state_dict(),
        'state_dim': state_dim,
        'action_dim': action_dim,
        'chunk_size': C,
        'config': vars(args),
    }, os.path.join(out_dir, 'tea_q.pt'))

    with open(os.path.join(out_dir, 'learning_curve.json'), 'w') as f:
        json.dump(curve, f)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f}min. Saved to {out_dir}")


if __name__ == '__main__':
    main()
