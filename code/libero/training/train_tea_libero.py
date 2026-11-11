#!/usr/bin/env python3
"""TEA Q-network training on LIBERO demos (Cal-QL + PARAFAC multilinear).

Clones train_tea_mdt.py logic for LIBERO:
- state_dim=15 (proprio: ee_pos+ori+gripper+joints)
- action_dim=7
- chunk_size: CLI default 8, but the paper's shipped critics use --chunk_size 1
  (scoring the first action of each FLOWER chunk)
- reward=+1 at episode_end, 0 elsewhere
- Cal-QL penalty: random action Q <= max(Q_demo.detach(), V_MC_return_to_go),
  loss = TD + 0.5 * penalty, one uniform-random action per state
"""

import argparse, os, sys, json, time, numpy as np, torch, torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                os.pardir, os.pardir, "calvin", "training"))
from train_tea_mdt import ActionChunkQNetwork


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--suite", type=str, default="spatial")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.output_dir, f"tea_libero_{args.suite}_s{args.seed}_{ts}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print(f"TEA Q-net for LIBERO-{args.suite}")
    print("=" * 60)

    obs = np.load(os.path.join(args.cache_dir, 'obs.npy'))         # [N, 15]
    actions = np.load(os.path.join(args.cache_dir, 'actions.npy')) # [N, 7]
    ends = np.load(os.path.join(args.cache_dir, 'episode_ends.npy'))
    n = len(obs)
    print(f"  frames={n}, episodes={len(ends)}")
    print(f"  state_dim={obs.shape[1]}, action_dim={actions.shape[1]}")

    # Per-frame reward + task_end
    reward_per_frame = np.zeros(n, dtype=np.float32)
    task_end_for_frame = -np.ones(n, dtype=np.int64)
    prev = 0
    for e in ends:
        reward_per_frame[e] = 1.0
        task_end_for_frame[prev:e + 1] = e
        prev = e + 1

    C = args.chunk_size
    state_dim = obs.shape[1]
    action_dim = actions.shape[1]

    q_net = ActionChunkQNetwork(
        state_dim=state_dim, action_dim=action_dim, chunk_size=C
    ).to(args.device)
    target_q = ActionChunkQNetwork(
        state_dim=state_dim, action_dim=action_dim, chunk_size=C
    ).to(args.device)
    target_q.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    print(f"  Q-net: {sum(p.numel() for p in q_net.parameters()):,} params")

    # Valid indices: within task segments and with chunk+next-chunk available
    valid_idx = np.where(task_end_for_frame >= 0)[0]
    valid_idx = valid_idx[valid_idx < n - 2 * C - 1]
    print(f"  valid_idx: {len(valid_idx)}")

    curve = {'step': [], 'loss': [], 'q_mean': []}
    t0 = time.time()

    for step in range(args.steps):
        idx = valid_idx[np.random.randint(0, len(valid_idx), args.batch_size)]
        s_t = torch.tensor(obs[idx], dtype=torch.float32, device=args.device)
        s_next = torch.tensor(obs[idx + C], dtype=torch.float32, device=args.device)
        a_chunk = np.stack([actions[i:i + C] for i in idx])
        a_chunk = torch.tensor(a_chunk, dtype=torch.float32, device=args.device)

        r_chunk = np.array([reward_per_frame[i:i + C].sum() for i in idx], dtype=np.float32)
        r = torch.tensor(r_chunk, device=args.device).unsqueeze(1)

        dist_to_reward = (task_end_for_frame[idx] - idx).astype(np.float32)
        v_cal = np.where(dist_to_reward >= 0, args.gamma ** dist_to_reward, 0.0)
        v_cal = torch.tensor(v_cal, dtype=torch.float32, device=args.device).unsqueeze(1)

        a_next = np.stack([actions[i + C:i + 2 * C] for i in idx])
        a_next = torch.tensor(a_next, dtype=torch.float32, device=args.device)

        done = (r > 0.5).float()
        with torch.no_grad():
            q_next = target_q(s_next, a_next)
            target = r + args.gamma * (1.0 - done) * q_next

        q_pred = q_net(s_t, a_chunk)
        td_loss = ((q_pred - target) ** 2).mean()

        a_random = torch.empty_like(a_chunk).uniform_(-1, 1)
        q_random = q_net(s_t, a_random)
        upper = torch.maximum(q_pred.detach(), v_cal)
        cql_loss = (q_random - upper).clamp(min=0).mean()

        loss = td_loss + 0.5 * cql_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for p_, tp in zip(q_net.parameters(), target_q.parameters()):
                tp.data.mul_(1 - args.tau).add_(p_.data, alpha=args.tau)

        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta = (args.steps - step - 1) / rate / 60
            q_mean = q_pred.mean().item()
            q_rnd = q_random.mean().item()
            print(f"Step {step+1:>6d} | td={td_loss.item():.4f} "
                  f"cql={cql_loss.item():.4f} "
                  f"q_demo={q_mean:.2f} q_rand={q_rnd:.2f} "
                  f"diff={q_mean-q_rnd:.2f} | {rate:.0f} st/s ETA {eta:.1f}min", flush=True)
            curve['step'].append(step + 1)
            curve['loss'].append(loss.item())
            curve['q_mean'].append(q_mean)

    torch.save({
        'q_net': q_net.state_dict(),
        'state_dim': state_dim, 'action_dim': action_dim, 'chunk_size': C,
        'config': vars(args),
    }, os.path.join(out, 'tea_q.pt'))
    with open(os.path.join(out, 'learning_curve.json'), 'w') as f:
        json.dump(curve, f)
    print(f"\nDone. Saved to {out}")


if __name__ == '__main__':
    main()
