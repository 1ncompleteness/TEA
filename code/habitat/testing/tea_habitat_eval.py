#!/usr/bin/env python3
"""TEA Habitat 3.0 Social-Nav evaluation with full social-nav metrics.

Extends run_habitat_eval.py with:
  * Social-nav metrics matching Habitat 3.0 paper Fig. 3:
      S   = social_nav_stats.has_found_human            (finding success)
      SPS = social_nav_stats.first_encounter_spl        (success weighted by path steps)
      F   = social_nav_stats.follow_human_steps_ratio_after_frist_encounter
      CR  = fraction of episodes with any agent-agent collision (num_agents_collide > 0)
  * Soft Best-of-N rerank (TEA Eq. 5):
      score_k = -||a_k - a_greedy||^2 / (2 sigma^2) + Q(s, a_k) / alpha
    with a_greedy the policy's deterministic action. --hard_argmax falls back
    to plain argmax_k Q.
  * Multi-seed loop; per-seed and aggregate JSON output.

Usage:
    python tea_habitat_eval.py --ckpt ckpt/social_nav_latest.pth --K 1 \
        --num_episodes 30 --seeds 42 43 44 45 46 --output_dir results/habitat_K1
    python tea_habitat_eval.py --ckpt ... --q_path tea_q.pt --K 4 \
        --soft_bon --alpha 2.0 --sigma 0.2 ...
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "training"))

from train_tea_habitat import (
    build_joint_action,
    flatten_obs, make_env, load_pretrained_policy, policy_act, policy_specs,
    safe_reset,
)


def get_measure(info: dict, key: str, default=None):
    """Fetch a measure that may be agent-prefixed or nested."""
    if key in info:
        return info[key]
    for prefix in ("agent_0_", "agent_1_"):
        if prefix + key in info:
            return info[prefix + key]
    for v in info.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return default


def extract_social_metrics(info: dict) -> dict:
    stats = get_measure(info, "social_nav_stats", {}) or {}
    num_collide = get_measure(info, "num_agents_collide", 0) or 0
    return {
        "S": float(bool(stats.get("has_found_human", False))),
        "SPS": float(stats.get("first_encounter_spl", 0.0) or 0.0),
        "F": float(
            stats.get("follow_human_steps_ratio_after_frist_encounter", 0.0)
            or 0.0
        ),
        "collided": float((num_collide or 0) > 0),
        "num_agents_collide": float(num_collide or 0),
        "raw_stats": {k: float(v) for k, v in stats.items()
                      if isinstance(v, (int, float, bool, np.floating))},
    }


def act_deterministic(policy, obs_batch, rnn_state, prev_action, masks):
    return policy_act(policy, obs_batch, rnn_state, prev_action, masks,
                      deterministic=True)


def sample_k_actions(policy, obs_batch, rnn_state, prev_action, masks, K):
    actions = []
    rnn_out = None
    for _ in range(K):
        a, rnn = policy_act(policy, obs_batch, rnn_state, prev_action,
                            masks, deterministic=False)
        actions.append(a.squeeze(0))
        rnn_out = rnn
    return torch.stack(actions, dim=0), rnn_out


def select_action(q_ensemble, state_t, actions_K, a_greedy, *,
                  soft_bon: bool, alpha: float, sigma: float):
    K = actions_K.shape[0]
    s_rep = state_t.unsqueeze(0).expand(K, -1)
    with torch.no_grad():
        q_vals = q_ensemble(s_rep, actions_K.unsqueeze(1),
                            reduce="mean").squeeze(-1)  # [K]
    if soft_bon:
        dist2 = ((actions_K - a_greedy.unsqueeze(0)) ** 2).sum(dim=-1)
        scores = -dist2 / (2.0 * sigma ** 2) + q_vals / alpha
    else:
        scores = q_vals
    best = int(scores.argmax().item())
    return best, q_vals.cpu().numpy().tolist(), scores.cpu().numpy().tolist()


def run_seed(env, policy, q_ensemble, args, seed: int, output_dir: str):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = args.device
    n_layers, hidden, act_dim = policy_specs(policy)
    per_ep = []
    for ep in range(args.num_episodes):
        obs = safe_reset(env)
        rnn_state = torch.zeros(1, n_layers, hidden, device=device)
        prev_action = torch.zeros(1, act_dim, device=device)
        masks = torch.zeros(1, 1, dtype=torch.bool, device=device)
        ep_ret, info = 0.0, {}
        for t in range(args.max_episode_steps):
            obs_batch = {k: torch.as_tensor(np.asarray(v), device=device).unsqueeze(0)
                         for k, v in obs.items()
                         if isinstance(v, (np.ndarray, list))}
            if args.K == 1 or q_ensemble is None:
                action, rnn_state = act_deterministic(
                    policy, obs_batch, rnn_state, prev_action, masks)
                a_np = action.squeeze(0).cpu().numpy()
            else:
                a_greedy, _ = act_deterministic(
                    policy, obs_batch, rnn_state, prev_action, masks)
                actions_K, rnn_state = sample_k_actions(
                    policy, obs_batch, rnn_state, prev_action, masks, args.K)
                state_flat = torch.from_numpy(flatten_obs(obs)).to(device)
                best_k, _, _ = select_action(
                    q_ensemble, state_flat, actions_K, a_greedy.squeeze(0),
                    soft_bon=args.soft_bon, alpha=args.alpha, sigma=args.sigma)
                action = actions_K[best_k].unsqueeze(0)
                a_np = actions_K[best_k].cpu().numpy()
            step_out = env.step(build_joint_action(a_np))
            if isinstance(step_out, list):
                obs, r, done, info = step_out[0]
            else:
                obs, r, done, info = step_out
            if isinstance(obs, list):
                obs = obs[0]
            r = float(np.asarray(r).reshape(-1)[0])
            done = bool(np.asarray(done).reshape(-1)[0])
            if isinstance(info, list):
                info = info[0]
            ep_ret += float(r)
            prev_action = action
            masks = torch.ones(1, 1, dtype=torch.bool, device=device)
            if done:
                break
        m = extract_social_metrics(info)
        m.update({"ep": ep, "seed": seed, "ret": ep_ret, "steps": t + 1})
        per_ep.append(m)
        print(f"  seed {seed} ep {ep+1}/{args.num_episodes}: "
              f"S={m['S']:.0f} SPS={m['SPS']:.3f} F={m['F']:.3f} "
              f"collided={m['collided']:.0f} steps={t+1}", flush=True)
        with open(os.path.join(output_dir, f"s{seed}_ep{ep:03d}.json"),
                  "w") as f:
            json.dump(per_ep[-1], f)
    return per_ep


def aggregate(per_ep):
    arr = lambda k: np.array([e[k] for e in per_ep], dtype=np.float64)
    return {
        "n": len(per_ep),
        "S": float(arr("S").mean()),
        "SPS": float(arr("SPS").mean()),
        "F": float(arr("F").mean()),
        "CR": float(arr("collided").mean()),
        "S_std": float(arr("S").std()),
        "SPS_std": float(arr("SPS").std()),
        "F_std": float(arr("F").std()),
        "CR_std": float(arr("collided").std()),
        "mean_steps": float(arr("steps").mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--q_path", default=None)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--soft_bon", action="store_true")
    p.add_argument("--hard_argmax", action="store_true")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--num_episodes", type=int, default=30)
    p.add_argument("--max_episode_steps", type=int, default=1500)
    p.add_argument("--state_dim", type=int, default=8)
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--n_ensemble", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", default="val",
                   help="dataset split (val = held-out scenes)")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = (f"teaK{args.K}_" + ("soft" if args.soft_bon else "hard")
                if args.K > 1 else "baseK1")
        args.output_dir = f"results/habitat_{mode}_{ts}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}", flush=True)

    env, cfg = make_env(args.config, split=args.split)
    policy = load_pretrained_policy(args.ckpt, env, cfg, device=args.device)
    policy.eval()

    q_ensemble = None
    if args.K > 1:
        if args.q_path is None:
            raise SystemExit("K>1 requires --q_path")
        from pa_rl import QEnsemble
        q_ensemble = QEnsemble(
            n_ensemble=args.n_ensemble, state_dim=args.state_dim,
            action_dim=args.action_dim, chunk_size=1, hidden_dim=256,
            num_modes=3,
        ).to(args.device)
        q_ensemble.load_state_dict(
            torch.load(args.q_path, map_location=args.device,
                       weights_only=True))
        q_ensemble.eval()
        print(f"Loaded Q-ensemble from {args.q_path}", flush=True)

    t0 = time.time()
    all_ep = []
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        all_ep.extend(run_seed(env, policy, q_ensemble, args, seed,
                               args.output_dir))
    agg = aggregate(all_ep)
    agg["wall_time_s"] = time.time() - t0
    agg["config"] = {k: v for k, v in vars(args).items()}
    with open(os.path.join(args.output_dir, "aggregate.json"), "w") as f:
        json.dump({"aggregate": agg, "per_episode": all_ep}, f, indent=2)
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
