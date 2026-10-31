#!/usr/bin/env python3
"""Train TEA Q-net on Habitat 3.0 Social Nav demos collected from
the pretrained DD-PPO Gaussian Spot policy.

Pipeline:
  1. Load pretrained policy from social_nav_latest.pth (agent_0).
  2. Roll out N episodes in habitat_baselines social_nav env to collect (s,a,r,s') tuples.
  3. Train PARAFAC Q-ensemble with Cal-QL loss on collected demos.
  4. Optionally finetune actor with AWR+Cal-QL via HabitatGaussianRecurrentAdapter.

Run inside tea-habitat:latest Docker container.

Usage:
    python train_tea_habitat.py \
        --ckpt /workspace/results/habitat_pretrained/social_nav_latest.pth \
        --num_demo_episodes 100 \
        --train_steps 20000 \
        --output_dir /workspace/results/tea_habitat_s42_<timestamp>

Output:
    - tea_q.pt: trained PARAFAC Q-ensemble
    - demos.npz: cached demo tuples
    - learning_curve.json: train metrics
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# State featurization for Q-net
# ---------------------------------------------------------------------------
# Habitat dict obs has many keys (head_depth, articulated_agent_arm_depth,
# obj_start_gps_compass, etc.). For Q-net, flatten a fixed subset to a
# state vector of stable dimension. Visual is downsampled via mean pool.

# State featurization v2 (39-D):
#   [0:3]   agent_1_localization_sensor       (humanoid position)  [neighbor]
#   [3:7]   humanoid detector bbox image: mean, max, center-of-mass x, y
#                                                                  [neighbor]
#   [7:23]  arm depth, 4x4 average-pooled grid
#   [23:39] head stereo depth, 4x4 average-pooled grid
# With TEA_NEIGHBOR_BLIND=1 the neighbor block [0:7] is zeroed (dims
# preserved) for the RSGG neighbor-blind ablation.
NEIGHBOR_SLICE = slice(0, 7)
STATE_DIM_V2 = 39
NEIGHBOR_BLIND = os.environ.get("TEA_NEIGHBOR_BLIND", "0") == "1"


def _pool_grid(img: np.ndarray, g: int = 4) -> np.ndarray:
    v = np.asarray(img, dtype=np.float32)
    if v.ndim == 3:
        v = v[..., 0]
    h, w = v.shape
    hs, ws = h // g, w // g
    v = v[: hs * g, : ws * g].reshape(g, hs, g, ws)
    return v.mean(axis=(1, 3)).reshape(-1)


def _bbox_stats(img: np.ndarray) -> np.ndarray:
    v = np.asarray(img, dtype=np.float32)
    if v.ndim == 3:
        v = v[..., 0]
    mean, vmax = float(v.mean()), float(v.max())
    if vmax > 0:
        ys, xs = np.nonzero(v)
        com_y = float(ys.mean()) / v.shape[0]
        com_x = float(xs.mean()) / v.shape[1]
    else:
        com_x = com_y = 0.0
    return np.array([mean, vmax, com_x, com_y], dtype=np.float32)


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Project Habitat dict obs to a 39-D state vector for the Q-net."""
    loc = np.zeros(3, dtype=np.float32)
    if "agent_1_localization_sensor" in obs:
        loc = np.asarray(obs["agent_1_localization_sensor"],
                         dtype=np.float32).reshape(-1)[:3]
    det = np.zeros(4, dtype=np.float32)
    if "agent_0_humanoid_detector_sensor" in obs:
        det = _bbox_stats(obs["agent_0_humanoid_detector_sensor"])
    arm = np.zeros(16, dtype=np.float32)
    if "agent_0_articulated_agent_arm_depth" in obs:
        arm = _pool_grid(obs["agent_0_articulated_agent_arm_depth"])
    stereo = np.zeros(16, dtype=np.float32)
    if "agent_0_spot_head_stereo_depth_sensor" in obs:
        stereo = _pool_grid(obs["agent_0_spot_head_stereo_depth_sensor"])
    s = np.concatenate([loc, det, arm, stereo]).astype(np.float32)
    if NEIGHBOR_BLIND:
        s[NEIGHBOR_SLICE] = 0.0
    return s


# ---------------------------------------------------------------------------
# Demo collection
# ---------------------------------------------------------------------------

# Flat joint-action layout of the social-nav benchmark (25 dims):
# [0:2]=agent_0 base_vel | [2:4]=agent_1 base_vel | [4:5]=agent_1 oracle_nav
# [5:6]=agent_1 oracle_nav_randcoord (1 => walk to random waypoints, replan)
# [6:24]=agent_1 pddl (18) | [24:25]=agent_1 rearrange_stop
JOINT_ACTION_DIM = 25


def build_joint_action(a0_np: np.ndarray) -> np.ndarray:
    """Embed agent_0's 2-D base velocity into the joint action; the scripted
    humanoid (agent_1) walks randomized waypoints via oracle_nav_randcoord."""
    joint = np.zeros(JOINT_ACTION_DIM, dtype=np.float32)
    joint[0:2] = a0_np
    joint[5] = 1.0
    return joint


def policy_specs(policy):
    """(num_recurrent_layers, hidden_size, action_dim) across API versions."""
    net = policy.net
    n_layers = getattr(net, "num_recurrent_layers", 2)
    hidden = getattr(net, "recurrent_hidden_size",
                     getattr(net, "_hidden_size", 512))
    act_dim = policy.prev_action_shape[0] if hasattr(
        policy, "prev_action_shape") else 2
    return n_layers, hidden, act_dim


def policy_act(policy, obs_batch, rnn, prev_a, masks, *, deterministic):
    """Call policy.act across API versions; returns (action, rnn_state)."""
    with torch.no_grad():
        out = policy.act(obs_batch, rnn, prev_a, masks,
                         deterministic=deterministic)
    if hasattr(out, "actions"):
        return out.actions, out.rnn_hidden_states
    value_or_a, a_or_rnn, _, last = out
    # Heuristic for legacy tuple order (value, action, logp, rnn).
    return a_or_rnn, last


def safe_reset(env, max_tries: int = 25):
    """Reset, skipping episodes whose scene/object init is broken."""
    last = None
    for i in range(max_tries):
        try:
            obs = env.reset()
            return obs[0] if isinstance(obs, list) else obs
        except Exception as e:  # noqa: BLE001 - habitat raises bare asserts
            last = e
            print(f"  WARN reset failed ({type(e).__name__}: {str(e)[:90]}); "
                  f"skipping episode ({i+1}/{max_tries})", flush=True)
            if hasattr(env, "rebuild") and i > 0 and i % 6 == 0:
                env.rebuild()
    raise RuntimeError(f"env.reset failed {max_tries}x: {last}")


def collect_demos(env, policy, num_episodes: int, max_steps: int = 500,
                  device: str = "cuda",
                  checkpoint_path: str = None,
                  checkpoint_every: int = 25) -> Dict[str, np.ndarray]:
    """Roll out pretrained policy. Returns offline replay buffer."""
    states, actions, rewards, next_states, dones = [], [], [], [], []

    def _snapshot():
        return {
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.asarray(next_states, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }

    n_layers, hidden, act_dim = policy_specs(policy)
    for ep in range(num_episodes):
        obs = safe_reset(env)
        rnn_state = torch.zeros(1, n_layers, hidden, device=device)
        prev_action = torch.zeros(1, act_dim, device=device)
        masks = torch.zeros(1, 1, dtype=torch.bool, device=device)
        ep_ret = 0.0
        for t in range(max_steps):
            obs_batch = {k: torch.as_tensor(np.asarray(v), device=device).unsqueeze(0)
                         for k, v in obs.items() if isinstance(v, (np.ndarray, list))}
            action, rnn_state = policy_act(
                policy, obs_batch, rnn_state, prev_action, masks,
                deterministic=False)
            a_np = action.squeeze(0).cpu().numpy()
            s = flatten_obs(obs)
            step_out = env.step(build_joint_action(a_np))
            if isinstance(step_out, list):
                obs, r, done, info = step_out[0]
            else:
                obs, r, done, info = step_out
            if isinstance(obs, list):
                obs = obs[0]
            r = float(np.asarray(r).reshape(-1)[0])
            done = bool(np.asarray(done).reshape(-1)[0])
            s2 = flatten_obs(obs)
            states.append(s); actions.append(a_np); rewards.append(r)
            next_states.append(s2); dones.append(done)
            ep_ret += r
            prev_action = action
            masks = torch.ones(1, 1, dtype=torch.bool, device=device)
            if done:
                break
        print(f"  ep {ep+1}/{num_episodes}: return={ep_ret:.3f}, steps={t+1}",
              flush=True)
        if checkpoint_path and (ep + 1) % checkpoint_every == 0:
            np.savez(checkpoint_path, **_snapshot())
            print(f"  [checkpoint] {ep+1} eps -> {checkpoint_path}",
                  flush=True)

    return _snapshot()


# ---------------------------------------------------------------------------
# Q-net training
# ---------------------------------------------------------------------------

def train_q_net(demos: Dict[str, np.ndarray], state_dim: int, action_dim: int,
                steps: int, batch_size: int, lr: float, n_ensemble: int,
                cql_alpha: float, gamma: float, device: str,
                log_every: int = 200) -> Tuple[object, List[Dict]]:
    """Cal-QL training of PARAFAC Q-ensemble on offline demos."""
    from pa_rl import QEnsemble, calql_loss, soft_update_

    q_kwargs = dict(state_dim=state_dim, action_dim=action_dim,
                    chunk_size=1, hidden_dim=256, num_modes=3)
    q = QEnsemble(n_ensemble=n_ensemble, **q_kwargs).to(device)
    target_q = QEnsemble(n_ensemble=n_ensemble, **q_kwargs).to(device)
    target_q.load_state_dict(q.state_dict())
    for p in target_q.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    s_all = torch.from_numpy(demos["states"]).to(device)
    a_all = torch.from_numpy(demos["actions"]).to(device)
    r_all = torch.from_numpy(demos["rewards"]).to(device)
    s2_all = torch.from_numpy(demos["next_states"]).to(device)
    d_all = torch.from_numpy(demos["dones"]).to(device)
    N = s_all.shape[0]

    metrics_log: List[Dict] = []
    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,), device=device)
        batch = {
            "state_t": s_all[idx],
            "action_t": a_all[idx].unsqueeze(1),  # chunk dim
            "reward_t": r_all[idx],
            "state_tp1": s2_all[idx],
            "action_tp1": a_all[idx].unsqueeze(1),  # placeholder; pi(s') needed in real Cal-QL
            "done_t": d_all[idx],
        }
        loss, m = calql_loss(q, target_q, batch, gamma=gamma,
                             cql_alpha=cql_alpha, use_calql=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        opt.step()
        soft_update_(target_q, q, tau=0.005)

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                q_demo = q(s_all[:512], a_all[:512].unsqueeze(1), reduce="mean").mean().item()
                a_rand = torch.empty_like(a_all[:512]).uniform_(-1, 1)
                q_rand = q(s_all[:512], a_rand.unsqueeze(1), reduce="mean").mean().item()
            entry = {"step": step + 1, "loss": float(loss),
                     "q_demo": q_demo, "q_rand": q_rand,
                     "diff": q_demo - q_rand, **m}
            metrics_log.append(entry)
            print(f"  step {step+1}/{steps}: loss={loss:.4f}, "
                  f"q_demo={q_demo:.3f}, q_rand={q_rand:.3f}, "
                  f"diff={q_demo-q_rand:.3f}", flush=True)

    return q, metrics_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_SOCIAL_NAV_CFG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "habitat-lab", "habitat-baselines", "habitat_baselines", "config",
    "social_nav", "social_nav.yaml",
)
if not os.path.exists(DEFAULT_SOCIAL_NAV_CFG):
    # fall back to the installed habitat_baselines package
    import importlib.util as _ilu
    _spec = _ilu.find_spec("habitat_baselines")
    if _spec is not None and _spec.origin is not None:
        DEFAULT_SOCIAL_NAV_CFG = os.path.join(
            os.path.dirname(_spec.origin), "config", "social_nav",
            "social_nav.yaml")


class ResilientGymEnv:
    """Single-process habitat gym env with rebuild-on-fatal-episode.

    Exceptions from broken episodes propagate to the caller (unlike
    VectorEnv, where a dead worker hangs the pipe); `rebuild()` recreates
    the underlying env when its internal state is corrupted.
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._env = None
        self._build()

    def _build(self):
        import copy

        from habitat.gym.gym_definitions import make_gym_from_config
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        # make_gym_from_config mutates the config it receives; hand it a
        # fresh copy so rebuilds start from the pristine config.
        self._env = make_gym_from_config(copy.deepcopy(self._cfg))

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def inner_action_space(self):
        base = self._env
        while hasattr(base, "env"):
            base = base.env
        return base.action_space

    def reset(self):
        out = self._env.reset()
        return out[0] if isinstance(out, tuple) else out

    def step(self, action):
        out = self._env.step(action)
        if len(out) == 5:
            obs, r, term, trunc, info = out
            return obs, r, bool(term) or bool(trunc), info
        return out

    def rebuild(self):
        print("  INFO rebuilding env after fatal episode", flush=True)
        self._build()


def make_env(config_path: str = None, split: str = None,
             habitat_seed: int = None, shuffle_episodes: bool = False):
    """Build social_nav env using habitat-baselines config."""
    from omegaconf import OmegaConf
    from habitat_baselines.config.default import get_config

    cfg = get_config(config_path or DEFAULT_SOCIAL_NAV_CFG)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_readonly(cfg, False)
    cfg.habitat_baselines.evaluate = True
    cfg.habitat_baselines.num_environments = 1
    # Match the released social_nav_latest.pth training config: the humanoid
    # detector is a bbox image (3rd visual channel of the policy).
    det = cfg.habitat.task.lab_sensors.get(
        "agent_0_humanoid_detector_sensor",
        cfg.habitat.task.lab_sensors.get("humanoid_detector_sensor"),
    )
    if det is not None:
        det.return_image = True
        det.is_return_image_bbox = True
    if split is not None:
        cfg.habitat.dataset.split = split
    # Paper protocol horizon (the shipped config caps episodes at 750).
    cfg.habitat.environment.max_episode_steps = 1500
    if habitat_seed is not None:
        cfg.habitat.seed = int(habitat_seed)
    if shuffle_episodes:
        cfg.habitat.environment.iterator_options.shuffle = True
        cfg.habitat.environment.iterator_options.group_by_scene = False
    OmegaConf.set_readonly(cfg, True)
    OmegaConf.set_struct(cfg, True)

    return ResilientGymEnv(cfg), cfg


def _agent0_spaces(env):
    """Split multi-agent joint spaces down to agent_0's own spaces.

    Tries prefixed and prefix-stripped observation keys; the caller picks
    the variant whose parameter shapes match the checkpoint.
    """
    import gym.spaces as gsp

    obs_space = env.observation_space
    act_space = env.inner_action_space

    a0_items = {k: v for k, v in obs_space.spaces.items()
                if k.startswith("agent_0_") and "third_rgb" not in k}
    depth_only = {k: v for k, v in a0_items.items()
                  if "detector" not in k}
    a0_obs_variants = [
        gsp.Dict(depth_only),
        gsp.Dict({k[len("agent_0_"):]: v for k, v in depth_only.items()}),
        gsp.Dict(a0_items),
        gsp.Dict({k[len("agent_0_"):]: v for k, v in a0_items.items()}),
    ]
    a0_act = gsp.Dict({
        k: v for k, v in act_space.spaces.items() if k.startswith("agent_0_")
    })
    return a0_obs_variants, a0_act


def load_pretrained_policy(ckpt_path: str, env, cfg, device: str = "cuda"):
    """Reconstruct agent_0's PointNavResNetPolicy and load its weights."""
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck[0]["state_dict"]
    ck_cfg = ck.get("config", cfg)

    obs_variants, a0_act = _agent0_spaces(env)
    last_err = None
    for use_cfg in (ck_cfg, cfg):
        for obs_space in obs_variants:
            try:
                policy = PointNavResNetPolicy.from_config(
                    use_cfg, obs_space, a0_act, agent_name="agent_0",
                ).to(device)
                policy.load_state_dict(sd, strict=True)
                print(f"Loaded policy at training step="
                      f"{ck['extra_state']['step']} "
                      f"(obs keys: {list(obs_space.spaces)[:3]}...)")
                policy._tea_obs_keys = list(obs_space.spaces.keys())
                return policy
            except (RuntimeError, KeyError) as e:
                last_err = e
    raise RuntimeError(f"No obs-space/config variant matched checkpoint: "
                       f"{last_err}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--demos_npz", default=None,
                   help="comma-separated demo buffers to load instead of "
                        "collecting; concatenated in order")
    p.add_argument("--neighbor_blind", action="store_true",
                   help="zero the neighbor block [0:7] of states before "
                        "training (RSGG neighbor-blind ablation)")
    p.add_argument("--collect_only", action="store_true",
                   help="collect demos.npz and exit without training")
    p.add_argument("--habitat_seed", type=int, default=None,
                   help="habitat sim/episode-iterator seed")
    p.add_argument("--shuffle_episodes", action="store_true")
    p.add_argument("--num_demo_episodes", type=int, default=100)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--train_steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_ensemble", type=int, default=4)
    p.add_argument("--cql_alpha", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/workspace/results/tea_habitat_s{args.seed}_{ts}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}", flush=True)

    if args.demos_npz:
        parts = []
        for path in args.demos_npz.split(","):
            print(f"Loading demos from {path}", flush=True)
            loaded = np.load(path)
            parts.append({k: loaded[k] for k in loaded.files})
        demos = {k: np.concatenate([p[k] for p in parts], axis=0)
                 for k in parts[0]}
        print(f"Merged {len(parts)} buffers -> "
              f"{demos['states'].shape[0]} transitions", flush=True)
    else:
        print("Building env + loading policy...", flush=True)
        env, cfg = make_env(args.config, habitat_seed=args.habitat_seed,
                            shuffle_episodes=args.shuffle_episodes)
        policy = load_pretrained_policy(args.ckpt, env, cfg,
                                        device=args.device)
        policy.eval()

        print(f"Collecting {args.num_demo_episodes} demo episodes...",
              flush=True)
        t0 = time.time()
        demos = collect_demos(env, policy,
                              num_episodes=args.num_demo_episodes,
                              max_steps=args.max_episode_steps,
                              device=args.device,
                              checkpoint_path=os.path.join(
                                  args.output_dir, "demos.npz"))
        print(f"Collected {len(demos['states'])} transitions in "
              f"{time.time()-t0:.1f}s", flush=True)
        np.savez(os.path.join(args.output_dir, "demos.npz"), **demos)

    if args.collect_only:
        print("collect_only: done.", flush=True)
        return

    if args.neighbor_blind:
        print("Neighbor-blind: zeroing state block [0:7]", flush=True)
        demos["states"] = demos["states"].copy()
        demos["next_states"] = demos["next_states"].copy()
        demos["states"][:, NEIGHBOR_SLICE] = 0.0
        demos["next_states"][:, NEIGHBOR_SLICE] = 0.0

    state_dim = demos["states"].shape[1]
    action_dim = demos["actions"].shape[1]
    print(f"state_dim={state_dim}, action_dim={action_dim}", flush=True)

    print("Training Q-net (Cal-QL)...", flush=True)
    q, log = train_q_net(
        demos, state_dim, action_dim,
        steps=args.train_steps, batch_size=args.batch_size,
        lr=args.lr, n_ensemble=args.n_ensemble, cql_alpha=args.cql_alpha,
        gamma=args.gamma, device=args.device,
    )

    torch.save(q.state_dict(), os.path.join(args.output_dir, "tea_q.pt"))
    with open(os.path.join(args.output_dir, "learning_curve.json"), "w") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "state_dim": state_dim, "action_dim": action_dim},
                  f, indent=2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
