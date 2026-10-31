#!/usr/bin/env python3
"""Policy-Agnostic RL (PA-RL) core module.

Reference: Mark et al., ICLR 2025, arxiv 2412.06685.
PyTorch port of github.com/MaxSobolMark/PolicyAgnosticRL (JAX, no LICENSE).

Core algorithm:
  1. Sample K action chunks from any base policy pi_base(s)
  2. Global: keep top-m chunks by Q(s, a_k)
  3. Local: for L steps, a_k += lr * grad_a Q(s, a_k); clamp
  4. Select argmax Q across all visited actions (or sample via softmax over Q)

Unified critic: PARAFAC multilinear Q (TEA's Adaptive State Tensor) with
Cal-QL (calibrated conservative Q-learning) loss for offline + online
fine-tuning compatibility.

Framing (paper): single-agent MDP = degenerate Nash game; soft-Q + entropy
fixed point = QRE (McKelvey-Palfrey 1995); temperature annealing recovers
Nash equilibrium (Sokota et al. 2022, MMD paper).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Q-Network: PARAFAC multilinear over (state, action_chunk)
# ============================================================================

class PARLQNetwork(nn.Module):
    """Q(state, action_chunk) via PARAFAC multilinear decomposition.

    Args:
        state_dim: flat state dim (proprio+scene concat)
        action_dim: per-step action dim (7 for 7-DOF, 2 for LT)
        chunk_size: action chunk length (1 for per-step, 10 for CALVIN, etc.)
        hidden_dim: state/action encoder hidden size
        num_modes: PARAFAC rank (mode count in product)
    """

    def __init__(self, state_dim=39, action_dim=7, chunk_size=1,
                 hidden_dim=512, num_modes=3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(chunk_size * action_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.mode_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            for _ in range(num_modes)
        ])
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action_chunk):
        """
        state: [B, state_dim]
        action_chunk: [B, chunk_size, action_dim] or [B, chunk_size*action_dim]
        Returns: [B, 1]
        """
        B = state.shape[0]
        if action_chunk.dim() == 3:
            a_flat = action_chunk.reshape(B, -1)
        else:
            a_flat = action_chunk
        s_enc = self.state_encoder(state)
        a_enc = self.action_encoder(a_flat)
        fused = s_enc * a_enc
        joint = self.mode_encoders[0](fused)
        for mod in self.mode_encoders[1:]:
            joint = joint * mod(fused)
        return self.q_head(joint)


class QEnsemble(nn.Module):
    """Ensemble of Q-networks for pessimism + UTD.

    Double-Q via .min() reduction over ensemble for target computation.
    """

    def __init__(self, n_ensemble=4, **q_kwargs):
        super().__init__()
        self.nets = nn.ModuleList([PARLQNetwork(**q_kwargs) for _ in range(n_ensemble)])
        self.n_ensemble = n_ensemble

    def forward(self, state, action, reduce='mean'):
        """Return Q-values with ensemble reduction.

        reduce: 'mean' | 'min' | 'none' (returns [E, B, 1])
        """
        qs = torch.stack([n(state, action) for n in self.nets], dim=0)
        if reduce == 'mean':
            return qs.mean(0)
        elif reduce == 'min':
            return qs.min(0).values
        return qs


# ============================================================================
# Cal-QL Loss (Calibrated Conservative Q-Learning)
# ============================================================================

def calql_loss(q_net, target_q_net, batch, *, gamma=0.99, n_random=10,
               cql_alpha=0.01, use_calql=True):
    """Cal-QL critic loss = TD + calibrated CQL penalty.

    Calibration anchor: max(Q(s, a_random), V_ref) where V_ref is Monte-Carlo
    return-to-go from behavior policy. Prevents over-pessimism that breaks
    online fine-tuning (Nakamoto et al., arxiv 2303.05479).

    batch keys:
        state_t: [B, S]
        action_t: [B, C, A] or [B, C*A]
        reward_t: [B, 1]   (sum over chunk or terminal bonus)
        state_tp1: [B, S]
        action_tp1: [B, C, A]  (next chunk from demo or online buffer)
        done: [B, 1]  (1 if terminal)
        vref: [B, 1]  (MC return-to-go from demo; for calibration)

    Returns:
        loss (scalar tensor), metrics dict
    """
    s, a = batch['state_t'], batch['action_t']
    r = batch['reward_t']
    sp = batch['state_tp1']
    ap = batch['action_tp1']
    done = batch.get('done', batch.get('done_t'))
    if done is None:
        raise KeyError("batch missing 'done' or 'done_t'")
    B = s.shape[0]

    if r.dim() == 1:
        r = r.unsqueeze(-1)
    if done.dim() == 1:
        done = done.unsqueeze(-1)

    with torch.no_grad():
        q_next = target_q_net(sp, ap, reduce='min')
        if q_next.dim() == 1:
            q_next = q_next.unsqueeze(-1)
        y = r + gamma * (1.0 - done) * q_next

    # TD loss on ensemble (each net gets same target)
    qs = q_net(s, a, reduce='none')  # [E, B, 1]
    td_loss = ((qs - y[None]) ** 2).mean()

    # Cal-QL penalty: random action Q should not exceed V_ref
    device = s.device
    a_rand_shape = a.shape if a.dim() == 3 else (B, q_net.nets[0].chunk_size, q_net.nets[0].action_dim)
    a_rand = torch.empty(n_random, *a_rand_shape, device=device).uniform_(-1, 1)
    # compute Q on random actions (broadcast state)
    s_rep = s[None].expand(n_random, -1, -1).reshape(n_random * B, -1)
    ar_flat = a_rand.reshape(n_random * B, *a_rand_shape[1:])
    q_rand = q_net(s_rep, ar_flat, reduce='mean').reshape(n_random, B, 1)
    q_rand_max = q_rand.max(0).values  # [B, 1]

    q_demo = q_net(s, a, reduce='mean')  # [B, 1]

    if use_calql:
        vref = batch.get('vref', torch.zeros_like(y))
        # Calibration: penalty only if random Q exceeds max(Q_demo, V_ref)
        anchor = torch.maximum(q_demo.detach(), vref)
        cql_pen = (q_rand_max - anchor).clamp(min=0).mean()
    else:
        cql_pen = (q_rand_max - q_demo.detach()).clamp(min=0).mean()

    loss = td_loss + cql_alpha * cql_pen
    metrics = {
        'td_loss': td_loss.item(),
        'cql_pen': cql_pen.item(),
        'q_demo': q_demo.mean().item(),
        'q_rand_max': q_rand_max.mean().item(),
        'y_mean': y.mean().item(),
    }
    return loss, metrics


# ============================================================================
# PA-RL Inference: K-sample + top-m + local gradient opt
# ============================================================================

@torch.no_grad()
def _q_eval(q_net, state, actions):
    """Batched Q eval. state: [S], actions: [N, C, A]. Returns [N]."""
    N = actions.shape[0]
    s = state[None].expand(N, -1) if state.dim() == 1 else state.expand(N, -1)
    q = q_net(s, actions, reduce='mean')  # [N, 1]
    return q.squeeze(-1)


def parl_rerank(state, candidates, q_net, *, m=10, L=10, lr=3e-4,
                action_low=-1.0, action_high=1.0, argmax=True,
                softmax_temperature=1.0):
    """PA-RL two-stage action selector.

    Stage 1 (global): of K candidates, keep top-m by Q.
    Stage 2 (local): L gradient-ascent steps on kept actions to maximize Q.
    Track best action seen across all local steps.

    Args:
        state: [state_dim] tensor (single timestep)
        candidates: [K, chunk_size, action_dim] candidate actions from base policy
        q_net: Q-network (ensemble or single)
        m: number of top candidates to keep for local opt
        L: local gradient steps
        lr: action-space learning rate
        action_low/high: action clamp bounds
        argmax: if True, return argmax; else sample via Categorical(logits=Q/tau)
        softmax_temperature: tau for stochastic selection (QRE softmax)

    Returns:
        selected_action: [chunk_size, action_dim]
        info: dict with 'q_initial', 'q_final', 'q_best', 'step_gains'
    """
    K = candidates.shape[0]
    device = candidates.device
    if state.dim() == 1:
        state = state.to(device)
    m = min(m, K)

    # Stage 1: global search
    q0 = _q_eval(q_net, state, candidates)  # [K]
    topm_idx = q0.topk(m).indices
    a = candidates[topm_idx].detach().clone().requires_grad_(True)  # [m, C, A]

    best_a = a.detach().clone()
    best_q = q0[topm_idx].clone()
    step_gains = []

    # Stage 2: local gradient ascent
    s_batch = state[None].expand(m, -1) if state.dim() == 1 else state.expand(m, -1)
    for step in range(L):
        q = q_net(s_batch, a, reduce='mean').squeeze(-1)  # [m]
        g = torch.autograd.grad(q.sum(), a, retain_graph=False)[0]
        with torch.no_grad():
            a_new = (a + lr * g).clamp(action_low, action_high)
            q_new = q_net(s_batch, a_new, reduce='mean').squeeze(-1)
            mask = q_new > best_q
            best_q = torch.where(mask, q_new, best_q)
            best_a = torch.where(mask[:, None, None], a_new, best_a)
            step_gains.append((q_new - q).mean().item())
        a = a_new.detach().requires_grad_(True)

    info = {
        'q_initial': q0[topm_idx].mean().item(),
        'q_final': best_q.mean().item(),
        'q_best': best_q.max().item(),
        'step_gains': step_gains,
        'K': K, 'm': m, 'L': L,
    }

    if argmax:
        selected = best_a[best_q.argmax()]
    else:
        tau = max(softmax_temperature, 1e-6)
        probs = torch.softmax(best_q / tau, dim=0)
        idx = torch.distributions.Categorical(probs=probs).sample()
        selected = best_a[idx]
    return selected, info


def parl_rerank_multi_agent(state, candidates_per_agent, q_shared, *,
                            m=10, L=10, lr=3e-4, **kw):
    """Multi-agent PA-RL (CTDE): per-agent local opt on own actions with
    shared centralized Q. Used for ALOHA bimanual (n=2).

    candidates_per_agent: list of [K_i, C, A_i] per agent
    q_shared: Q-network over joint state + joint action (concat per agent)
    """
    per_agent = [
        parl_rerank(state, c, q_shared, m=m, L=L, lr=lr, **kw)
        for c in candidates_per_agent
    ]
    joint_action = torch.cat([pa[0] for pa in per_agent], dim=-1)
    info = {'per_agent': [pa[1] for pa in per_agent]}
    return joint_action, info


# ============================================================================
# QRE Policy (McKelvey-Palfrey 1995 / Sokota 2022 soft-Q = QRE bridge)
# ============================================================================

def qre_softmax_select(actions, q_values, tau=1.0):
    """Quantal Response Equilibrium softmax selection over K sampled actions.

    Nash framing: soft-Q + entropy fixed point = QRE. As tau -> 0, recovers
    Nash equilibrium best-response (Theorem 1 in paper).

    Args:
        actions: [K, ...] candidate actions
        q_values: [K] Q-scores
        tau: temperature; 0.0 = hard argmax
    """
    if tau <= 1e-6:
        return actions[q_values.argmax()]
    probs = torch.softmax(q_values / tau, dim=0)
    idx = torch.distributions.Categorical(probs=probs).sample()
    return actions[idx]


# ============================================================================
# Training helpers
# ============================================================================

def soft_update_(target, source, tau=0.005):
    """Polyak target update: theta_target <- (1-tau) * theta_target + tau * theta."""
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)


def compute_mc_return_to_go(rewards, gamma=0.99):
    """Monte-Carlo return-to-go for Cal-QL calibration anchor.

    rewards: np.ndarray [T] scalar rewards along trajectory.
    Returns: np.ndarray [T] with R_t = sum_{k>=t} gamma^{k-t} r_k
    """
    T = len(rewards)
    out = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in range(T - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out
