"""Advantage estimators. Every algorithm here shares train.py's clipped-surrogate update;
the only difference is how rewards become a per-token advantage [N, T]:
raw returns -> constant baseline -> learned critic (GAE) -> group baseline."""
from __future__ import annotations

import functools

import torch

from core import Batch
from utils import discounted_returns, gae, masked_mean, whiten


def reinforce(batch: Batch, policy, gamma: float = 0.99) -> torch.Tensor:
    """A_t = return-to-go. The highest-variance, simplest truth (Karpathy's Pong)."""
    return discounted_returns(batch.rewards, batch.mask, gamma)


def reinforce_baseline(batch: Batch, policy, gamma: float = 0.99) -> torch.Tensor:
    """Subtract a constant baseline (the batch-mean return): same expectation, less variance."""
    G = discounted_returns(batch.rewards, batch.mask, gamma)
    return (G - masked_mean(G, batch.mask)) * batch.mask


def ppo(batch: Batch, policy, gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """GAE with a learned critic, whitened. Stashes the critic target on batch.returns;
    computed once under the OLD value net, as canonical PPO requires."""
    with torch.no_grad():
        V = policy.value(batch.states)  # [N, T]
    A, returns = gae(batch.rewards, V, batch.mask, gamma, lam)
    batch.returns = returns
    return whiten(A, batch.mask)


def _require_group(name: str, batch: Batch) -> None:
    """At group_size=1 the group mean IS the sample: advantage 0, std NaN. Fail loudly here
    instead of NaN-ing steps later inside Categorical."""
    if batch.group_size < 2:
        raise ValueError(
            f"--algo {name} needs --group-size >= 2 (got {batch.group_size}): its baseline is "
            f"built from the OTHER samples of the same prompt. Try --group-size 8.")


def grpo(batch: Batch, policy, eps: float = 1e-4, norm_std: bool = True, **kw) -> torch.Tensor:
    """Normalize terminal reward within each prompt's group; broadcast one scalar advantage
    per completion. norm_std=False is the Dr.GRPO advantage. Never normalize across groups."""
    _require_group("grpo", batch)
    R = batch.group_terminal_rewards()  # [n_prompts, group_size]
    centered = R - R.mean(dim=1, keepdim=True)
    if norm_std:
        centered = centered / (R.std(dim=1, keepdim=True) + eps)
    A = centered.reshape(-1)  # [N], one advantage per completion
    return A[:, None] * batch.mask  # broadcast to every completion token


def rloo(batch: Batch, policy, eps: float = 1e-4, **kw) -> torch.Tensor:
    """Leave-one-out: baseline for sample i = mean of the OTHER samples in its group."""
    _require_group("rloo", batch)
    R = batch.group_terminal_rewards()  # [n_prompts, group_size]
    loo = (R.sum(dim=1, keepdim=True) - R) / (batch.group_size - 1)
    A = (R - loo).reshape(-1)
    return A[:, None] * batch.mask


ADVANTAGE_FNS = {
    "reinforce": reinforce,
    "reinforce_baseline": reinforce_baseline,
    "ppo": ppo,
    "grpo": grpo,
    "rloo": rloo,
}
NEEDS_CRITIC = {"ppo"}
NEEDS_GROUP = {"grpo", "rloo"}


def make_advfn(algo: str, cfg=None):
    """Bind the estimator's hyper-parameters from the config."""
    if algo not in ADVANTAGE_FNS:
        raise ValueError(f"unknown algo {algo!r}; choose from {list(ADVANTAGE_FNS)}")
    fn = ADVANTAGE_FNS[algo]
    if cfg is None:
        return fn
    kw = {}
    if algo in ("reinforce", "reinforce_baseline"):
        kw = {"gamma": cfg.gamma}
    elif algo == "ppo":
        kw = {"gamma": cfg.gamma, "lam": cfg.lam}
    elif algo == "grpo":
        kw = {"norm_std": cfg.norm_std}
    return functools.partial(fn, **kw)
