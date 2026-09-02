"""Shared math and plumbing. Every reduction over trajectories is masked (prompt tokens,
padding, and post-EOS steps must never contribute to the loss)."""
from __future__ import annotations

import csv
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
# data-parallel plumbing (no DDP)
# --------------------------------------------------------------------------- #
# Each rank rolls out its own prompt shard (a GRPO group never spans ranks). Only two
# things are global: the loss denominator and the gradient. A SUM all-reduce plus a
# global denominator reproduces the exact single-GPU gradient; see optimize() in train.py.

def dist_init(timeout_s: float = 1800.0) -> tuple[int, int, int]:
    """Join the process group if launched under torchrun. Returns (rank, world, local_rank).
    timeout_s bounds every collective; the async trainer derives it from pop_timeout."""
    from datetime import timedelta
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) == 1:
        return 0, 1, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_s))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return not is_dist() or dist.get_rank() == 0


def all_sum(x: float | torch.Tensor, device=None) -> torch.Tensor:
    """Sum a scalar across ranks (identity when running on one process)."""
    t = x.detach().clone() if torch.is_tensor(x) else torch.tensor(float(x), device=device)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_grads(params) -> None:
    """SUM every rank's gradients in place (must be SUM, not MEAN: see loss_denoms).

    Every rank must reduce EVERY param: a rank whose whole batch was skipped (all groups
    dead — routine on hard tasks) has grad None, and skipping its reduces desyncs the
    NCCL collective sequence across ranks (same SeqNum, different tensors) -> 30-min
    watchdog -> SIGABRT. A zero tensor is that rank's correct contribution."""
    if not is_dist():
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def broadcast_params(module) -> None:
    """Make every rank start from rank 0's weights, so the replicas can never drift."""
    if not is_dist():
        return
    for p in module.state_dict().values():
        if torch.is_tensor(p):
            dist.broadcast(p, src=0)


def dist_cleanup() -> None:
    if is_dist():
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# masked reductions
# --------------------------------------------------------------------------- #
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    """Mean of x over positions where mask==1."""
    mask = mask.to(x.dtype)
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp_min(1.0)
    return (x * mask).sum(dim) / mask.sum(dim).clamp_min(1.0)


def masked_sum(x: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum() if dim is None else (x * mask).sum(dim)


def whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(x - mean) / (std + eps) over masked positions, re-masked so padding stays exactly 0."""
    mask = mask.to(x.dtype)
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return ((x - mean) / (var.sqrt() + eps)) * mask


# --------------------------------------------------------------------------- #
# credit assignment — return-to-go and GAE (right-padded [N, T] tensors)
# --------------------------------------------------------------------------- #
def discounted_returns(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """Return-to-go per row: sum_{k>=t} gamma^{k-t} r_k. Right-padded tails contribute 0."""
    T = rewards.shape[-1]
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in reversed(range(T)):
        running = rewards[..., t] + gamma * running * mask[..., t]
        returns[..., t] = running
    return returns * mask


def gae(rewards, values, mask, gamma: float, lam: float):
    """GAE. Returns (advantages, value_targets = A + V). V beyond the episode is 0."""
    T = rewards.shape[-1]
    adv = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(rewards[..., 0])
    for t in reversed(range(T)):
        next_v = values[..., t + 1] if t + 1 < T else torch.zeros_like(values[..., 0])
        # if the next step is padding (or this is terminal), bootstrap = 0
        next_nonterminal = mask[..., t + 1] if t + 1 < T else torch.zeros_like(mask[..., 0])
        delta = rewards[..., t] + gamma * next_v * next_nonterminal - values[..., t]
        last_adv = delta + gamma * lam * next_nonterminal * last_adv
        adv[..., t] = last_adv
    adv = adv * mask
    return adv, (adv + values) * mask


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
class Logger:
    """Plain prints + an appended long-format CSV (step,time_s,key,value). No wandb."""

    def __init__(self, csv_path: str | None = None, resume: bool = False):
        self.csv_path = csv_path
        self.t0 = time.time()
        self._fh = None
        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            new = not (resume and os.path.exists(csv_path))
            self._fh = open(csv_path, "a" if not new else "w", newline="")
            self._w = csv.writer(self._fh)
            if new:
                self._w.writerow(["step", "time_s", "key", "value"])

    def log(self, step: int, **kv):
        dt = round(time.time() - self.t0, 1)
        print(" | ".join([f"step={step}", f"time_s={dt}"] +
                         [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in kv.items()]), flush=True)
        if self._fh:
            self._w.writerows([[step, dt, k, v] for k, v in kv.items()])
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
