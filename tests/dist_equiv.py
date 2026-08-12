"""tests/dist_equiv.py — data-parallel must be numerically identical to single-process.

Run directly (the test suite shells out to it):
    python tests/dist_equiv.py                       # 1 process, writes the reference
    torchrun --nproc_per_node=2 tests/dist_equiv.py  # 2 processes, compares against it

The claim under test is the one that makes multi-GPU nanoRL correct: with a GLOBAL loss
denominator (summed across ranks) and a SUM all-reduce of gradients, W ranks each holding a
shard produce exactly the gradient one process would produce on the whole batch. Get either
half wrong — per-rank denominators, or a MEAN all-reduce — and the effective learning rate
silently scales with the GPU count.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Batch, Trajectory  # noqa: E402
from model import MLPPolicy  # noqa: E402
from train import Config, optimize  # noqa: E402
from utils import dist_cleanup, dist_init  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dist_ref.pt")
N_SEQ = 8


def make_batch(lo, hi):
    """Deterministic trajectories [lo, hi) — the same data however it is sharded."""
    trs = []
    for i in range(lo, hi):
        g = torch.Generator().manual_seed(100 + i)
        T = 2 + i % 3
        trs.append(Trajectory(torch.randn(T, 2, generator=g), torch.randint(0, 3, (T,), generator=g),
                              torch.zeros(T), torch.ones(T), torch.ones(T)))
    b = Batch.from_groups([[t] for t in trs])
    adv = torch.cat([torch.full((1, b.mask.shape[1]), float(i)) for i in range(lo, hi)])
    return b, adv * b.mask


def main():
    rank, world, _ = dist_init()
    per = N_SEQ // world
    batch, adv = make_batch(rank * per, (rank + 1) * per)

    torch.manual_seed(1)                       # identical init on every rank
    policy = MLPPolicy(2, 3)
    opt = torch.optim.SGD(policy.parameters(), lr=0.0)   # lr=0 -> compare grads, not updates
    cfg = Config(model="llm", micro_batch=3, max_new_tokens=4, max_grad_norm=1e9)
    optimize(policy, opt, batch, adv, cfg)
    grads = [p.grad.clone() for p in policy.parameters()]

    if world == 1:
        torch.save(grads, REF)
        print("wrote reference gradients")
    else:
        ref = torch.load(REF, weights_only=False)
        ok = all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ref, grads))
        worst = max(float((a - b).abs().max()) for a, b in zip(ref, grads))
        if rank == 0:
            print(f"world={world} max|grad diff| = {worst:.3e} -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(1)
    dist_cleanup()


if __name__ == "__main__":
    main()
