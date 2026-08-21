"""The one training loop: loss = -(advantage * logprob).mean() + ratio clip + optional KL.
Algorithms differ only in advantage(...); sync vs async differs only in where batches
come from (SyncSource / AsyncSource).

    python train.py --task cartpole  --algo reinforce
    python train.py --task countdown --algo grpo --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, fields

import torch

from algos import make_advfn
from tasks import make_task
from utils import (Logger, all_reduce_grads, all_sum, broadcast_params, dist_cleanup,
                   dist_init, is_main, masked_mean, masked_sum, set_seed)


def device_banner(role: str, cfg) -> str:
    """Print the hardware each role landed on (heterogeneous by design)."""
    gpu = "cpu"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = f"{p.name} {p.total_memory / 2**30:.0f}GiB"
    return f"[{role}] {gpu} | dtype={cfg.dtype} | gen_batch={cfg.gen_batch or cfg.micro_batch} " \
           f"micro_batch={cfg.micro_batch}"


def trainable(policy):
    """Params that actually get gradients (under LoRA the frozen base does not)."""
    return [p for p in policy.parameters() if p.requires_grad]


@dataclass
class Config:
    task: str = "cartpole"
    algo: str = "reinforce"
    model: str = ""                 # HF model id for LLM tasks; "" -> MLP for control
    steps: int = 200
    n_prompts: int = 16             # episodes (control) / questions (LLM) per step
    group_size: int = 1             # 1 for control; 8 for GRPO
    inner_epochs: int = 1           # 1 -> vanilla REINFORCE; >1 -> ratio != 1
    lr: float = 2e-3
    gamma: float = 0.99
    lam: float = 0.95
    eps: float = 0.2                # PPO/GRPO clip range (ignored when inner_epochs==1)
    eps_high: float = 0.0           # DAPO clip-higher: raise ONLY the upper bound (try 0.28)
                                    # so low-probability tokens can still grow; 0 -> use eps
    dual_clip: float = 0.0          # cap the A<0 surrogate at dual_clip*A (try 3.0); 0 -> off
    skip_zero_adv: bool = True      # skip micro-batches with zero advantage (exact, see
                                    # optimize) — a whole group that scored uniformly
    kl_coef: float = 0.0            # >0 turns on a frozen reference model
    vf_coef: float = 0.5            # critic loss weight (ppo)
    ent_coef: float = 0.0           # entropy bonus (control exploration)
    max_grad_norm: float = 1.0
    length_norm: bool = True        # per-sequence mean; --no-length-norm = Dr.GRPO aggregator
    norm_std: bool = True           # divide by group std; --no-norm-std = Dr.GRPO advantage
    # LLM knobs
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    overlong_coef: float = 0.0      # weight of the DAPO soft length penalty; 0 -> off
    dtype: str = "bfloat16"         # COMPUTE dtype (autocast); weights stay fp32
    device: str = "cpu"
    lora: bool = False
    lora_r: int = 16
    # comma-separated for argparse; adding MLP projections grows the adapter several-fold
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj"
    grad_ckpt: bool = False         # recompute activations: ~30% slower, much less memory
    think: bool = False             # hybrid-reasoning chat template (Qwen3): long CoT, slow
    debug_samples: int = 0          # print N real completions at the start (new-model triage)
    micro_batch: int = 4            # sequences per SCORING fwd/bwd (logits-bound memory)
    gen_batch: int = 0               # sequences per generate() call; 0 -> same as micro_batch
    # bookkeeping
    seed: int = 0
    eval_every: int = 10
    eval_n: int = 32                # held-out problems / episodes per eval
    eval_k: int = 1                 # samples per eval problem; >1 reports pass@k at temp 1
    ckpt_every: int = 50            # 0 disables checkpointing entirely
    resume: str = ""                # path to a .pt checkpoint to continue from
    # --- disaggregated async RL (see serve.py). role="" is the synchronous default. ---
    role: str = ""                  # "" | trainer | rollout
    vllm: bool = False              # rollout workers generate with vLLM (paged attn) instead
                                    # of HF generate. Trainer always scores with HF.
    vllm_gpu_frac: float = 0.85
    vllm_max_len: int = 2048
    tis_clip: float = 0.0           # truncated importance sampling ceiling (try 2-3); corrects
                                    # the worker ENGINE's numerics against the trainer's. 0 ->
                                    # off. See rl_loss; --role trainer only
    trainer_url: str = ""           # rollout workers: http://trainer-0.<job-group>:8000
    serve_port: int = 8000
    max_staleness: int = 2          # reject batches sampled >N versions behind; also sizes the
                                    # rollout queue ((max_staleness+1) batches) and the snapshot
                                    # retention window — one number, one policy
    pop_timeout: float = 1800.0     # give up waiting for rollouts; must exceed ONE worker's
                                    # full generation cycle, not the average arrival gap
    out: str = "runs/run"
    n_examples: int = 2000


def make_policy(cfg: Config, task):
    if cfg.model:  # LLM
        from model import HFPolicy
        return HFPolicy(cfg.model, device=cfg.device, dtype=cfg.dtype, lora=cfg.lora,
                        lora_r=cfg.lora_r, micro_batch=cfg.micro_batch,
                        gen_batch=cfg.gen_batch, grad_ckpt=cfg.grad_ckpt, think=cfg.think,
                        lora_targets=cfg.lora_targets)
    from model import MLPPolicy
    return MLPPolicy(task.obs_dim, task.n_actions, value_head=(cfg.algo == "ppo")).to(cfg.device)


def make_ref(policy, cfg):
    """KL reference model, None unless kl_coef>0. LoRA needs no copy (the frozen base is
    the reference); full fine-tuning pays for a deepcopy."""
    if cfg.kl_coef <= 0:
        return None
    if getattr(policy, "is_lora", False):
        return "adapter"  # sentinel: use policy.ref_logprobs (disable_adapter)
    if cfg.model:
        print("[warn] --kl-coef>0 without --lora deep-copies the full model onto the same "
              "device (2x weights). Prefer --lora, or keep kl-coef=0 (R1-Zero-style).", flush=True)
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


@torch.no_grad()
def recompute_old_logp(policy, batch, snap: dict, cfg):
    """Recompute old_logp under the weights that sampled this batch, with the trainer's own
    kernels. Worker logprobs (esp. vLLM's) come from different numerics and would bias the
    importance ratio by arithmetic rather than staleness; a run diverged from exactly that.
    """
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    missing = [n for n, _ in named if n not in snap]
    if missing:
        # a partial swap would recompute under a chimera of two versions
        raise RuntimeError(f"sampling snapshot is missing {len(missing)}/{len(named)} trainable "
                           f"tensors (e.g. {missing[0]!r}); trainer/worker naming has drifted")
    backup = {n: p.detach().clone() for n, p in named}
    for n, p in named:                     # swap in the sampling version
        p.copy_(snap[n].to(device=p.device, dtype=p.dtype))
    size = cfg.micro_batch if cfg.model else 0
    old = torch.cat([policy.logprobs(mb).detach().cpu()
                     for _, mb in batch.micro_batches(size)], dim=0)
    for n, p in named:                     # restore the training version
        p.copy_(backup[n])
    return old


def split_fingerprint(cfg: Config) -> str:
    """The fields that determine the train/eval split and task identity. Trainer and workers
    derive the split independently; if these differ, workers TRAIN on the trainer's held-out
    problems and the eval metric silently inflates. The trainer serves this string and every
    worker asserts against it before generating."""
    import json
    keys = ("task", "model", "seed", "n_examples", "eval_n", "think")
    return json.dumps({k: getattr(cfg, k) for k in keys}, sort_keys=True)


def kl_penalty(logp, ref_logp):
    """k3 KL estimator (GRPO): exp(Δ) - Δ - 1 >= 0, per token. Δ = ref_logp - logp."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0


def loss_denoms(batch, cfg) -> dict:
    """Denominators global over micro-batches AND ranks, so chunked/sharded losses sum to
    exactly the single-big-backward loss (per-rank denominators would scale the step by
    world_size)."""
    dev = batch.mask.device
    return {"tokens": all_sum(batch.mask.sum(), dev).clamp_min(1.0),
            "seqs": all_sum(float(batch.mask.shape[0]), dev),
            "const": all_sum(batch.mask.shape[0] * float(cfg.max_new_tokens), dev)}


def rollout_kwargs(cfg) -> dict:
    """The knobs task.rollout() takes. ONE definition on purpose: the trainer builds these in
    sync mode and the worker builds them in async mode, and if they drift the same config
    computes different rewards depending on which process happened to do the rollout."""
    if cfg.task == "cartpole":
        return {}
    return dict(max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                top_p=cfg.top_p, overlong_coef=cfg.overlong_coef)


def batch_stats(batch, cfg) -> dict:
    """Cheap per-batch health metrics (rank 0's shard, so a sample rather than the fleet).

    resp_len is the signature curve of RL-for-reasoning; trunc is what tells you whether
    --max-new-tokens is strangling it (keep it under ~5%); dead is the fraction of groups
    that scored uniformly and so carry no gradient at all — it rises at both ends of
    training and is the honest measure of how much of the batch was wasted.
    """
    out = {"resp_len": round(float(batch.mask.sum(-1).float().mean()), 1)}
    if cfg.model and batch.truncated is not None:   # control episodes end, they don't truncate
        out["trunc"] = round(float(batch.truncated.float().mean()), 3)
    if batch.group_size > 1:
        R = batch.group_terminal_rewards()
        out["dead"] = round(float((R.std(dim=1) < 1e-6).float().mean()), 3)
    return out


def rl_loss(policy, batch, adv, cfg, ref=None, denoms=None):
    """The clipped surrogate (ratio, clip, min, aggregation, value, KL). Works on a
    micro-batch: pass the whole batch's `denoms` and sum the chunk losses."""
    d = denoms if denoms is not None else loss_denoms(batch, cfg)
    logp = policy.logprobs(batch)                        # recompute under current theta
    ratio = torch.exp(logp - batch.old_logp)             # 1 on the first epoch
    hi = cfg.eps_high or cfg.eps                         # DAPO clip-higher when set
    clipped = torch.clamp(ratio, 1 - cfg.eps, 1 + hi)
    surr = torch.min(ratio * adv, clipped * adv)
    if cfg.dual_clip:
        # For A<0 the UNCLIPPED branch is the min, and it runs to -inf as the ratio grows:
        # one stale token can own the whole step. Floor that branch at dual_clip*A.
        surr = torch.where(adv < 0, torch.max(surr, cfg.dual_clip * adv), surr)
    if cfg.tis_clip and batch.sample_logp is not None:
        # old_logp is what the sampling WEIGHTS say under the trainer's kernels; sample_logp is
        # what the engine that actually drew the tokens said. Recomputing fixes the ratio's
        # denominator but not the fact that the tokens came from a slightly different
        # distribution. This is the missing importance weight — detached, and truncated
        # because it is unbounded above and one bad token would otherwise dominate.
        surr = surr * torch.exp(batch.old_logp - batch.sample_logp).clamp(max=cfg.tis_clip)

    if not cfg.model:
        # control: mean over all timesteps (length-weights long episodes, as wanted)
        pg_loss = -masked_sum(surr, batch.mask) / d["tokens"]
    elif cfg.length_norm:
        # per-sequence mean, then mean across completions: equal weight regardless of length
        per_seq = masked_sum(surr, batch.mask, dim=-1) / batch.mask.sum(-1).clamp_min(1.0)
        pg_loss = -per_seq.sum() / d["seqs"]
    else:
        # Dr.GRPO: divide by a FIXED constant, not the batch's padded T
        pg_loss = -masked_sum(surr, batch.mask) / d["const"]
    loss = pg_loss

    if batch.returns is not None:                        # ppo: fit the critic to GAE returns
        v = policy.value(batch.states)
        loss = loss + cfg.vf_coef * masked_sum((v - batch.returns) ** 2, batch.mask) / d["tokens"]
    if cfg.ent_coef and hasattr(policy, "entropy"):
        loss = loss - cfg.ent_coef * masked_sum(policy.entropy(batch), batch.mask) / d["tokens"]
    if ref is not None:
        ref_logp = policy.ref_logprobs(batch) if ref == "adapter" else ref.logprobs(batch)
        kl = masked_sum(kl_penalty(logp, ref_logp.detach()), batch.mask) / d["tokens"]
        loss = loss + cfg.kl_coef * kl

    # Sums (not means) so micro-batch chunks can be added up; optimize() divides once.
    # entropy is E_{a~pi}[-log pi(a)] estimated on the sampled tokens themselves — exactly
    # the policy entropy in expectation, for free, instead of a second [N,T,vocab] softmax.
    # It is the earliest warning of collapse: it falls before the reward does.
    r = ratio.detach()
    diag = {"loss": loss.item(), "pg": pg_loss.item(),
            "ratio_sum": masked_sum(r, batch.mask).item(),
            "clipfrac_sum": masked_sum(((r < 1 - cfg.eps) | (r > 1 + hi)).float(),
                                       batch.mask).item(),
            # how far the sampling policy has drifted from the current one: the number that
            # says whether --max-staleness is set right
            "akl_sum": masked_sum((batch.old_logp - logp).detach(), batch.mask).item(),
            "entropy_sum": masked_sum(-logp.detach(), batch.mask).item()}
    return loss, diag


def optimize(policy, opt, batch, adv, cfg, ref=None):
    """One optimizer step: accumulate over micro-batches, then all-reduce across ranks.

    backward() runs INSIDE the chunk loop so peak memory is O(micro_batch), and gradients
    are SUMmed across ranks against loss_denoms' global denominators, which telescopes to
    exactly the single-GPU gradient. Clipping happens after the all-reduce.

    A chunk whose advantage is identically 0 — every completion in a group scored the same,
    which is most of them at both ends of training — contributes 0 to the loss and 0 to the
    gradient, so skipping it is EXACT, not an approximation: `d` is global and deliberately
    still counts its tokens. Exact only while the loss is pure policy gradient, though; a
    KL, entropy or value term does not vanish with the advantage, hence `free`.
    """
    d = loss_denoms(batch, cfg)
    size = cfg.micro_batch if cfg.model else 0           # control is small; one chunk
    free = ref is None and not cfg.ent_coef and batch.returns is None
    opt.zero_grad(set_to_none=True)
    tot = {"loss": 0.0, "pg": 0.0, "ratio_sum": 0.0, "clipfrac_sum": 0.0,
           "akl_sum": 0.0, "entropy_sum": 0.0}
    skipped, seen = 0, 0.0
    for i, mb in batch.micro_batches(size):
        n = mb.states.shape[0]
        a = adv[i : i + n]
        if cfg.skip_zero_adv and free and not a.any():
            skipped += n
            continue
        loss, diag = rl_loss(policy, mb, a, cfg, ref, denoms=d)
        loss.backward()                                  # frees this chunk's graph now
        seen += float(mb.mask.sum())
        for k in tot:
            tot[k] += diag[k]

    all_reduce_grads(trainable(policy))
    groups = (policy.grad_groups() if hasattr(policy, "grad_groups")
              else {"policy": trainable(policy)})
    gnorm = max(float(torch.nn.utils.clip_grad_norm_(ps, cfg.max_grad_norm))
                for ps in groups.values() if ps)
    opt.step()

    dev = batch.mask.device
    # sums -> global (cross-rank, cross-chunk) per-token means. Divided by the tokens actually
    # SCORED, not d["tokens"]: a skipped chunk contributes no ratio, and dividing it in would
    # report drift that never happened (ratio is the headline async metric — it has to be the
    # mean over the tokens it was measured on).
    den = all_sum(seen, dev).clamp_min(1.0)
    means = {k: round(float(all_sum(tot.pop(f"{k}_sum"), dev) / den), 4)
             for k in ("ratio", "clipfrac", "akl", "entropy")}
    return {**tot, **means, "gnorm": float(gnorm), "skipped": skipped}


def validate(cfg: Config) -> None:
    """Catch impossible combinations up front, with a message that names the fix."""
    import os

    from algos import NEEDS_CRITIC, NEEDS_GROUP
    if cfg.role not in ("", "trainer", "rollout"):
        raise ValueError(f"--role must be '', 'trainer' or 'rollout' (got {cfg.role!r})")
    if cfg.vllm and cfg.role != "rollout":
        raise ValueError("--vllm applies to --role rollout only; the trainer scores with HF "
                         "(and must, or old_logp and logp would use different kernels)")
    if cfg.vllm and not cfg.lora:
        raise ValueError("--vllm requires --lora: weights reach vLLM as a PEFT adapter "
                         "(full fine-tuning would ship the whole model every step).")
    if cfg.tis_clip and cfg.role != "trainer":
        raise ValueError("--tis-clip corrects the sampling engine's numerics against the "
                         "trainer's, so it needs the worker logprobs that only --role trainer "
                         "receives. In sync mode both come from one forward and the weight "
                         "would be exactly 1 — a silent no-op.")
    if cfg.tis_clip and cfg.tis_clip < 1.0:
        raise ValueError(f"--tis-clip {cfg.tis_clip} is a CEILING on an importance ratio and "
                         f"must be >= 1 (2-3 is usual); 0 disables it.")
    if cfg.dual_clip and cfg.dual_clip <= 1.0:
        raise ValueError(f"--dual-clip {cfg.dual_clip} must be > 1: it floors the A<0 "
                         f"surrogate at dual_clip*A, and <=1 would clip inside the trust "
                         f"region. 3.0 is usual; 0 disables it.")
    if cfg.eps_high and cfg.eps_high < cfg.eps:
        raise ValueError(f"--eps-high {cfg.eps_high} < --eps {cfg.eps}: clip-higher RAISES the "
                         f"upper bound (try 0.28 against eps 0.2); 0 makes the clip symmetric.")
    if cfg.eval_k < 1:
        raise ValueError(f"--eval-k must be >= 1 (got {cfg.eval_k}); 1 is greedy pass@1.")
    if cfg.ent_coef and cfg.model:
        raise ValueError("--ent-coef only applies to control (MLP) policies; HFPolicy has no "
                         "entropy() and the bonus would be silently skipped.")
    if cfg.role == "rollout" and not cfg.trainer_url:
        raise ValueError("--role rollout needs --trainer-url http://<trainer-host>:<port>")
    if cfg.algo in NEEDS_CRITIC and cfg.model:
        raise ValueError(
            f"--algo {cfg.algo} needs a critic, which HFPolicy does not have (RLVR drops the "
            f"value net). Use --algo grpo or rloo for LLM tasks.")
    if cfg.algo in NEEDS_GROUP and cfg.group_size < 2:
        raise ValueError(f"--algo {cfg.algo} needs --group-size >= 2 (got {cfg.group_size}).")
    if cfg.task != "cartpole" and not cfg.model:
        raise ValueError(f"--task {cfg.task} is an LLM task; pass --model <hf-id>.")
    if cfg.role and not os.environ.get("NANORL_TOKEN"):
        raise ValueError(
            "async roles require NANORL_TOKEN (same value on trainer and workers): the "
            "trainer's HTTP endpoint serves weights and ingests training batches on 0.0.0.0 "
            "and must not run unauthenticated.")


# --------------------------------------------------------------------------- #
# Where batches come from: the only structural difference between sync and async.
# --------------------------------------------------------------------------- #
class SyncSource:
    """Generate on the trainer itself, in lockstep: staleness 0, ratio exactly 1."""

    def __init__(self, task, cfg, roll_kw, n_prompts):
        self.task, self.cfg, self.roll_kw, self.n = task, cfg, roll_kw, n_prompts

    def next_batch(self, policy):
        prompts = self.task.sample(self.n)
        return self.task.rollout(policy, prompts, self.cfg.group_size, **self.roll_kw), 0

    def publish(self, policy):
        pass

    def stats(self):
        return {}


class AsyncSource:
    """Batches arrive from remote rollout workers; weights are published back to them.

    Multi-rank: rank 0 runs the HTTP server, pops `world` batches per step and scatters one
    per rank. Snapshots need no scattering: ranks apply identical all-reduced updates, so
    every rank's adapters at version v match rank 0's.
    """

    def __init__(self, cfg, rank: int = 0, world: int = 1):
        from serve import TrainerServer
        self.cfg, self.rank, self.world = cfg, rank, world
        # One number drives the whole staleness policy: the queue holds (max_staleness+1)
        # batches PER CONSUMED-BATCH-PER-STEP (a world-rank trainer pops `world` per step,
        # and the fleet delivers in bursts), and snapshots outlive every acceptable version.
        self.keep = cfg.max_staleness + 2
        self.server = (TrainerServer(port=cfg.serve_port,
                                     queue_max=(cfg.max_staleness + 1) * world,
                                     meta=split_fingerprint(cfg))
                       if rank == 0 else None)
        self.local_snaps: dict = {}       # version -> adapter state, mirrored on every rank
        self.version = 0
        self.snap_dropped = 0             # batches rejected for lacking a sampling snapshot

    def _pop_fresh(self, timeout: float):
        """Pop until a batch whose sampling version has a live snapshot. A batch without one
        could only be trained by trusting worker logprobs — the exact failure
        recompute_old_logp exists to prevent — so it is dropped, never trained on."""
        import time
        deadline = time.time() + timeout
        while True:
            batch, lag = self.server.pop(self.cfg.max_staleness,
                                         timeout=max(0.0, deadline - time.time()))
            if (self.version - lag) in self.local_snaps:
                return batch, lag
            self.snap_dropped += 1
            print(f"[warn] dropped batch sampled at v{self.version - lag}: no snapshot to "
                  f"recompute old_logp under (would have to trust worker logprobs)", flush=True)

    def next_batch(self, policy):
        if self.world > 1:
            import torch.distributed as dist
            payload = [None] * self.world
            if self.rank == 0:            # rank 0 owns the queue; one batch per rank per step
                try:
                    payload = [tuple(self._pop_fresh(self.cfg.pop_timeout))
                               for _ in range(self.world)]
                except TimeoutError:
                    pass                  # broadcast the Nones so every rank fails together
                                          # instead of ranks 1..N hanging in the collective
            dist.broadcast_object_list(payload, src=0)
            if payload[self.rank] is None:
                raise TimeoutError(f"rank 0 got no fresh rollouts within {self.cfg.pop_timeout}s")
            batch, lag = payload[self.rank]
        else:
            batch, lag = self._pop_fresh(self.cfg.pop_timeout)

        # ALWAYS recompute old_logp under the sampling version's weights; worker logprobs
        # are kept only to measure the kernel/architecture gap (logp_gap in the log).
        # _pop_fresh guarantees the snapshot exists (snapshot dicts are identical on every
        # rank: all ranks apply the same all-reduced updates and publish in lockstep).
        snap = self.local_snaps.get(self.version - lag)
        if snap is None:
            raise RuntimeError(f"no snapshot for sampled version {self.version - lag} after "
                               f"_pop_fresh accepted it — rank snapshot dicts have diverged")
        batch = batch.to(self.cfg.device)
        exact = recompute_old_logp(policy, batch, snap, self.cfg).to(batch.old_logp.device)
        m = batch.mask.bool().cpu()
        self.mismatch = (float((exact.cpu()[m] - batch.old_logp.cpu()[m]).abs().mean())
                         if m.any() else 0.0)
        # the worker's own logprobs are the BEHAVIOUR policy — never old_logp, but the
        # numerator --tis-clip needs to correct for having sampled from a different engine
        batch.sample_logp = batch.old_logp
        batch.old_logp = exact.to(batch.mask.device)
        return batch, lag

    def stats(self):
        base = self.server.stats() if self.server is not None else {}
        return {**base, "snap_dropped": self.snap_dropped,
                "logp_gap": round(getattr(self, "mismatch", 0.0), 4)}

    def publish(self, policy):
        # every rank snapshots locally; only rank 0 serves the outside world
        from serve import adapter_state
        self.version += 1
        self.local_snaps[self.version] = adapter_state(policy)
        for v in [v for v in self.local_snaps if v <= self.version - self.keep]:
            del self.local_snaps[v]
        if self.server is not None:
            self.server.publish(policy)


def rollout_worker(cfg: Config):
    """`--role rollout`: pull weights, generate, submit, repeat. Never computes a gradient;
    a dying worker costs its in-flight batch and nothing else."""
    from serve import RolloutClient

    import os

    rank, world, local = dist_init()
    if world > 1:
        cfg.device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    # SkyPilot sets no RANK without torchrun; fold the node rank in or every worker would
    # sample the SAME prompts
    node_rank = int(os.environ.get("SKYPILOT_NODE_RANK", 0))
    num_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", 1))
    wid, wtot = node_rank * world + rank, num_nodes * world
    set_seed(cfg.seed + 1000 + wid)      # workers must not duplicate each other's prompts
    # n_eval must match the trainer's, or workers train on its held-out problems
    task = make_task(cfg.task, n_examples=cfg.n_examples, seed=cfg.seed, gamma=cfg.gamma,
                     n_eval=cfg.eval_n, rank=wid, world=wtot)
    if cfg.vllm:
        from model import VLLMGenerator
        policy = VLLMGenerator(cfg.model, dtype=cfg.dtype, lora_r=cfg.lora_r,
                               gpu_frac=cfg.vllm_gpu_frac, max_len=cfg.vllm_max_len,
                               think=cfg.think)
    else:
        policy = make_policy(cfg, task)
        policy.eval()
    client = RolloutClient(cfg.trainer_url)
    print(device_banner(f"rollout w{wid}/{wtot}", cfg), flush=True)
    client.wait_for_trainer()
    # A worker with a different seed/n_examples/eval_n derives a DIFFERENT split and trains
    # on the trainer's held-out problems — silently inflating eval. Fail loudly instead.
    mine, theirs = split_fingerprint(cfg), client.fetch_config()
    if theirs and theirs != mine:
        raise RuntimeError(f"worker/trainer config mismatch — the data split would differ.\n"
                           f"  trainer: {theirs}\n  worker : {mine}")
    roll_kw = rollout_kwargs(cfg)

    import urllib.error

    version, sent, misses = 0, 0, 0
    while version < cfg.steps:           # trainer bumps the version once per training step
        try:
            if cfg.vllm:
                # vLLM can only take LoRA from a path, so pull PEFT's own on-disk format.
                new_ver, path = client.pull_adapter(version, "/tmp/nanorl_adapters")
                if path is not None:
                    policy.set_adapter(path, new_ver)
                    version = new_ver
            else:
                new_ver, sd = client.pull_weights(version)
                if sd is not None:
                    expected = {n for n, p in policy.named_parameters() if p.requires_grad}
                    if set(sd) != expected:
                        # strict=False would "succeed" while loading nothing; the worker would
                        # then generate from frozen weights forever, silently, all run long.
                        raise RuntimeError(
                            f"weight pull mismatch: trainer sent {len(sd)} tensors, worker has "
                            f"{len(expected)} trainable — key naming has drifted between roles")
                    policy.load_state_dict(sd, strict=False)   # adapters only under LoRA
                    version = new_ver
            prompts = task.sample(cfg.n_prompts)   # per-worker batch; data already sharded
            with torch.no_grad():
                batch = task.rollout(policy, prompts, cfg.group_size, **roll_kw)
            if str(cfg.device).startswith("cuda"):
                torch.cuda.empty_cache()
            client.submit(version, batch)
            misses, sent = 0, sent + 1
            print(f"[rollout w{wid}] sent batch {sent} @ policy v{version} "
                  f"reward={batch.reward_mean():.3f}", flush=True)
        except (urllib.error.URLError, OSError, ConnectionError) as e:
            # the trainer finishing (server gone mid-request) is the NORMAL way this ends;
            # treat a few consecutive failures as "training is over", not a crash
            misses += 1
            print(f"[rollout w{wid}] trainer unreachable ({type(e).__name__}) "
                  f"{misses}/3", flush=True)
            if misses >= 3:
                print(f"[rollout w{wid}] trainer gone; exiting after {sent} batches", flush=True)
                break
            import time
            time.sleep(2.0)
    dist_cleanup()


def train(cfg: Config):
    validate(cfg)                        # both roles: workers must fail as loudly as trainers
    if cfg.role == "rollout":
        return rollout_worker(cfg)
    import os
    # async collectives legitimately wait while rank 0 pops `world` batches; the process-group
    # timeout must dominate that
    world_env = int(os.environ.get("WORLD_SIZE", 1))
    timeout_s = cfg.pop_timeout * world_env + 600 if cfg.role == "trainer" else 1800.0
    rank, world, local = dist_init(timeout_s=timeout_s)
    if world > 1:
        cfg.device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    # ranks differ in data (per-rank seed) but agree on weights (broadcast_params below)
    set_seed(cfg.seed + rank)
    # n_prompts is the GLOBAL batch, identical meaning on 1 GPU and on 8
    if cfg.role != "trainer" and cfg.n_prompts % world:
        raise ValueError(f"--n-prompts {cfg.n_prompts} must be divisible by world size {world}")
    local_prompts = cfg.n_prompts // world

    task = make_task(cfg.task, n_examples=cfg.n_examples, seed=cfg.seed, gamma=cfg.gamma,
                     n_eval=cfg.eval_n, rank=rank, world=world)
    if cfg.debug_samples and is_main():
        task.debug_samples = cfg.debug_samples
    policy = make_policy(cfg, task)
    broadcast_params(policy)                             # every replica starts identical
    advfn = make_advfn(cfg.algo, cfg)
    ref = make_ref(policy, cfg)
    opt = torch.optim.AdamW(trainable(policy), lr=cfg.lr)
    start = load(cfg.resume, policy, opt) if cfg.resume else 0
    log = Logger(csv_path=cfg.out + ".csv", resume=start > 0) if is_main() else None
    if is_main() and world > 1:
        if cfg.role == "trainer":
            # async: each rank consumes a WHOLE worker submission, so n_prompts is per-batch
            print(f"[dist] {world} trainer ranks x {cfg.n_prompts} prompts x {cfg.group_size} "
                  f"completions = {world * cfg.n_prompts * cfg.group_size} sequences/step "
                  f"(one worker batch per rank)", flush=True)
        else:
            print(f"[dist] {world} ranks x {local_prompts} prompts x {cfg.group_size} "
                  f"completions = {cfg.n_prompts * cfg.group_size} sequences/step", flush=True)

    roll_kw = rollout_kwargs(cfg)
    eval_kw = {"k": cfg.eval_k} if cfg.model else {}    # pass@k is an LLM-task notion
    # the one-line fork between sync and async
    source = (AsyncSource(cfg, rank, world) if cfg.role == "trainer"
              else SyncSource(task, cfg, roll_kw, local_prompts))
    if is_main():
        print(device_banner(cfg.role or "sync", cfg), flush=True)
    source.publish(policy)                   # v1 must exist before any worker asks

    for step in range(start, cfg.steps):
        # 1 — rollout (no grad)
        batch, staleness = source.next_batch(policy)
        batch = batch.to(cfg.device)
        # release generation's cached blocks or the update's [micro_batch, T, vocab] logits
        # may not fit
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()

        # 2 — advantage: the only thing that varies across algorithms. No communication:
        # a GRPO group never spans ranks.
        with torch.no_grad():
            adv = advfn(batch, policy)                       # [N, T], same shape as mask

        # 3 — optimize: mu inner epochs
        for _ in range(cfg.inner_epochs):
            diag = optimize(policy, opt, batch, adv, cfg, ref)

        # 4 — publish: no-op for sync; the next version workers pull for async
        source.publish(policy)

        rew = float(all_sum(batch.reward_mean(), cfg.device)) / world   # mean over all ranks
        if log:
            log.log(step, reward=rew, staleness=staleness, **diag,
                    **batch_stats(batch, cfg), **source.stats())
        if cfg.eval_every and step % cfg.eval_every == 0 and hasattr(task, "evaluate"):
            ev = task.evaluate(policy, n=cfg.eval_n, **eval_kw)   # sharded + reduced inside
            if log:
                log.log(step, **{f"eval_{k}": v for k, v in ev.items()})
        if cfg.ckpt_every and step and step % cfg.ckpt_every == 0 and is_main():
            save(policy, opt, step, cfg)

    if cfg.ckpt_every and is_main():                         # --ckpt-every 0 means NO writes
        save(policy, opt, cfg.steps, cfg)
    if log:
        log.close()
    dist_cleanup()
    return policy


def save(policy, opt, step, cfg):
    import os
    from serve import adapter_state
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    path = f"{cfg.out}_step{step}.pt"
    # Under LoRA the frozen base is reproducible from the HF hub; checkpointing it would make
    # every save ~100x larger (16GB+ for an 8B model) for zero information.
    sd = adapter_state(policy) if getattr(policy, "is_lora", False) else policy.state_dict()
    torch.save({"model": sd, "opt": opt.state_dict(), "step": step, "cfg": vars(cfg)}, path)
    print(f"[ckpt] {path}", flush=True)


def load(path: str, policy, opt) -> int:
    """Restore model+optimizer and return the step to resume from."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    res = policy.load_state_dict(ck["model"], strict=False)     # adapter-only under LoRA
    lost = {n for n, p in policy.named_parameters() if p.requires_grad} & set(res.missing_keys)
    if lost:
        raise RuntimeError(f"checkpoint is missing {len(lost)} trainable tensors "
                           f"(e.g. {next(iter(lost))!r}) — wrong --lora/--model for this ckpt?")
    opt.load_state_dict(ck["opt"])
    print(f"[resume] {path} @ step {ck['step']}", flush=True)
    return int(ck["step"])


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML preset (CLI flags override it)")
    # YAML preset becomes the defaults; explicit CLI flags then override.
    pre, _ = p.parse_known_args()
    defaults = {}
    if pre.config:
        import yaml
        with open(pre.config) as fh:
            defaults = yaml.safe_load(fh) or {}
        unknown = set(defaults) - {f.name for f in fields(Config)}
        if unknown:
            # A typo'd key would otherwise silently run with the default value.
            raise SystemExit(f"unknown config keys in {pre.config}: {sorted(unknown)}")
    for f in fields(Config):
        default = defaults.get(f.name, f.default)
        if isinstance(f.default, bool):
            # BooleanOptionalAction gives --flag / --no-flag, so default-True flags are settable
            p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name,
                           action=argparse.BooleanOptionalAction, default=default)
        else:
            typ = {int: int, float: float, str: str}[type(f.default)]
            p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, type=typ, default=default)
    ns = vars(p.parse_args())
    ns.pop("config", None)
    return Config(**ns)


if __name__ == "__main__":
    train(parse_args())
