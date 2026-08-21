"""The two data types that flow through the repo: Trajectory (one rollout) and Batch
(right-padded stack, with the [n_prompts, group_size] axis kept explicit for GRPO)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Trajectory:
    """One rollout. For control: a full episode. For an LLM: one completion.

    states   : [T, ...] observations (float) or full token ids (long)
    actions  : [T]      action taken / token generated at each step
    old_logp : [T]      logπ_old(a|s) captured at rollout time (no grad)
    rewards  : [T]      per-step reward (only the terminal step is nonzero for an LLM)
    mask     : [T]      1 on decision steps, 0 on prompt tokens / padding / post-EOS
    truncated: hit the token budget without emitting a stop token — its answer never
               appeared, so it must not be scored like an ordinary wrong answer
    """

    states: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    rewards: torch.Tensor
    mask: torch.Tensor
    truncated: bool = False


def _pad_stack(tensors: list[torch.Tensor], T: int, pad_value=0.0) -> torch.Tensor:
    """Right-pad a list of [t, ...] tensors to [N, T, ...]."""
    out = []
    for x in tensors:
        if x.shape[0] < T:
            pad_shape = (T - x.shape[0], *x.shape[1:])
            pad = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=0)
        out.append(x)
    return torch.stack(out, dim=0)


@dataclass
class Batch:
    """Flattened, right-padded tensors plus the explicit [n_prompts][group_size] axis.

    states/actions/old_logp/rewards/mask are [N, T(, ...)] with N = n_prompts*group_size.
    group_id[i] is the prompt index of flattened trajectory i (the GRPO group key).
    """

    states: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    rewards: torch.Tensor
    mask: torch.Tensor
    group_id: torch.Tensor
    n_prompts: int
    group_size: int
    returns: torch.Tensor | None = None   # critic target, written by the ppo advantage fn
    temperature: float = 1.0              # sampling temp, so logprob recomputes match it
    # [N, T] logprobs as reported by whatever engine actually SAMPLED these tokens. Never
    # old_logp (different kernels, different numerics — that is the bug recompute_old_logp
    # exists to prevent); it is the true behaviour policy, which is what --tis-clip corrects
    # against. None in sync mode, where sampling and scoring share one forward.
    sample_logp: torch.Tensor | None = None
    truncated: torch.Tensor | None = None  # [N] 1.0 where generation hit the token budget

    @classmethod
    def from_groups(cls, groups: list[list[Trajectory]]) -> "Batch":
        n_prompts = len(groups)
        group_size = len(groups[0])
        flat = [tr for g in groups for tr in g]
        T = max(tr.states.shape[0] for tr in flat)
        states = _pad_stack([tr.states for tr in flat], T, 0)
        actions = _pad_stack([tr.actions for tr in flat], T, 0)
        old_logp = _pad_stack([tr.old_logp for tr in flat], T, 0.0)
        rewards = _pad_stack([tr.rewards for tr in flat], T, 0.0)
        mask = _pad_stack([tr.mask for tr in flat], T, 0.0)
        group_id = torch.tensor(
            [p for p in range(n_prompts) for _ in range(group_size)], dtype=torch.long
        )
        b = cls(states, actions, old_logp, rewards, mask, group_id, n_prompts, group_size)
        b.truncated = torch.tensor([float(tr.truncated) for tr in flat])
        return b

    def to(self, device) -> "Batch":
        for f in ("states", "actions", "old_logp", "rewards", "mask", "group_id",
                  "sample_logp", "truncated"):
            x = getattr(self, f)
            if x is not None:
                setattr(self, f, x.to(device))
        return self

    def slice(self, i: int, j: int) -> "Batch":
        """Rows [i:j) as a Batch; group bookkeeping is meaningless on a slice."""
        sub = Batch(self.states[i:j], self.actions[i:j], self.old_logp[i:j],
                    self.rewards[i:j], self.mask[i:j], self.group_id[i:j],
                    n_prompts=j - i, group_size=1, temperature=self.temperature)
        # a chunk that lost sample_logp would silently skip the TIS correction for its tokens
        for f in ("returns", "sample_logp", "truncated"):
            x = getattr(self, f)
            if x is not None:
                setattr(sub, f, x[i:j])
        return sub

    def micro_batches(self, size: int):
        """Yield (start, sub_batch) in chunks of `size` rows. size<=0 means one chunk."""
        n = self.states.shape[0]
        step = n if size is None or size <= 0 else size
        for i in range(0, n, step):
            yield i, self.slice(i, min(i + step, n))

    # ---- convenience views ------------------------------------------------ #
    def terminal_rewards(self) -> torch.Tensor:
        """[N] total reward per trajectory (sum over steps = terminal reward for an LLM)."""
        return (self.rewards * self.mask).sum(-1)

    def group_terminal_rewards(self) -> torch.Tensor:
        """[n_prompts, group_size] terminal reward, grouped — the GRPO baseline input."""
        return self.terminal_rewards().view(self.n_prompts, self.group_size)

    def reward_mean(self) -> float:
        return self.terminal_rewards().mean().item()
