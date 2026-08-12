"""Tasks + verifiable rewards. Every task implements sample(n) and rollout(...); rewards are
a composable list of (fn, weight) callables with fn(prompt, completion, answer) -> float."""
from __future__ import annotations

import ast
import operator
import re
from fractions import Fraction

import torch

from core import Batch, Trajectory


# ========================================================================== #
# Control — CartPole (gymnasium). group_size is always 1.
# ========================================================================== #
class CartPoleTask:
    def __init__(self, max_steps: int = 500, gamma: float = 0.99, rank: int = 0, world: int = 1):
        import gymnasium as gym

        self.env = gym.make("CartPole-v1", max_episode_steps=max_steps)
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.reward_fns: list = []  # control reward is intrinsic to env.step()
        self.gamma = gamma          # for the truncation bootstrap (M1)
        # 1M seeds per rank keeps multi-rank env resets disjoint (eval lives at 10M+).
        self._seed = 1_000_000 * rank

    def sample(self, n: int) -> list[int]:
        seeds = list(range(self._seed, self._seed + n))
        self._seed += n
        return seeds

    def evaluate(self, policy, n: int = 20) -> dict:
        """Greedy episodes on held-out seeds (10M+, never handed out by sample())."""
        returns = []
        for k in range(n):
            obs, _ = self.env.reset(seed=10_000_000 + k)
            done, tot = False, 0.0
            while not done:
                s = torch.tensor(obs, dtype=torch.float32)
                a, _, _ = policy.act(s[None, :], greedy=True)
                obs, r, term, trunc, _ = self.env.step(int(a.item()))
                tot += r
                done = term or trunc
            returns.append(tot)
        return {"return": sum(returns) / len(returns)}

    def rollout(self, policy, prompts: list[int], group_size: int = 1) -> Batch:
        has_value = getattr(policy, "v", None) is not None
        groups = []
        for seed in prompts:
            group = []
            for g in range(group_size):
                # every member of a group starts from the SAME state; the group baseline is
                # only valid when the group shares its "prompt"
                obs, _ = self.env.reset(seed=seed)
                states, actions, logps, rewards = [], [], [], []
                term = trunc = False
                while not (term or trunc):
                    s = torch.tensor(obs, dtype=torch.float32)
                    a, logp, _ = policy.act(s[None, :])
                    obs, r, term, trunc, _ = self.env.step(int(a.item()))
                    states.append(s)
                    actions.append(a.squeeze(0))
                    logps.append(logp.squeeze(0))
                    rewards.append(float(r))
                # truncation is not a true terminal: bootstrap V(s_last) into the final reward
                if trunc and not term and has_value:
                    with torch.no_grad():
                        vlast = policy.act(torch.tensor(obs, dtype=torch.float32)[None, :])[2]
                    rewards[-1] += self.gamma * float(vlast.item())
                T = len(states)
                group.append(Trajectory(
                    states=torch.stack(states),
                    actions=torch.stack(actions).long(),
                    old_logp=torch.stack(logps),
                    rewards=torch.tensor(rewards),
                    mask=torch.ones(T),
                ))
            groups.append(group)
        return Batch.from_groups(groups)


# ========================================================================== #
# Reasoning — verifiable-reward tasks (RLVR)
# ========================================================================== #
# The required format must be one the UNTRAINED model already produces sometimes, or every
# group is uniform and GRPO's advantage is identically 0 (the run trains nothing while
# looking healthy). Hence: free-form reasoning, only the <answer> tag required.
SYSTEM_PROMPT = (
    "You solve arithmetic puzzles. Reason briefly, then put ONLY the final answer inside "
    "<answer> </answer> tags."
)

_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(completion: str) -> str | None:
    m = _ANSWER_RE.findall(completion)
    return m[-1].strip() if m else None


def format_reward(prompt, completion, answer) -> float:
    """1.0 for a well-formed <answer></answer>, 0.5 for opening but never closing it.
    Partial credit adds within-group variance at cold start."""
    if _ANSWER_RE.search(completion):
        return 1.0
    return 0.5 if "<answer>" in completion else 0.0


def think_format_reward(prompt, completion, answer) -> float:
    """Full <think>...</think><answer>...</answer> shape. Not used by default: small models
    never emit it cold, which zeroes every gradient. Useful as a second-stage reward."""
    return 1.0 if _FORMAT_RE.search(completion) else 0.0


# ---- a tiny safe arithmetic evaluator (compiler-as-environment) ----------- #
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def _safe_eval(expr: str) -> Fraction | None:
    """Evaluate an arithmetic expression exactly (Fraction), or None if malformed."""
    try:
        node = ast.parse(expr, mode="eval").body
    except (SyntaxError, ValueError):
        return None

    def ev(n):
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Div) and b == 0:
                return None
            return _OPS[type(n.op)](Fraction(a), Fraction(b))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            v = ev(n.operand)
            return None if v is None else _OPS[type(n.op)](v)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return Fraction(n.value)
        return None

    return ev(node)


def countdown_reward(prompt, completion, answer) -> float:
    """1.0 iff <answer> is an expression using each provided number exactly once and
    evaluating to the target. The calculator IS the reward model — no labels, no RM."""
    ans = extract_answer(completion)
    if ans is None:
        return 0.0
    expr = ans.split("=")[0].strip()   # tolerate "expr = 23": score the LHS only
    if not re.fullmatch(r"[0-9+\-*/() .]+", expr or ""):
        return 0.0
    used = [float(x) for x in re.findall(r"\d+\.?\d*", expr)]   # "3.0" is one number, not [3,0]
    if sorted(used) != sorted(float(n) for n in answer["nums"]):  # each number used exactly once
        return 0.0
    val = _safe_eval(expr)
    return 1.0 if val is not None and val == Fraction(answer["target"]) else 0.0


def gsm8k_reward(prompt, completion, answer) -> float:
    """1.0 iff the final numeric answer matches the gold number."""
    ans = extract_answer(completion)
    if ans is None:
        return 0.0
    nums = re.findall(r"-?\d[\d,]*\.?\d*", ans.replace(",", ""))
    if not nums:
        return 0.0
    try:
        return 1.0 if abs(float(nums[-1]) - float(answer["value"])) < 1e-4 else 0.0
    except ValueError:
        return 0.0


class LLMTask:
    """Shared RLVR plumbing: format prompts -> policy.act -> apply reward list.
    Subclasses load `data`/`eval_data`, define `build_prompt`, and pick `reward_fns`."""

    reward_fns: list = []

    def build_prompt(self, problem) -> str:
        raise NotImplementedError

    # subclasses set: self.data (train pool) and self.eval_data (held-out, never sampled)
    data: list = []
    eval_data: list = []
    rank: int = 0
    world: int = 1
    debug_samples: int = 0      # print this many real (prompt, completion, reward) triples

    def _split(self, rows: list, n_eval: int) -> None:
        """First n_eval rows are the eval holdout; the rest is strided disjointly by rank."""
        self.eval_data = rows[:n_eval]
        self.data = rows[n_eval:][self.rank :: self.world]
        if not self.data:
            raise ValueError(f"n_examples must exceed the eval split ({n_eval})")
        self._i = 0

    def sample(self, n: int) -> list[dict]:
        """The next n training problems, walking the shard as a ring buffer."""
        out = [self.data[(self._i + k) % len(self.data)] for k in range(n)]
        self._i = (self._i + n) % len(self.data)
        return out

    def prompt_text(self, policy, p) -> str:
        return policy.format_prompt(self.build_prompt(p), SYSTEM_PROMPT)

    def rollout(self, policy, prompts: list, group_size: int, max_new_tokens: int = 256,
                temperature: float = 1.0, top_p: float = 1.0) -> Batch:
        texts = [self.prompt_text(policy, p) for p in prompts]
        flat = policy.act(texts, group_size, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p)
        # flat is a list of Trajectory, ordered [prompt0]*G, [prompt1]*G, ...
        groups = []
        for pi, problem in enumerate(prompts):
            group = flat[pi * group_size : (pi + 1) * group_size]
            for tr in group:
                comp_ids = tr.states[tr.mask.bool()]
                completion = policy.tok.decode(comp_ids, skip_special_tokens=True)
                total = 0.0
                for fn, w in self.reward_fns:
                    total += w * fn(texts[pi], completion, problem)
                # terminal reward on the LAST completion token
                last = int(tr.mask.nonzero()[-1].item())
                tr.rewards[last] = total
            groups.append(group)
        batch = Batch.from_groups(groups)
        batch.temperature = temperature if temperature > 0 else 1.0
        if self.debug_samples:
            # print a real completion + reward: "model is bad" and "we parse the wrong thing"
            # look identical in the metrics
            self.debug_samples -= 1
            tr = groups[0][0]
            comp = policy.tok.decode(tr.states[tr.mask.bool()], skip_special_tokens=False)
            print(f"[sample] prompt={texts[0][-200:]!r}\n[sample] completion={comp[:600]!r}\n"
                  f"[sample] reward={float(tr.rewards.sum()):.3f}", flush=True)
        return batch

    def evaluate(self, policy, n: int = 32, max_new_tokens: int = 256) -> dict:
        """Greedy accuracy on the held-out split, sharded across ranks and reduced."""
        from utils import all_sum

        problems = self.eval_data[:n][self.rank :: self.world]
        if problems:
            texts = [self.prompt_text(policy, p) for p in problems]
            # generate-only: eval never needs old_logp, computing it would double the cost
            completions = policy.generate(texts, max_new_tokens=max_new_tokens, temperature=0.0)
            primary = self.reward_fns[0][0]
            hits = sum(primary(t, c, p) for t, c, p in zip(texts, completions, problems))
        else:
            hits = 0.0
        dev = getattr(policy, "device", None)
        tot_hits = float(all_sum(hits, dev))
        tot_n = float(all_sum(len(problems), dev))
        return {"acc": tot_hits / max(tot_n, 1.0)}


class CountdownTask(LLMTask):
    def __init__(self, split: str = "train", n_examples: int = 2000, seed: int = 0,
                 n_eval: int = 128, rank: int = 0, world: int = 1):
        from datasets import load_dataset

        self.rank, self.world = rank, world
        ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n_examples, len(ds))))
        self._split([{"nums": list(ex["nums"]), "target": int(ex["target"])} for ex in ds], n_eval)
        self.reward_fns = [(countdown_reward, 1.0), (format_reward, 0.1)]

    def build_prompt(self, p: dict) -> str:
        return (f"Using the numbers {p['nums']}, each exactly once with +, -, *, / and "
                f"parentheses, write an expression equal to {p['target']}. Reason briefly, then "
                f"put ONLY the expression (no '=') inside <answer> </answer> tags.")


class GSM8KTask(LLMTask):
    def __init__(self, split: str = "train", n_examples: int = 2000, seed: int = 0,
                 n_eval: int = 128, rank: int = 0, world: int = 1):
        from datasets import load_dataset

        self.rank, self.world = rank, world
        ds = load_dataset("openai/gsm8k", "main", split=split)
        ds = ds.shuffle(seed=seed).select(range(min(n_examples, len(ds))))
        rows = []
        for ex in ds:
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            rows.append({"question": ex["question"], "value": float(gold)})
        self._split(rows, n_eval)
        self.reward_fns = [(gsm8k_reward, 1.0), (format_reward, 0.1)]

    def build_prompt(self, p: dict) -> str:
        return (p["question"] + "\nReason briefly, then put ONLY the final number inside "
                "<answer> </answer> tags.")


def make_task(name: str, **kw):
    keep = lambda *names: {k: v for k, v in kw.items() if k in names}  # noqa: E731
    if name == "cartpole":
        return CartPoleTask(**keep("max_steps", "gamma", "rank", "world"))
    if name == "countdown":
        return CountdownTask(**keep("n_examples", "seed", "n_eval", "rank", "world"))
    if name == "gsm8k":
        return GSM8KTask(**keep("n_examples", "seed", "n_eval", "rank", "world"))
    raise ValueError(f"unknown task {name!r}")
