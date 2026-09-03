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


def overlong_penalty(n_gen: int, budget: int, cache: int = 0) -> float:
    """DAPO's soft overlong punishment: free until the last `cache` tokens of the budget
    (default a quarter of it), then linear to -1.0 at the budget itself.

    Without it a completion cut off mid-sentence scores exactly like a finished wrong answer,
    so nothing ever teaches the model to wrap up — the length just grows until every rollout
    is truncated. Graded rather than a flat penalty on truncation: the gradient has to point
    somewhere before the cliff, not only at it.
    """
    cache = cache or max(1, budget // 4)
    over = n_gen - (budget - cache)
    return 0.0 if over <= 0 else -min(over / cache, 1.0)


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


_FRAC_RE = re.compile(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}")


def _num_value(s: str) -> Fraction | None:
    """Parse a plain number, a/b, \\frac{a}{b} or N% into an exact Fraction; None otherwise."""
    s = s.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace(" ", "").replace(",", "").rstrip(".")
    m = re.fullmatch(r"\\boxed\{(.*)\}", s)      # tolerate a model echoing \boxed{}
    if m:
        s = m.group(1)
    m = _FRAC_RE.fullmatch(s) or re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", s)
    try:
        if m:
            return Fraction(m.group(1)) / Fraction(m.group(2))
        if s.endswith("%"):
            return Fraction(s[:-1]) / 100
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def math_reward(prompt, completion, answer) -> float:
    """1.0 iff <answer> parses to exactly the gold value. Rows are pre-filtered to numeric
    golds, so Fraction equality is the whole verifier — no CAS."""
    ans = extract_answer(completion)
    if ans is None:
        return 0.0
    got = _num_value(ans)
    return 1.0 if got is not None and got == answer["value"] else 0.0


class LLMTask:
    """Shared RLVR plumbing: format prompts -> policy.act -> apply reward list.
    Subclasses load `data`/`eval_data`, define `build_prompt`, and pick `reward_fns`."""

    reward_fns: list = []
    system_prompt: str = SYSTEM_PROMPT   # overridden by tasks that want no <answer> tags

    def build_prompt(self, problem) -> str:
        raise NotImplementedError

    # subclasses set: self.data (train pool) and self.eval_data (held-out, never sampled)
    data: list = []
    eval_data: list = []
    rank: int = 0
    world: int = 1
    debug_samples: int = 0      # print this many real (prompt, completion, reward) triples

    adapt_sample: bool = False  # weight sampling toward prompts whose groups still spread
    overlong_filter: bool = False  # DAPO-style: no gradient through truncated sequences

    def _split(self, rows: list, n_eval: int, seed: int = 0) -> None:
        """First n_eval rows are the eval holdout; the rest is strided disjointly by rank.

        Callers pass rows in a FIXED-seed order so every training seed evaluates on the
        SAME problems (seed-varying holdouts made cross-seed comparisons meaningless —
        measured baselines ranged 0.453-0.578 across seeds on "the" eval set). Only the
        train remainder is reshuffled by the training seed."""
        import random as _random
        self.eval_data = rows[:n_eval]
        rest = rows[n_eval:]
        _random.Random(seed).shuffle(rest)
        self.data = rest[self.rank :: self.world]
        if not self.data:
            raise ValueError(f"n_examples must exceed the eval split ({n_eval})")
        self._i = 0
        for j, r in enumerate(self.data):
            r["_i"] = j
        # GRESO-lite (arXiv 2506.02177): EMA of each prompt's group-reward spread. A group
        # that scores uniformly (dead) yields zero advantage — its generation FLOPs bought
        # nothing — and on a bimodal-difficulty pool most groups are dead (0.92 measured on
        # DeepScaleR at G=4). Start optimistic so every prompt gets tried.
        self._prio = [1.0] * len(self.data)
        import random as _random
        self._rng = _random.Random(1234 + self.rank)

    def sample(self, n: int) -> list[dict]:
        """The next n training problems: a ring walk, or (adapt_sample) a draw weighted by
        each prompt's EMA spread + a floor so hard prompts are revisited as the policy moves."""
        if self.adapt_sample:
            w = [p + 0.05 for p in self._prio]
            idx = self._rng.choices(range(len(self.data)), weights=w, k=n)
            return [self.data[j] for j in idx]
        out = [self.data[(self._i + k) % len(self.data)] for k in range(n)]
        self._i = (self._i + n) % len(self.data)
        return out

    def _update_prio(self, problem: dict, rewards: list[float]) -> None:
        if not self.adapt_sample or "_i" not in problem:
            return
        spread = float(max(rewards) - min(rewards) > 1e-6)
        j = problem["_i"]
        self._prio[j] = 0.7 * self._prio[j] + 0.3 * spread

    def prompt_text(self, policy, p) -> str:
        return policy.format_prompt(self.build_prompt(p), self.system_prompt)

    def rollout(self, policy, prompts: list, group_size: int, max_new_tokens: int = 256,
                temperature: float = 1.0, top_p: float = 1.0,
                overlong_coef: float = 0.0) -> Batch:
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
                if overlong_coef:   # length shaping is a property of the rollout, not the text
                    total += overlong_coef * overlong_penalty(int(tr.mask.sum()), max_new_tokens)
                # terminal reward on the LAST completion token
                last = int(tr.mask.nonzero()[-1].item())
                tr.rewards[last] = total
                # DAPO's overlong FILTERING (not the soft penalty): a truncated sequence
                # contributes no gradient. The penalty variant makes short outputs strictly
                # safe and turns a high-lr transient into terminal length collapse
                # (measured: resp_len 1300->10, eval 0.55->0.18); with mask zeroed the seq
                # reads as reward 0 in the group baseline — closer to truth than -1.
                if self.overlong_filter and tr.truncated:
                    tr.mask.zero_()
            self._update_prio(problem, [float(tr.rewards.sum()) for tr in group])
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

    def evaluate(self, policy, n: int = 32, max_new_tokens: int = 256, k: int = 1) -> dict:
        """Greedy accuracy on the held-out split, sharded across ranks and reduced.

        k>1 switches to pass@k: k samples per problem at temperature 1, reporting the
        per-sample rate (acc) alongside the fraction of problems solved at least once
        (pass_k). Greedy pass@1 cannot separate "the policy learned something" from "the
        policy sharpened onto what it could already do" — a rising acc with a flat pass_k is
        the second one, and it is the usual reason a run plateaus.
        """
        from utils import all_sum

        problems = self.eval_data[:n][self.rank :: self.world]
        hits = solved = 0.0
        if problems:
            texts = [self.prompt_text(policy, p) for p in problems]
            # generate-only: eval never needs old_logp, computing it would double the cost
            completions = policy.generate([t for t in texts for _ in range(k)],
                                          max_new_tokens=max_new_tokens,
                                          temperature=1.0 if k > 1 else 0.0)
            primary = self.reward_fns[0][0]
            for i, (t, p) in enumerate(zip(texts, problems)):
                got = [primary(t, c, p) for c in completions[i * k : (i + 1) * k]]
                hits += sum(got)
                solved += float(any(g > 0 for g in got))
        dev = getattr(policy, "device", None)
        tot_hits = float(all_sum(hits, dev))
        tot_n = float(all_sum(len(problems), dev))
        out = {"acc": tot_hits / max(tot_n * k, 1.0)}
        if k > 1:
            out["pass_k"] = float(all_sum(solved, dev)) / max(tot_n, 1.0)
        return out


# ========================================================================== #
# Instruction following — verifiable constraints on real prompts (RLVR)
# ========================================================================== #
# Prompts come from allenai/RLVR-IFeval (Tulu 3); the constraints are attached here so their
# difficulty is a knob rather than a property of the row. Each is checked by a couple of lines
# of Python — the interpreter is the reward model, exactly as the calculator is for Countdown.
#
# Why THREE constraints and a graded reward, when the dataset ships one per row: a 0.6B model
# is deterministic per prompt. Measured on Qwen3-0.6B, one binary constraint leaves 58% of
# groups scoring identically — no advantage, no gradient — while three with partial credit
# leaves 17%. K is the difficulty lever: raise it for a denser signal, lower it for a harder
# task.
IFEVAL_K = 3

def _n_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


# (instruction shown to the model, verifier). Two properties are load-bearing.
#
# MUTUALLY SATISFIABLE: any K of these can hold at once, or some prompts are unwinnable by
# construction. That is why there is exactly one length constraint, and why the phrase/word
# checks are case-insensitive (they have to survive "write in capitals").
#
# SPREAD ACROSS DIFFICULTY: measured per-constraint on Qwen3-0.6B, shown below. A constraint
# the model already satisfies is paid for free and contributes no gradient — an earlier pool
# of five such (p>=0.91) started training at 0.693 with almost nothing left to learn.
IFEVAL_CONSTRAINTS: list[tuple] = [
    ("Answer with around 40 words.",                                              # p=0.09
     lambda c: abs(_n_words(c) - 40) <= 8),
    ('Use the word "however" exactly twice.',                                     # p~0.26
     lambda c: len(re.findall(r"\bhowever\b", c.lower())) == 2),
    ("Use exactly two bullet points, each on its own line starting with '- '.",   # p~0.27
     lambda c: len(re.findall(r"^\s*-\s+\S", c, re.MULTILINE)) == 2),
    ('Never use the words "the" or "and".',                                       # p=0.33
     lambda c: not re.search(r"\b(the|and)\b", c.lower())),
    ("Write exactly 3 sentences.",                                                # p~0.46
     lambda c: len([s for s in re.split(r"(?<=[.!?])\s+", c.strip()) if s.strip()]) == 3),
    ("Begin your response with the word Yes or the word No.",                     # p~0.5
     lambda c: bool(re.match(r"\W*(yes|no)\b", c.strip().lower()))),
    ("Do not use any commas.",                                                    # p=0.67
     lambda c: c.strip() != "" and "," not in c),
    ("Write your entire response in capital letters.",                            # p~0.75
     lambda c: c.strip() != "" and c == c.upper()),
    ("Finish with the exact phrase: Hope this helps.",                            # p=0.80
     lambda c: c.strip().rstrip('".').lower().endswith("hope this helps")),
]


def ifeval_reward(prompt, completion, answer) -> float:
    """Fraction of this prompt's constraints that hold. Graded on purpose: a binary
    all-or-nothing check collapses the within-group variance GRPO needs."""
    ids = answer["cids"]
    return sum(float(bool(IFEVAL_CONSTRAINTS[i][1](completion))) for i in ids) / len(ids)


def ifeval_strict_reward(prompt, completion, answer) -> float:
    """IFEval's own metric: every constraint or nothing. Reported, never trained on."""
    return float(ifeval_reward(prompt, completion, answer) == 1.0)


class IFEvalTask(LLMTask):
    """Real instruction-following text, scored by code. No reward model, no judge."""

    # the shared prompt asks for <answer> tags, which would fight constraints like
    # "wrap your entire response in quotation marks"
    system_prompt = ("You are a helpful assistant. Follow every formatting constraint in the "
                     "request exactly.")

    def __init__(self, n_examples: int = 2000, seed: int = 0, n_eval: int = 128,
                 rank: int = 0, world: int = 1):
        import random

        from datasets import load_dataset

        self.rank, self.world = rank, world
        ds = load_dataset("allenai/RLVR-IFeval", split="train")
        rng = random.Random(1234)  # pool, constraints and eval holdout fixed across seeds
        rows, seen = [], set()
        for ex in ds:
            # Strip the row's own constraint so only OUR constraints are asked for and scored.
            # `constraint` is a TEMPLATE ("...the word {word}..."), so it matches the prompt
            # verbatim on only a third of rows; the rest are skipped rather than half-stripped,
            # which would leave an unscored — and sometimes contradictory — extra instruction
            # in the prompt. Slicing at the ends keeps the stem a whole sentence.
            prompt, con = ex["messages"][0]["content"], ex["constraint"]
            i = prompt.find(con)
            if i < 0:
                continue
            stem = (prompt[i + len(con):] if i == 0 else prompt[:i]).strip()
            # Dedup is load-bearing, not tidiness: the same Tulu prompt appears under several
            # different constraints (2,347 rows are only 816 distinct stems), so without this
            # the eval holdout and the training shard share prompts and eval silently inflates.
            # bound in CHARS so loading stays cheap: this holds prompts near 42 tokens (p90 77)
            if 40 < len(stem) < 400 and stem not in seen:
                seen.add(stem)
                rows.append(stem)
            if len(rows) >= n_examples:
                break
        rng.shuffle(rows)
        # Constraints are drawn from a seeded RNG in a fixed order, so a trainer and its
        # rollout workers derive identical prompts without sending any of this over the wire.
        data = [{"stem": s, "cids": sorted(rng.sample(range(len(IFEVAL_CONSTRAINTS)),
                                                      IFEVAL_K))} for s in rows]
        self._split(data, n_eval, seed)
        self.reward_fns = [(ifeval_reward, 1.0)]

    def build_prompt(self, p: dict) -> str:
        reqs = "\n".join(f"- {IFEVAL_CONSTRAINTS[i][0]}" for i in p["cids"])
        return f"{p['stem']}\n\nConstraints:\n{reqs}"


class CountdownTask(LLMTask):
    def __init__(self, split: str = "train", n_examples: int = 2000, seed: int = 0,
                 n_eval: int = 128, rank: int = 0, world: int = 1):
        from datasets import load_dataset

        self.rank, self.world = rank, world
        ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        ds = ds.shuffle(seed=1234).select(range(min(n_examples, len(ds))))  # pool+eval fixed
        self._split([{"nums": list(ex["nums"]), "target": int(ex["target"])} for ex in ds],
                    n_eval, seed)
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
        ds = ds.shuffle(seed=1234).select(range(min(n_examples, len(ds))))  # pool+eval fixed
        rows = []
        for ex in ds:
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            rows.append({"question": ex["question"], "value": float(gold)})
        self._split(rows, n_eval, seed)
        self.reward_fns = [(gsm8k_reward, 1.0), (format_reward, 0.1)]

    def build_prompt(self, p: dict) -> str:
        return (p["question"] + "\nReason briefly, then put ONLY the final number inside "
                "<answer> </answer> tags.")


class MathTask(LLMTask):
    """Competition math (DeepScaleR-Preview: AMC/AIME/MATH/Omni-MATH mix). Kept rows have
    numeric golds so `math_reward` needs no CAS; most of the 40k rows survive the filter.
    Deliberately harder than GSM8K/Countdown: a ~9B model should start mid-range, leaving
    headroom in both directions — Countdown starts at ~0.56 for Qwen3.5-9B with a known
    ceiling near 0.64, too narrow to measure anything against."""

    def __init__(self, split: str = "train", n_examples: int = 2000, seed: int = 0,
                 n_eval: int = 128, rank: int = 0, world: int = 1):
        from datasets import load_dataset

        self.rank, self.world = rank, world
        ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
        ds = ds.shuffle(seed=1234)  # pool+eval fixed across training seeds
        rows = []
        for ex in ds:
            v = _num_value(ex["answer"])
            if v is None:
                continue
            rows.append({"problem": ex["problem"], "value": v})
            if len(rows) >= n_examples:
                break
        self._split(rows, n_eval, seed)
        self.reward_fns = [(math_reward, 1.0), (format_reward, 0.1)]

    def build_prompt(self, p: dict) -> str:
        return (p["problem"] + "\nReason step by step, then put ONLY the final answer "
                "inside <answer> </answer> tags.")


def make_task(name: str, **kw):
    keep = lambda *names: {k: v for k, v in kw.items() if k in names}  # noqa: E731
    if name == "cartpole":
        return CartPoleTask(**keep("max_steps", "gamma", "rank", "world"))
    llm = {"countdown": CountdownTask, "ifeval": IFEvalTask,
           "gsm8k": GSM8KTask, "math": MathTask}.get(name)
    if llm is None:
        raise ValueError(f"unknown task {name!r}")
    task = llm(**keep("n_examples", "seed", "n_eval", "rank", "world"))
    task.adapt_sample = bool(kw.get("adapt_sample", False))
    task.overlong_filter = bool(kw.get("overlong_filter", False))
    return task
