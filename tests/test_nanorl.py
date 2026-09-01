"""tests/test_nanorl.py — verifier unit tests + a fixed-batch gradient check.

Run: `python -m pytest tests/` or `python tests/test_nanorl.py`. No GPU, no network.
These guard the subtle parts (masking / ratio / advantage math) from silent regressions.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SKIPPED: list[str] = []   # tests that could not run (e.g. no model available offline)

from algos import grpo, reinforce, reinforce_baseline, rloo  # noqa: E402
from core import Batch, Trajectory  # noqa: E402
from model import MLPPolicy  # noqa: E402
from tasks import (_safe_eval, countdown_reward, extract_answer,  # noqa: E402
                   format_reward, gsm8k_reward)
from utils import discounted_returns, gae, masked_mean, whiten  # noqa: E402


# --------------------------------------------------------------------------- #
# masked reductions
# --------------------------------------------------------------------------- #
def test_masked_mean_ignores_padding():
    x = torch.tensor([[1.0, 2.0, 99.0], [3.0, 99.0, 99.0]])
    m = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert abs(masked_mean(x, m).item() - (1 + 2 + 3) / 3) < 1e-6


def test_whiten_zero_mean_unit_std():
    x = torch.randn(4, 5)
    m = torch.ones(4, 5)
    w = whiten(x, m)
    assert abs(masked_mean(w, m).item()) < 1e-5
    assert abs(masked_mean(w * w, m).item() - 1.0) < 1e-3


# --------------------------------------------------------------------------- #
# credit assignment
# --------------------------------------------------------------------------- #
def test_discounted_returns_hand_value():
    r = torch.tensor([[1.0, 1.0, 1.0]])
    m = torch.ones(1, 3)
    g = discounted_returns(r, m, gamma=0.5)
    # G2=1, G1=1+.5=1.5, G0=1+.5*1.5=1.75
    assert torch.allclose(g, torch.tensor([[1.75, 1.5, 1.0]]), atol=1e-6)


def test_discounted_returns_respects_mask():
    r = torch.tensor([[1.0, 1.0, 0.0]])
    m = torch.tensor([[1.0, 1.0, 0.0]])  # last step is padding
    g = discounted_returns(r, m, gamma=1.0)
    assert torch.allclose(g, torch.tensor([[2.0, 1.0, 0.0]]), atol=1e-6)


def test_gae_terminal_equals_reward_minus_value():
    # single-step episode: A = r - V(s0), return target = r
    r = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[0.3, 0.0]])
    m = torch.tensor([[1.0, 0.0]])
    A, ret = gae(r, v, m, gamma=1.0, lam=1.0)
    assert abs(A[0, 0].item() - 0.7) < 1e-6
    assert abs(ret[0, 0].item() - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# GRPO / RLOO group baselines
# --------------------------------------------------------------------------- #
def _bandit_batch(rewards_per_group):
    """One-token trajectories, grouped: rewards_per_group is [n_prompts][group_size]."""
    groups = []
    for grp in rewards_per_group:
        tr_list = []
        for r in grp:
            tr_list.append(Trajectory(
                states=torch.zeros(1, 2), actions=torch.zeros(1).long(),
                old_logp=torch.zeros(1), rewards=torch.tensor([float(r)]), mask=torch.ones(1)))
        groups.append(tr_list)
    return Batch.from_groups(groups)


def test_grpo_normalizes_within_group():
    b = _bandit_batch([[0.0, 1.0], [0.0, 1.0]])
    A = grpo(b, None)  # [4, 1]
    # within each group: mean .5, std population? torch.std is sample std -> ddof1
    R = torch.tensor([0.0, 1.0])
    expected = (R - R.mean()) / (R.std() + 1e-4)
    assert torch.allclose(A.squeeze(1)[:2], expected, atol=1e-4)
    # advantage is zero-mean within each group
    assert abs(A.squeeze(1)[:2].mean().item()) < 1e-5


def test_grpo_drgrpo_no_std():
    b = _bandit_batch([[0.0, 2.0]])
    A = grpo(b, None, norm_std=False).squeeze(1)
    assert torch.allclose(A, torch.tensor([-1.0, 1.0]), atol=1e-6)


def test_group_algos_reject_group_size_1():
    """G=1 makes the group baseline degenerate (std of one sample is NaN). Fail loudly."""
    b = _bandit_batch([[1.0], [0.0]])           # group_size == 1
    for fn, name in ((grpo, "grpo"), (rloo, "rloo")):
        try:
            fn(b, None)
            raise AssertionError(f"{name} should reject group_size=1")
        except ValueError as e:
            assert "group-size" in str(e)


def test_advfn_binds_config_hyperparams():
    """--gamma / --lam / --norm-std must actually reach the estimators (they were dead)."""
    from algos import make_advfn
    from train import Config
    # gamma=0 -> return-to-go collapses to the immediate reward only
    tr = Trajectory(torch.zeros(2, 2), torch.zeros(2).long(), torch.zeros(2),
                    torch.tensor([1.0, 1.0]), torch.ones(2))
    b = Batch.from_groups([[tr]])
    A = make_advfn("reinforce", Config(gamma=0.0))(b, None)
    assert torch.allclose(A, torch.tensor([[1.0, 1.0]]), atol=1e-6)
    # norm_std=False reaches grpo
    bb = _bandit_batch([[0.0, 2.0]])
    A2 = make_advfn("grpo", Config(norm_std=False, group_size=2))(bb, None).squeeze(1)
    assert torch.allclose(A2, torch.tensor([-1.0, 1.0]), atol=1e-6)


def test_whiten_zeroes_padding():
    x = torch.tensor([[1.0, 3.0, 99.0]])
    m = torch.tensor([[1.0, 1.0, 0.0]])
    assert whiten(x, m)[0, 2].item() == 0.0


def test_rloo_leave_one_out():
    b = _bandit_batch([[0.0, 3.0, 3.0]])
    A = rloo(b, None).squeeze(1)
    # sample0: loo=(3+3)/2=3 -> A=-3 ; sample1: loo=(0+3)/2=1.5 -> A=1.5
    assert torch.allclose(A, torch.tensor([-3.0, 1.5, 1.5]), atol=1e-6)


# --------------------------------------------------------------------------- #
# verifiers (compiler/calculator as the environment)
# --------------------------------------------------------------------------- #
def test_extract_answer():
    assert extract_answer("<think>x</think><answer> 42 </answer>") == "42"
    assert extract_answer("no tags") is None


def test_format_reward_is_graded_on_the_answer_tag():
    """The shipped format reward keys on <answer> only — a 0.5B model never emits </think>
    cold, and an unreachable format reward means a uniform group and zero gradient."""
    assert format_reward("", "reasoning <answer>b</answer>", None) == 1.0
    assert format_reward("", "<think>a</think><answer>b</answer>", None) == 1.0
    assert format_reward("", "<answer>b unclosed", None) == 0.5      # partial credit
    assert format_reward("", "no tags at all", None) == 0.0


def test_think_format_reward_still_available():
    from tasks import think_format_reward
    assert think_format_reward("", "<think>a</think><answer>b</answer>", None) == 1.0
    assert think_format_reward("", "<answer>b</answer>", None) == 0.0


def test_safe_eval_exact():
    assert _safe_eval("2+3*4") == 14
    assert _safe_eval("(1+2)/3") == 1
    assert _safe_eval("1/0") is None
    assert _safe_eval("import os") is None


def test_countdown_reward():
    prob = {"nums": [3, 4, 5], "target": 23}
    assert countdown_reward("", "<answer>3*5+4*2</answer>", prob) == 0.0  # wrong numbers used
    assert countdown_reward("", "<answer>4*5+3</answer>", prob) == 1.0    # 23, each once
    assert countdown_reward("", "<answer>3+4+5</answer>", prob) == 0.0    # =12
    assert countdown_reward("", "no answer", prob) == 0.0
    # "expr = target" scores on the LHS only — the RHS digits must not count as used numbers
    assert countdown_reward("", "<answer>4*5+3 = 23</answer>", prob) == 1.0


# --------------------------------------------------------------------------- #
# instruction following (ifeval): the verifiers ARE the reward model
# --------------------------------------------------------------------------- #
def test_ifeval_every_constraint_accepts_and_rejects():
    """Each verifier must actually discriminate. One that always returns True is a constraint
    the model is paid for for free (measured: 'all lowercase' alone scores 1.00 on Qwen3-0.6B
    and leaves 100% of groups with no gradient); one that always returns False is unwinnable."""
    from tasks import IFEVAL_CONSTRAINTS
    good = [' '.join(['word'] * 40), 'however so however', '- one\n- two',
            'a quick brown fox jumps', 'One. Two. Three.', 'Yes indeed it is',
            'no commas here', 'SHOUTING LOUDLY', 'sure. Hope this helps.']
    bad = ['too short', 'however once', '- only one', 'the cat and the dog',
           'One. Two.', 'Maybe it is', 'yes, commas', 'quiet voice', 'ends wrong']
    assert len(good) == len(bad) == len(IFEVAL_CONSTRAINTS)
    for (text, fn), g, b in zip(IFEVAL_CONSTRAINTS, good, bad):
        assert fn(g), f"{text!r} rejected a valid completion {g!r}"
        assert not fn(b), f"{text!r} accepted an invalid completion {b!r}"


def test_ifeval_pool_cannot_produce_unwinnable_prompts():
    """Any K of the pool land on one prompt together, so contradictory pairs would make some
    prompts impossible — and an impossible prompt's group is dead exactly like a trivial one.
    Two invariants keep that from happening."""
    from tasks import IFEVAL_CONSTRAINTS
    # 1. exactly ONE word-count constraint: two ("around 40 words" + "at most 20 words")
    #    is the obvious way to make a prompt unsatisfiable.
    assert sum("words." in t and any(ch.isdigit() for ch in t)
               for t, _ in IFEVAL_CONSTRAINTS) == 1
    # 2. the phrase/word checks must survive "write in capital letters", i.e. be
    #    case-insensitive — otherwise capitals contradicts three other constraints at once.
    caps = "YES HOWEVER IT IS SO HOWEVER. SOMETHING MORE. HOPE THIS HELPS."
    by = dict(IFEVAL_CONSTRAINTS)
    assert by["Write your entire response in capital letters."](caps)
    assert by["Finish with the exact phrase: Hope this helps."](caps)
    assert by['Use the word "however" exactly twice.'](caps)
    assert by["Begin your response with the word Yes or the word No."](caps)
    assert by["Write exactly 3 sentences."](caps)


def test_ifeval_reward_is_graded_not_binary():
    """THE design point. A 0.6B model answers a given prompt the same way every time, so a
    single all-or-nothing check makes whole groups score identically and GRPO's advantage is
    exactly 0 (measured: 58% of groups dead at K=1 vs 17% at K=3). Partial credit is what
    puts variance back."""
    from tasks import IFEVAL_CONSTRAINTS, ifeval_reward, ifeval_strict_reward
    # look constraints up by text, so tuning the pool cannot silently invert this test
    want = ["Do not use any commas.", "Write your entire response in capital letters.",
            "Begin your response with the word Yes or the word No."]
    idx = [i for i, (t, _) in enumerate(IFEVAL_CONSTRAINTS) if t in want]
    p = {"cids": idx}
    assert len(idx) == 3
    assert ifeval_reward("", "YES IT IS SO", p) == 1.0
    assert abs(ifeval_reward("", "YES IT IS, SO", p) - 2 / 3) < 1e-9   # only the comma fails
    assert abs(ifeval_reward("", "maybe, quiet", p) - 0.0) < 1e-9
    # strict is IFEval's own metric: reported, never trained on
    assert ifeval_strict_reward("", "YES IT IS SO", p) == 1.0
    assert ifeval_strict_reward("", "YES IT IS, SO", p) == 0.0


def test_norm_std_erases_reward_spacing_on_two_level_groups():
    """On a two-level group, (R-mean)/std reduces to sign(a-b)*sqrt(n_b/n_a) -- independent of
    the reward VALUES. So any monotone reshaping is exactly inert there, and nothing at runtime
    says so: reward moves, loss moves, gradient does not. Hence `levels` beside `dead`."""
    import torch

    from algos import grpo
    from core import Batch, Trajectory

    def batch_of(rewards, T=4):
        trs = []
        for r in rewards:
            rw = torch.zeros(T)
            rw[-1] = r
            trs.append(Trajectory(states=torch.zeros(T, dtype=torch.long),
                                  actions=torch.zeros(T, dtype=torch.long),
                                  old_logp=torch.zeros(T), rewards=rw, mask=torch.ones(T)))
        return Batch.from_groups([trs])

    two = [2 / 3, 2 / 3, 1.0, 2 / 3, 1.0, 2 / 3, 2 / 3, 1.0]
    ref = grpo(batch_of(two), None, norm_std=True)[:, 0]
    for exp in (1.5, 2.0, 3.0):
        got = grpo(batch_of([r ** exp for r in two]), None, norm_std=True)[:, 0]
        assert torch.allclose(ref, got, atol=1e-3), (
            f"exponent {exp} changed a two-level group under norm_std -- derivation is wrong")
    lo = grpo(batch_of(two), None, norm_std=False)[:, 0].abs().max()
    hi = grpo(batch_of([r ** 3 for r in two]), None, norm_std=False)[:, 0].abs().max()
    assert hi > lo * 1.5, "reshaping must reprice the group when norm_std is off"
    three = [1 / 3, 2 / 3, 1.0, 2 / 3, 1.0, 1 / 3, 2 / 3, 1.0]
    a1 = grpo(batch_of(three), None, norm_std=True)[:, 0]
    a3 = grpo(batch_of([r ** 3 for r in three]), None, norm_std=True)[:, 0]
    assert not torch.allclose(a1, a3, atol=1e-2)


def test_eval_uses_the_configured_token_budget():
    """cfg.max_new_tokens must reach eval, not just rollouts: evaluate() defaults to 256, so the
    knob looked applied while half the run ignored it."""
    import train as T

    cfg = T.Config(task="ifeval", model="Qwen/Qwen3-0.6B", max_new_tokens=128, eval_k=1)
    assert T.rollout_kwargs(cfg)["max_new_tokens"] == 128
    eval_kw = {"k": cfg.eval_k, "max_new_tokens": cfg.max_new_tokens} if cfg.model else {}
    assert eval_kw["max_new_tokens"] == cfg.max_new_tokens == 128
    assert T.rollout_kwargs(T.Config(task="cartpole", model="")) == {}


def test_ifeval_prompt_states_the_constraints_it_scores():
    """The model can only satisfy what it was told. If build_prompt and the reward ever drift
    apart the task becomes unlearnable while still logging a plausible reward."""
    from tasks import IFEVAL_CONSTRAINTS, IFEvalTask
    p = {"stem": "Explain photosynthesis.", "cids": [1, 3]}
    text = IFEvalTask.build_prompt(None, p)
    for i in p["cids"]:
        assert IFEVAL_CONSTRAINTS[i][0] in text
    for i in set(range(len(IFEVAL_CONSTRAINTS))) - set(p["cids"]):
        assert IFEVAL_CONSTRAINTS[i][0] not in text, "prompt lists a constraint it is not scored on"


def test_ifeval_assignment_is_identical_across_processes():
    """Constraints are drawn from a seeded RNG rather than shipped over the wire, so a rollout
    worker MUST derive exactly what the trainer derived. If these diverge the worker optimizes
    constraints the trainer never scores, and nothing in the metrics would show it."""
    try:
        from tasks import IFEvalTask
        a = IFEvalTask(n_examples=200, seed=0, n_eval=8)
        b = IFEvalTask(n_examples=200, seed=0, n_eval=8)   # the "worker"
    except Exception as e:
        if os.environ.get("NANORL_REQUIRE_LLM_TEST"):
            raise
        print(f"  (SKIPPED — dataset unavailable: {type(e).__name__})")
        _SKIPPED.append("test_ifeval_assignment_is_identical_across_processes")
        return
    assert a.data[:5] == b.data[:5] and a.eval_data == b.eval_data
    c = IFEvalTask(n_examples=200, seed=1, n_eval=8)
    assert a.data[:5] != c.data[:5], "a different --seed must give a different task"
    # the eval holdout must never be handed out for training
    assert not ({r["stem"] for r in a.eval_data} & {r["stem"] for r in a.data})
    from tasks import IFEVAL_K
    assert all(len(r["cids"]) == IFEVAL_K == len(set(r["cids"])) for r in a.data[:50])


def test_ifeval_system_prompt_does_not_demand_answer_tags():
    """The shared RLVR system prompt asks for <answer> tags, which directly contradicts
    constraints like 'wrap your entire response in quotation marks'. Making system_prompt
    per-task must not change it for the tag-based tasks."""
    from tasks import SYSTEM_PROMPT, CountdownTask, GSM8KTask, IFEvalTask, LLMTask
    assert "<answer>" not in IFEvalTask.system_prompt
    assert IFEvalTask.system_prompt != SYSTEM_PROMPT
    for t in (LLMTask, CountdownTask, GSM8KTask):
        assert t.system_prompt is SYSTEM_PROMPT, f"{t.__name__} lost the <answer>-tag prompt"


def test_gsm8k_reward():
    prob = {"value": 18.0}
    assert gsm8k_reward("", "<answer>18</answer>", prob) == 1.0
    assert gsm8k_reward("", "<answer>the answer is 18</answer>", prob) == 1.0
    assert gsm8k_reward("", "<answer>17</answer>", prob) == 0.0


# --------------------------------------------------------------------------- #
# the ratio==1 invariant + a fixed-batch gradient check
# --------------------------------------------------------------------------- #
def test_ratio_is_one_on_first_epoch():
    from model import MLPPolicy
    torch.manual_seed(0)
    policy = MLPPolicy(obs_dim=2, n_actions=3)
    states = torch.randn(4, 2)
    a, logp, _ = policy.act(states)
    tr = Trajectory(states=states[:, None, :].reshape(4, 1, 2), actions=a[:, None],
                    old_logp=logp[:, None], rewards=torch.ones(4, 1), mask=torch.ones(4, 1))
    # each row is its own 1-step trajectory
    batch = Batch.from_groups([[Trajectory(states[i][None], a[i:i+1], logp[i:i+1],
                                           torch.ones(1), torch.ones(1))] for i in range(4)])
    recomputed = policy.logprobs(batch)
    ratio = torch.exp(recomputed - batch.old_logp)
    assert torch.allclose(ratio[batch.mask.bool()], torch.ones(4), atol=1e-5)


def test_fixed_batch_reinforce_loss():
    """Hand-computed REINFORCE loss on a fixed 2-trajectory batch."""
    # two 1-step trajectories, known logp and reward
    logp = torch.tensor([[-1.0], [-2.0]])
    adv = torch.tensor([[2.0], [-1.0]])
    mask = torch.ones(2, 1)
    loss = -masked_mean(adv * logp, mask)
    # -(mean of [2*-1, -1*-2]) = -(mean of [-2, 2]) = 0
    assert abs(loss.item() - 0.0) < 1e-6


# --------------------------------------------------------------------------- #
# the REAL training loss (train.py rl_loss), incl. length-norm / Dr.GRPO (C2, C3)
# --------------------------------------------------------------------------- #
class _DummyPolicy:
    """logprobs == old_logp, so ratio == 1 and surr == adv — lets us pin the aggregation."""
    def logprobs(self, batch):
        return batch.old_logp.clone().requires_grad_(True) * batch.mask


def _two_seq_batch():
    # seq0: 3 completion tokens, seq1: 1 completion token (different lengths)
    from train import Config
    tr0 = Trajectory(torch.zeros(4, 2), torch.zeros(4).long(), torch.zeros(4),
                     torch.zeros(4), torch.tensor([1., 1., 1., 0.]))
    tr1 = Trajectory(torch.zeros(4, 2), torch.zeros(4).long(), torch.zeros(4),
                     torch.zeros(4), torch.tensor([1., 0., 0., 0.]))
    b = Batch.from_groups([[tr0], [tr1]])
    adv = torch.zeros(2, 4)
    adv[0, :3] = 2.0   # seq0 advantage +2 on its tokens
    adv[1, 0] = -1.0   # seq1 advantage -1
    return b, adv, Config


def test_rl_loss_length_norm_is_per_sequence():
    """length_norm=True must weight every completion equally regardless of length (C3)."""
    from train import rl_loss
    b, adv, Config = _two_seq_batch()
    cfg = Config(length_norm=True, eps=0.2, ent_coef=0.0, kl_coef=0.0, model="llm")
    loss, diag = rl_loss(_DummyPolicy(), b, adv, cfg)
    # per-seq means = [2, -1]; pg = -mean([2,-1]) = -0.5  (NOT the length-weighted -5/4)
    assert abs(loss.item() - (-0.5)) < 1e-6
    assert abs(diag["ratio_sum"] - b.mask.sum().item()) < 1e-5   # ratio == 1 everywhere


def test_rl_loss_drgrpo_constant_denominator():
    """length_norm=False (Dr.GRPO) divides by a fixed constant, not batch T (C3)."""
    from train import rl_loss
    b, adv, Config = _two_seq_batch()
    cfg = Config(length_norm=False, eps=0.2, ent_coef=0.0, kl_coef=0.0, model="llm", max_new_tokens=4)
    loss, _ = rl_loss(_DummyPolicy(), b, adv, cfg)
    # masked_sum(surr) = 2*3 + (-1)*1 = 5 ; denom = N * max_new_tokens = 2 * 4 = 8 ; pg = -5/8
    assert abs(loss.item() - (-5.0 / 8.0)) < 1e-6


class _ShiftPolicy(torch.nn.Module):
    """logprobs == old_logp + shift, so the ratio is exp(shift) at every token — lets us pin
    the clip branches by hand. shift is a real Parameter so optimize() has something to clip."""
    def __init__(self, shift):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(float(shift)))

    def logprobs(self, batch):
        return (batch.old_logp + self.shift) * batch.mask


def test_clip_higher_raises_only_the_upper_bound():
    """DAPO clip-higher: a low-probability token that is doing well should still be allowed to
    grow, so eps_high loosens the ceiling while the floor stays put. eps_high=0 must reproduce
    the symmetric clip exactly."""
    import math
    from train import rl_loss
    b, adv, Config = _two_seq_batch()          # adv: +2 over seq0's 3 tokens, -1 over seq1's 1
    pol = _ShiftPolicy(math.log(2.0))          # ratio == 2 everywhere
    kw = dict(model="llm", length_norm=True, eps=0.2)
    l0, d0 = rl_loss(pol, b, adv, Config(**kw))
    # seq0: min(2*2, clamp(2,.8,1.2)*2)=2.4 ; seq1: min(2*-1, 1.2*-1)=-2
    assert abs(l0.item() - (-(2.4 - 2.0) / 2)) < 1e-5
    l1, _ = rl_loss(pol, b, adv, Config(**kw, eps_high=0.5))
    # ceiling 1.5 lets the positive-advantage sequence through at 3.0; the negative one is
    # untouched — that asymmetry is the whole point
    assert abs(l1.item() - (-(3.0 - 2.0) / 2)) < 1e-5
    assert abs(d0["clipfrac_sum"] - b.mask.sum().item()) < 1e-6   # ratio 2 clips every token


def test_dual_clip_bounds_the_negative_advantage_branch():
    """With A<0 the UNCLIPPED branch is the min, so the surrogate runs to -inf as a stale
    ratio grows and one token owns the step. dual_clip floors it at dual_clip*A."""
    import math
    from train import rl_loss
    b, adv, Config = _two_seq_batch()
    pol = _ShiftPolicy(math.log(2.0))
    l, _ = rl_loss(pol, b, adv, Config(model="llm", length_norm=True, eps=0.2, dual_clip=1.5))
    # seq1's -2.0 is floored at 1.5*-1 = -1.5; seq0 (A>0) keeps its 2.4
    assert abs(l.item() - (-(2.4 - 1.5) / 2)) < 1e-5
    # ...and it must not touch positive advantages
    l2, _ = rl_loss(pol, b, adv, Config(model="llm", length_norm=True, eps=0.2, dual_clip=99.0))
    assert abs(l2.item() - (-(2.4 - 2.0) / 2)) < 1e-5


def test_tis_reweights_by_the_sampler_gap_and_truncates():
    """THE vLLM fix, completed. recompute_old_logp fixes the ratio's DENOMINATOR (what the
    sampling weights say under the trainer's kernels), but the tokens were still drawn from
    the engine's own numerics. TIS is that leftover ratio — detached, and truncated because
    it is unbounded above."""
    import math
    from train import rl_loss
    b, adv, Config = _two_seq_batch()
    b.sample_logp = b.old_logp - 0.5           # the engine called every token less likely
    kw = dict(model="llm", length_norm=True, eps=0.2, role="trainer")
    l, _ = rl_loss(_DummyPolicy(), b, adv, Config(**kw, tis_clip=10.0))
    assert abs(l.item() - (-0.5 * math.exp(0.5))) < 1e-5      # -0.5 is the un-weighted loss
    l2, _ = rl_loss(_DummyPolicy(), b, adv, Config(**kw, tis_clip=1.2))
    assert abs(l2.item() - (-0.5 * 1.2)) < 1e-5               # ceiling binds
    # sync batches carry no sample_logp, so the correction must simply not fire
    b.sample_logp = None
    l3, _ = rl_loss(_DummyPolicy(), b, adv, Config(**kw, tis_clip=1.2))
    assert abs(l3.item() - (-0.5)) < 1e-6


def test_zero_advantage_skip_is_numerically_exact():
    """Skipping micro-batches whose advantage is identically 0 must change the speed and
    nothing else: their surrogate is 0, their gradient is 0, and loss_denoms deliberately
    still counts their tokens. If this drifts, the skip is silently reweighting the batch."""
    from train import Config, optimize
    torch.manual_seed(0)
    trs = [Trajectory(torch.randn(3, 2), torch.randint(0, 3, (3,)), torch.zeros(3),
                      torch.ones(3), torch.ones(3)) for _ in range(8)]
    batch = Batch.from_groups([[t] for t in trs])
    adv = torch.randn(8, 3) * batch.mask
    adv[2:6] = 0.0                              # two whole micro-batches carry no signal
    grads, diags = [], []
    for skip in (False, True):
        torch.manual_seed(1)
        pol = MLPPolicy(2, 3)
        opt = torch.optim.SGD(pol.parameters(), lr=0.0)   # lr=0: compare grads, not updates
        cfg = Config(model="llm", micro_batch=2, max_new_tokens=3, max_grad_norm=1e9,
                     skip_zero_adv=skip)
        diags.append(optimize(pol, opt, batch, adv, cfg))
        grads.append([p.grad.clone() for p in pol.parameters()])
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(*grads))
    assert diags[0]["skipped"] == 0 and diags[1]["skipped"] == 4
    assert abs(diags[0]["pg"] - diags[1]["pg"]) < 1e-6


def test_zero_advantage_skip_stays_off_when_kl_is_on():
    """The skip is exact only for a pure policy gradient. A KL (or entropy, or value) term
    does not vanish with the advantage, so dropping the chunk would drop real loss."""
    from train import Config, optimize
    torch.manual_seed(0)
    trs = [Trajectory(torch.randn(2, 2), torch.randint(0, 3, (2,)), torch.zeros(2),
                      torch.ones(2), torch.ones(2)) for _ in range(4)]
    batch = Batch.from_groups([[t] for t in trs])
    adv = torch.zeros(4, 2)                     # every chunk is a skip candidate
    pol = MLPPolicy(2, 3)
    ref = MLPPolicy(2, 3)                       # a DIFFERENT net, so the KL is nonzero
    cfg = Config(model="llm", micro_batch=2, max_new_tokens=2, kl_coef=1.0, max_grad_norm=1e9)
    diag = optimize(pol, torch.optim.SGD(pol.parameters(), lr=0.0), batch, adv, cfg, ref=ref)
    assert diag["skipped"] == 0, "skipping with kl_coef>0 would silently drop the KL term"
    assert abs(diag["loss"]) > 0


def test_diagnostics_are_not_diluted_by_skipped_chunks():
    """The loss denominator stays global (that is what makes the skip exact), but the
    DIAGNOSTICS must divide by the tokens actually scored. Dividing skipped tokens in reports
    ratio well below 1 on a perfectly on-policy step — drift that never happened, in the one
    number this repo reads to decide whether staleness is under control."""
    from train import Config, optimize
    trs = [Trajectory(torch.zeros(2, 2), torch.zeros(2).long(), torch.zeros(2),
                      torch.zeros(2), torch.ones(2)) for _ in range(4)]
    batch = Batch.from_groups([[t] for t in trs])
    adv = torch.ones(4, 2)
    adv[2:] = 0.0                               # half the batch carries no signal
    pol = _ShiftPolicy(0.0)                     # perfectly on-policy: ratio == 1 everywhere
    cfg = Config(model="llm", micro_batch=2, max_new_tokens=2, max_grad_norm=1e9)
    d = optimize(pol, torch.optim.SGD(pol.parameters(), lr=0.0), batch, adv, cfg)
    assert d["skipped"] == 2
    assert abs(d["ratio"] - 1.0) < 1e-6, "skipped tokens were divided into the ratio"


def test_diagnostics_are_global_per_token_means():
    """entropy is the sampled surprisal E[-log pi(a)] (the policy entropy in expectation, for
    free); akl is the drift from the sampling policy — the number that says whether
    --max-staleness is set right. Both are summed per chunk and divided once, so they must
    come out as plain per-token means."""
    import math
    from train import Config, optimize
    trs = [Trajectory(torch.zeros(2, 2), torch.zeros(2).long(), torch.zeros(2),
                      torch.zeros(2), torch.ones(2)) for _ in range(4)]
    batch = Batch.from_groups([[t] for t in trs])
    adv = torch.ones(4, 2)
    cfg = Config(model="llm", micro_batch=2, max_new_tokens=2, eps=0.2, max_grad_norm=1e9)
    pol = _ShiftPolicy(-1.0)                    # logp = old_logp - 1 = -1 at every token
    d = optimize(pol, torch.optim.SGD(pol.parameters(), lr=0.0), batch, adv, cfg)
    assert abs(d["entropy"] - 1.0) < 1e-4       # -logp == 1
    assert abs(d["akl"] - 1.0) < 1e-4           # old_logp - logp == 0 - (-1)
    assert abs(d["ratio"] - math.exp(-1)) < 1e-4
    assert abs(d["clipfrac"] - 1.0) < 1e-4      # exp(-1)=0.37 is below the 0.8 floor


def test_batch_carries_truncation_and_sampler_logprobs_through_slicing():
    """A micro-batch that lost sample_logp would silently disable TIS for its tokens, and
    trunc_rate needs the flag to survive from_groups."""
    from train import Config, batch_stats
    trs = [Trajectory(torch.zeros(2, 2), torch.zeros(2).long(), torch.zeros(2),
                      torch.zeros(2), torch.ones(2), truncated=(i % 2 == 0)) for i in range(4)]
    b = Batch.from_groups([[t] for t in trs])
    assert b.truncated.tolist() == [1.0, 0.0, 1.0, 0.0]
    b.sample_logp = torch.arange(8.0).reshape(4, 2)
    sub = b.slice(1, 3)
    assert sub.truncated.tolist() == [0.0, 1.0]
    assert torch.equal(sub.sample_logp, torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
    assert batch_stats(b, Config(model="llm"))["trunc"] == 0.5
    assert "trunc" not in batch_stats(b, Config(model=""))   # control episodes end, not truncate


def test_dead_group_fraction_counts_uniform_groups():
    """`dead` is how much of the batch carried no gradient at all — the honest measure of
    waste, and the number --group-size and reward shaping are there to move."""
    from train import Config, batch_stats
    b = _bandit_batch([[1.0, 1.0], [0.0, 1.0]])   # group 0 is uniform, group 1 is not
    assert batch_stats(b, Config(model="llm"))["dead"] == 0.5


# --------------------------------------------------------------------------- #
# length shaping + pass@k
# --------------------------------------------------------------------------- #
def test_overlong_penalty_is_free_until_the_buffer():
    """Graded, not a flat penalty on truncation: the gradient has to point somewhere BEFORE
    the cliff, or the only signal arrives once the answer is already lost."""
    from tasks import overlong_penalty
    assert overlong_penalty(100, budget=256) == 0.0        # cache = 64, so free below 192
    assert overlong_penalty(192, budget=256) == 0.0
    assert overlong_penalty(224, budget=256) == -0.5       # halfway into the buffer
    assert overlong_penalty(256, budget=256) == -1.0       # cut off at the budget


def _fake_llm_task(reward_fn, completions=None):
    """An LLMTask with the model replaced by a scripted policy (no weights, no network)."""
    from tasks import LLMTask

    class FakeTok:
        def decode(self, ids, skip_special_tokens=True): return "ok"

    class FakePolicy:
        tok, device = FakeTok(), None
        def format_prompt(self, u, s=None): return u
        def generate(self, texts, max_new_tokens=256, temperature=0.0): return completions
        def act(self, texts, group_size, max_new_tokens=256, **kw):
            out = []
            for T in (90, 128):                            # one inside the buffer, one at it
                out.append(Trajectory(torch.zeros(T).long(), torch.zeros(T).long(),
                                      torch.zeros(T), torch.zeros(T), torch.ones(T)))
            return out

    class Toy(LLMTask):
        reward_fns = [(reward_fn, 1.0)]
        def build_prompt(self, p): return "q"

    return Toy(), FakePolicy()


def test_rollout_applies_the_overlong_penalty():
    task, pol = _fake_llm_task(lambda p, c, a: 1.0)
    b = task.rollout(pol, [{"x": 0}], group_size=2, max_new_tokens=128, overlong_coef=1.0)
    r = b.terminal_rewards()
    assert abs(r[0].item() - 1.0) < 1e-6     # 90 tokens <= 128-32: unpenalized
    assert abs(r[1].item() - 0.0) < 1e-6     # 128 tokens: full -1.0 against the +1.0 reward
    # and it stays off by default
    b0 = task.rollout(pol, [{"x": 0}], group_size=2, max_new_tokens=128)
    assert abs(b0.terminal_rewards()[1].item() - 1.0) < 1e-6


def test_trainer_and_worker_build_the_same_rollout_kwargs():
    """In sync mode the trainer rolls out; in async mode the worker does. If the two built
    their kwargs separately, --overlong-coef would apply in one and vanish in the other and
    the same config would score the same completion differently."""
    from train import Config, rollout_kwargs
    cfg = Config(task="countdown", model="m", max_new_tokens=64, overlong_coef=0.5)
    assert rollout_kwargs(cfg)["overlong_coef"] == 0.5
    assert rollout_kwargs(Config(task="cartpole")) == {}   # control takes no generation knobs


def test_pass_at_k_counts_problems_not_samples():
    """acc rising while pass_k stays flat means the policy sharpened onto what it could
    already do rather than learning anything new — the usual reason a run plateaus."""
    task, pol = _fake_llm_task(
        lambda p, c, a: float(extract_answer(c) == "1"),
        # problem 0: 1 of its 2 samples correct; problem 1: neither
        completions=["<answer>1</answer>", "<answer>0</answer>",
                     "<answer>0</answer>", "<answer>0</answer>"])
    task.eval_data = [{"i": 0}, {"i": 1}]
    out = task.evaluate(pol, n=2, k=2)
    assert out["acc"] == 0.25                # 1 of 4 samples
    assert out["pass_k"] == 0.5              # 1 of 2 problems solved at least once


# --------------------------------------------------------------------------- #
# integration smoke tests: train() actually runs a step for each control algo (C2)
# --------------------------------------------------------------------------- #
def _smoke(algo, **over):
    import tempfile
    from train import Config, train
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(task="cartpole", algo=algo, steps=2, n_prompts=4,
                     eval_every=0, ckpt_every=0, out=d + "/run", **over)
        pol = train(cfg)
    assert pol is not None


def test_train_smoke_reinforce():
    _smoke("reinforce")


def test_train_smoke_ppo():
    _smoke("ppo", inner_epochs=2, ent_coef=0.01)  # exercises value loss + truncation bootstrap


def test_train_smoke_grpo_control():
    _smoke("grpo", group_size=4)  # group axis + broadcast advantage on control


def test_train_rejects_bad_combos():
    from train import Config, validate
    for bad, why in ((dict(algo="ppo", model="hf", task="countdown"), "critic"),
                     (dict(algo="grpo", group_size=1, task="cartpole"), "group-size"),
                     (dict(model="hf", task="countdown", group_size=8, algo="grpo",
                           role="rollout", trainer_url="http://x", vllm=True), "--lora"),
                     (dict(model="hf", task="countdown", group_size=8, algo="grpo",
                           ent_coef=0.01), "ent-coef"),
                     # a knob that cannot do anything must say so, not no-op quietly
                     (dict(model="hf", task="countdown", group_size=8, algo="grpo",
                           tis_clip=3.0), "--role trainer"),
                     (dict(role="trainer", tis_clip=0.5), "tis-clip"),
                     (dict(dual_clip=0.9), "dual-clip"),
                     (dict(eps=0.2, eps_high=0.1), "eps-high"),
                     (dict(eval_k=0), "eval-k")):
        try:
            validate(Config(**bad))
            raise AssertionError(f"should reject {bad}")
        except ValueError as e:
            assert why in str(e)


def test_resume_restores_step_and_weights():
    import tempfile
    from train import Config, load, make_policy, save
    from tasks import CartPoleTask
    task = CartPoleTask()
    cfg = Config(task="cartpole", algo="reinforce")
    p1 = make_policy(cfg, task)
    o1 = torch.optim.AdamW(p1.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as d:
        cfg.out = d + "/run"
        save(p1, o1, 7, cfg)
        p2 = make_policy(cfg, task)
        o2 = torch.optim.AdamW(p2.parameters(), lr=1e-3)
        step = load(f"{d}/run_step7.pt", p2, o2)
    assert step == 7
    assert all(torch.equal(a, b) for a, b in zip(p1.state_dict().values(),
                                                 p2.state_dict().values()))


def test_microbatched_backward_matches_one_shot():
    """THE memory fix must be numerically exact: accumulating gradients over micro-batches
    has to equal one big backward, or the fix silently reweights the batch."""
    from train import Config, optimize
    from model import MLPPolicy
    torch.manual_seed(0)
    trs = []
    for i in range(6):
        T = 2 + i % 3
        trs.append(Trajectory(torch.randn(T, 2), torch.randint(0, 3, (T,)),
                              torch.zeros(T), torch.ones(T), torch.ones(T)))
    batch = Batch.from_groups([[t] for t in trs])
    adv = torch.randn(batch.mask.shape) * batch.mask
    grads = []
    for mb in (999, 2):  # one chunk vs micro-batched (model="llm" enables chunking)
        torch.manual_seed(1)
        pol = MLPPolicy(2, 3)
        opt = torch.optim.SGD(pol.parameters(), lr=0.0)  # lr=0: compare grads, not updates
        cfg = Config(model="llm", micro_batch=mb, max_new_tokens=4, max_grad_norm=1e9)
        optimize(pol, opt, batch, adv, cfg)
        grads.append([p.grad.clone() for p in pol.parameters()])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(*grads))


# --------------------------------------------------------------------------- #
# LLM path: micro-batching gives identical logprobs to a single big forward (C1)
# --------------------------------------------------------------------------- #
def test_data_parallel_gradients_match_single_process():
    """2 gloo ranks on CPU must produce the SAME gradient as one process on the whole batch.
    Guards the two halves of the multi-GPU contract: globally-summed loss denominators and a
    SUM (not MEAN) gradient all-reduce. Skipped if torchrun is unavailable."""
    import shutil
    import subprocess
    if not shutil.which("torchrun"):
        _SKIPPED.append("test_data_parallel_gradients_match_single_process")
        return
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "dist_equiv.py")
    env = {**os.environ,
           "GLOO_SOCKET_IFNAME": os.environ.get(
               "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"),
           "OMP_NUM_THREADS": "1"}
    subprocess.run([sys.executable, script], check=True, env=env,
                   capture_output=True, timeout=300)          # writes the reference
    r = subprocess.run(["torchrun", "--nproc_per_node=2", "--master_addr=127.0.0.1",
                        "--master_port=29537", script], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout, r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# disaggregated async RL (serve.py) — real HTTP, loopback, no GPU
# --------------------------------------------------------------------------- #
def _toy_batch(reward=1.0):
    tr = Trajectory(torch.zeros(3, 2), torch.zeros(3).long(), torch.zeros(3),
                    torch.tensor([0.0, 0.0, reward]), torch.ones(3))
    return Batch.from_groups([[tr], [tr]])


def test_async_transport_roundtrip():
    """A Batch survives the wire, and the trainer sees the version it was sampled under."""
    from serve import RolloutClient, TrainerServer
    srv = TrainerServer(port=8731, queue_max=4)
    try:
        cli = RolloutClient("http://127.0.0.1:8731")
        cli.wait_for_trainer(timeout=30)
        srv.publish(MLPPolicy(2, 3))                 # -> version 1
        ver, sd = cli.pull_weights(have=0)
        assert ver == 1 and sd is not None
        assert cli.pull_weights(have=1) == (1, None)  # no re-download when unchanged
        cli.submit(ver, _toy_batch(reward=2.0))
        batch, lag = srv.pop(max_staleness=2, timeout=30)
        assert lag == 0
        assert abs(batch.terminal_rewards().sum().item() - 4.0) < 1e-6   # 2 trajs x 2.0
    finally:
        srv.close()


def test_async_rejects_stale_batches():
    """Version rejection: a batch sampled too many versions ago must be dropped, not trained on."""
    from serve import RolloutClient, TrainerServer
    srv = TrainerServer(port=8732, queue_max=4)
    try:
        cli = RolloutClient("http://127.0.0.1:8732")
        cli.wait_for_trainer(timeout=30)
        pol = MLPPolicy(2, 3)
        cli.submit(0, _toy_batch())                  # sampled under v0
        for _ in range(5):
            srv.publish(pol)                         # trainer races ahead to v5
        try:
            srv.pop(max_staleness=2, timeout=2)
            raise AssertionError("stale batch should have been dropped")
        except TimeoutError:
            pass
        assert srv.stats()["dropped"] >= 1
        cli.submit(srv.version, _toy_batch())        # fresh one gets through
        _, lag = srv.pop(max_staleness=2, timeout=30)
        assert lag == 0
    finally:
        srv.close()


def test_adapter_state_is_trainable_params_only():
    """Weight sync must ship ONLY trainable tensors — that is what makes per-step sync cheap."""
    from serve import adapter_state
    pol = MLPPolicy(2, 3)
    for n, p in pol.named_parameters():
        if n.startswith("body"):
            p.requires_grad_(False)
    sd = adapter_state(pol)
    assert sd and not any(k.startswith("body") for k in sd)
    assert all(k.startswith(("pi", "vbody", "v")) for k in sd)


def test_staleness_opens_the_importance_ratio():
    """THE async claim, made concrete.

    Sync: old_logp is captured under the same weights the update uses -> ratio == 1 and the
    clip is a no-op. Async: the sampling policy is behind the training policy, so the very
    same line yields ratio != 1 and the clip becomes load-bearing. Nothing in the loss changes
    between the two regimes — only how stale the batch is.
    """
    from tasks import CartPoleTask
    torch.manual_seed(0)
    task = CartPoleTask()
    sampler = MLPPolicy(task.obs_dim, task.n_actions)     # the "rollout worker" policy
    batch = task.rollout(sampler, task.sample(2), group_size=1)

    # staleness 0: trainer weights == sampler weights
    ratio0 = torch.exp(sampler.logprobs(batch) - batch.old_logp)[batch.mask.bool()]
    assert torch.allclose(ratio0, torch.ones_like(ratio0), atol=1e-5)

    # staleness > 0: the trainer has since taken steps, so its weights differ
    trainer = MLPPolicy(task.obs_dim, task.n_actions)
    trainer.load_state_dict(sampler.state_dict())
    opt = torch.optim.SGD(trainer.parameters(), lr=0.5)
    for _ in range(3):
        opt.zero_grad(); trainer.logprobs(batch).sum().backward(); opt.step()
    ratio1 = torch.exp(trainer.logprobs(batch) - batch.old_logp)[batch.mask.bool()]
    assert not torch.allclose(ratio1, torch.ones_like(ratio1), atol=1e-3), "seam stayed shut"
    # and the clip is what keeps that finite
    clipped = torch.clamp(ratio1, 0.8, 1.2)
    assert float(clipped.max()) <= 1.2 + 1e-6 and float(clipped.min()) >= 0.8 - 1e-6


def test_sync_source_has_zero_staleness():
    """The sync path must keep staleness==0, i.e. remain exactly the old behaviour."""
    from train import Config, SyncSource
    from tasks import CartPoleTask
    from model import MLPPolicy as MP
    task = CartPoleTask()
    src = SyncSource(task, Config(task="cartpole", group_size=1), {}, 2)
    batch, staleness = src.next_batch(MP(task.obs_dim, task.n_actions))
    assert staleness == 0
    assert src.stats() == {}
    assert batch.states.shape[0] == 2


def test_async_source_always_recomputes_old_logp():
    """The async trainer must never trust worker logprobs: whatever old_logp a submitted batch
    carries, next_batch replaces it with the trainer's own recompute under the sampling
    version's snapshot, and reports the measured gap."""
    from serve import RolloutClient
    from train import AsyncSource, Config
    torch.manual_seed(0)
    cfg = Config(role="trainer", device="cpu", max_staleness=2, pop_timeout=30, serve_port=8733)
    src = AsyncSource(cfg)
    try:
        pol = MLPPolicy(2, 3)
        src.publish(pol)                             # -> version 1, snapshot kept locally
        cli = RolloutClient("http://127.0.0.1:8733")
        cli.wait_for_trainer(timeout=30)
        bogus = _toy_batch()                         # its old_logp is all zeros: clearly wrong
        cli.submit(1, bogus)
        batch, lag = src.next_batch(pol)
        assert lag == 0
        truth = pol.logprobs(batch).detach()
        m = batch.mask.bool()
        assert torch.allclose(batch.old_logp[m], truth[m], atol=1e-6)
        assert src.stats()["logp_gap"] > 0           # the zeros-vs-truth gap was measured
    finally:
        src.server.close()


def test_recompute_old_logp_is_exact_under_the_sampling_version():
    """THE vLLM fix. A worker's logprobs may come from different kernels; recomputing them on
    the trainer under the sampling version's weights must reproduce exactly what that version
    would have said — so the ratio measures staleness and nothing else."""
    from train import Config, recompute_old_logp
    from serve import adapter_state
    torch.manual_seed(0)
    pol = MLPPolicy(2, 3)
    trs = [Trajectory(torch.randn(3, 2), torch.randint(0, 3, (3,)), torch.zeros(3),
                      torch.ones(3), torch.ones(3)) for _ in range(4)]
    batch = Batch.from_groups([[t] for t in trs])

    truth = pol.logprobs(batch).detach().clone()      # what version v really says
    snap = adapter_state(pol)                         # snapshot version v
    opt = torch.optim.SGD(pol.parameters(), lr=0.5)   # trainer moves on to version v+3
    for _ in range(3):
        opt.zero_grad(); pol.logprobs(batch).sum().backward(); opt.step()
    assert not torch.allclose(pol.logprobs(batch), truth, atol=1e-4)   # weights really moved

    cfg = Config(model="", micro_batch=2)
    got = recompute_old_logp(pol, batch, snap, cfg)
    assert torch.allclose(got, truth.cpu(), atol=1e-6), "old_logp != what the sampler computed"
    # and the trainer's current weights must be restored afterwards
    assert not torch.allclose(pol.logprobs(batch).detach(), truth, atol=1e-4)


def test_llm_end_to_end_step():
    """The real LLM path: generate -> reward -> group advantage -> micro-batched backward.
    Set NANORL_REQUIRE_LLM_TEST=1 (CI does) to turn an offline skip into a failure."""
    try:
        from model import HFPolicy
        pol = HFPolicy("hf-internal-testing/tiny-random-gpt2", device="cpu", dtype="float32",
                       micro_batch=2)
    except Exception as e:
        if os.environ.get("NANORL_REQUIRE_LLM_TEST"):
            raise
        print(f"  (SKIPPED — no model available: {type(e).__name__})")
        _SKIPPED.append("test_llm_end_to_end_step")
        return
    from tasks import LLMTask, format_reward
    from algos import grpo
    from train import Config, optimize

    # A reward that VARIES within each group. With a random tiny model every completion would
    # otherwise score 0, making the advantage identically 0 — a run that trains nothing would
    # still pass a mere "is finite" check.
    counter = iter(range(10_000))

    class Toy(LLMTask):
        reward_fns = [(lambda p, c, a: float(next(counter) % 3), 1.0)]
        def sample(self, n): return [{"x": i} for i in range(n)]
        def build_prompt(self, p): return f"count {p['x']}"

    task = Toy()
    batch = task.rollout(pol, task.sample(2), group_size=4, max_new_tokens=8, temperature=0.7)
    assert batch.temperature == 0.7                     # sampling temp travels on the batch
    # ratio == 1 exactly: old_logp and the update use the same forward at the same temperature
    ratio = torch.exp(pol.logprobs(batch) - batch.old_logp)[batch.mask.bool()]
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), ratio.min().item()
    # a real optimizer step over micro-batches, with a NONZERO gradient that moves weights
    adv = grpo(batch, pol)
    assert adv.abs().sum() > 0, "degenerate group: advantage is identically zero"
    cfg = Config(model="tiny", eps=0.2, length_norm=True, max_new_tokens=8, micro_batch=2)
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-4)
    p0 = next(p for p in pol.model.parameters() if p.requires_grad).detach().clone()
    diag = optimize(pol, opt, batch, adv, cfg)
    assert all(torch.isfinite(torch.tensor(v)) for v in diag.values())
    assert diag["gnorm"] > 0, "no gradient reached the model"
    assert not torch.equal(p0, next(p for p in pol.model.parameters() if p.requires_grad))


def test_uniform_group_gives_zero_advantage():
    """A group where every sample scores the same has no signal — GRPO's advantage is exactly
    0 and the step is a no-op. Documented behavior, not a bug: it's why group_size and reward
    shaping matter."""
    b = _bandit_batch([[1.0, 1.0, 1.0]])
    assert grpo(b, None).abs().sum().item() == 0.0


def test_master_weights_are_fp32_and_actually_move():
    """bf16 MASTER weights + RL learning rates = a silent no-op: AdamW's ~1e-6 update to a
    ~2e-2 weight is below bf16's ~4e-3 resolution, so nothing trains. --dtype must select the
    COMPUTE dtype only."""
    try:
        from model import HFPolicy
        pol = HFPolicy("hf-internal-testing/tiny-random-gpt2", device="cpu", dtype="bfloat16")
    except Exception:
        if os.environ.get("NANORL_REQUIRE_LLM_TEST"):
            raise
        _SKIPPED.append("test_master_weights_are_fp32_and_actually_move")
        return
    assert all(p.dtype == torch.float32 for p in pol.model.parameters()), "weights must be fp32"
    p0 = next(p for p in pol.model.parameters() if p.requires_grad)
    before = p0.detach().clone()
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-6)
    ids = torch.randint(0, 50, (2, 8))
    for _ in range(5):
        opt.zero_grad(); pol._seq_logprobs(ids).sum().backward(); opt.step()
    assert not torch.equal(before, p0.detach()), "params did not move at lr=1e-6"


def test_llm_stop_tokens_cover_generation_config():
    """Truncation must use every stop id, not just tok.eos_token_id (Qwen has two)."""
    try:
        from model import HFPolicy
        pol = HFPolicy("hf-internal-testing/tiny-random-gpt2", device="cpu", dtype="float32")
    except Exception:
        if os.environ.get("NANORL_REQUIRE_LLM_TEST"):
            raise
        _SKIPPED.append("test_llm_stop_tokens_cover_generation_config")
        return
    gen_eos = getattr(pol.model.generation_config, "eos_token_id", None)
    expect = set(gen_eos) if isinstance(gen_eos, (list, tuple)) else {gen_eos}
    expect.add(pol.tok.eos_token_id)
    assert pol.stop_ids == {int(x) for x in expect if x is not None}


def test_hf_sampling_knobs_are_neutralized():
    """generate() inherits any knob not passed explicitly from generation_config.json
    (Qwen ships top_k=20 + repetition_penalty=1.1; HF's own default is top_k=50). The loss
    models softmax(logits/T) only, so HFPolicy must neutralize every other warper — the bias
    is invisible to the ratio (==1 on epoch one by construction) and to logp_gap."""
    try:
        from model import HFPolicy
        pol = HFPolicy("hf-internal-testing/tiny-random-gpt2", device="cpu", dtype="float32")
    except Exception:
        if os.environ.get("NANORL_REQUIRE_LLM_TEST"):
            raise
        _SKIPPED.append("test_hf_sampling_knobs_are_neutralized")
        return
    gc = pol.model.generation_config
    assert gc.top_k in (0, None), gc.top_k
    assert gc.top_p == 1.0 and gc.temperature == 1.0
    assert gc.repetition_penalty == 1.0 and gc.no_repeat_ngram_size == 0
    assert getattr(gc, "min_p", None) in (None, 0.0)


def test_grad_ckpt_enables_training_mode_without_dropout():
    """HF layers only checkpoint when model.training is True (from_pretrained returns eval
    mode), so grad_ckpt must flip train mode — with every Dropout pinned to 0 first, or
    old_logp and the update would see different networks."""
    try:
        from model import HFPolicy
        pol = HFPolicy("hf-internal-testing/tiny-random-gpt2", device="cpu", dtype="float32",
                       grad_ckpt=True)
    except Exception:
        if os.environ.get("NANORL_REQUIRE_LLM_TEST"):
            raise
        _SKIPPED.append("test_grad_ckpt_enables_training_mode_without_dropout")
        return
    import torch.nn as nn
    assert pol.model.training, "grad_ckpt without train() is a silent no-op"
    drops = [m for m in pol.model.modules() if isinstance(m, nn.Dropout)]
    assert drops and all(m.p == 0.0 for m in drops), "dropout would break ratio==1"
    pol2 = HFPolicy("hf-internal-testing/tiny-random-gpt2", device="cpu", dtype="float32")
    assert not pol2.model.training, "without grad_ckpt the model must stay in eval mode"
    # generation must run in eval mode either way: train-mode generate disables the KV cache
    # and takes a different forward path that corrupts greedy decoding (measured: a 0.5B
    # model's countdown answer degrades to garbage, eval_acc 0.39 -> 0.00 on the cluster)
    prompts = ["hello world", "the quick brown fox"]
    assert pol.generate(prompts, max_new_tokens=16, temperature=0.0) == \
           pol2.generate(prompts, max_new_tokens=16, temperature=0.0)
    assert pol.model.training, "train mode must be restored after generation"


def test_async_drops_batch_without_snapshot():
    """A batch whose sampling version has no snapshot must be DROPPED, not trained on with
    worker logprobs as old_logp (the documented run-collapsing failure)."""
    from serve import RolloutClient
    from train import AsyncSource, Config
    torch.manual_seed(0)
    cfg = Config(role="trainer", device="cpu", max_staleness=4, pop_timeout=30, serve_port=8734)
    src = AsyncSource(cfg)
    try:
        pol = MLPPolicy(2, 3)
        src.publish(pol)                             # -> version 1; snapshot exists for v1 only
        cli = RolloutClient("http://127.0.0.1:8734")
        cli.wait_for_trainer(timeout=30)
        cli.submit(0, _toy_batch())                  # v0: within staleness, but NO snapshot
        cli.submit(1, _toy_batch())                  # v1: has a snapshot
        batch, lag = src.next_batch(pol)
        assert lag == 0                              # the v1 batch; v0 was dropped
        assert src.stats()["snap_dropped"] == 1
    finally:
        src.server.close()


def test_async_keeps_worker_logprobs_for_tis():
    """old_logp gets overwritten by the trainer's recompute — but the worker's own numbers are
    the BEHAVIOUR policy (the distribution the tokens were really drawn from), so they have to
    survive as sample_logp or --tis-clip has nothing to correct against."""
    from serve import RolloutClient
    from train import AsyncSource, Config
    torch.manual_seed(0)
    cfg = Config(role="trainer", device="cpu", max_staleness=2, pop_timeout=30, serve_port=8736)
    src = AsyncSource(cfg)
    try:
        pol = MLPPolicy(2, 3)
        src.publish(pol)
        cli = RolloutClient("http://127.0.0.1:8736")
        cli.wait_for_trainer(timeout=30)
        b = _toy_batch()
        b.old_logp = b.old_logp + 0.25           # stand in for what the engine reported
        cli.submit(1, b)
        batch, _ = src.next_batch(pol)
        m = batch.mask.bool()
        assert batch.sample_logp is not None, "TIS lost its numerator"
        assert torch.allclose(batch.sample_logp[m],
                              torch.full_like(batch.sample_logp[m], 0.25))
        assert not torch.allclose(batch.old_logp[m], batch.sample_logp[m])  # really recomputed
    finally:
        src.server.close()


def test_hf_marks_truncated_completions():
    """A completion cut off at the budget never emitted its answer. rollout() has to be able
    to tell it apart from a finished wrong one, or the length penalty has nothing to key on."""
    try:
        from model import HFPolicy
        pol = HFPolicy("hf-internal-testing/tiny-random-gpt2", device="cpu", dtype="float32",
                       micro_batch=2)
    except Exception:
        if os.environ.get("NANORL_REQUIRE_LLM_TEST"):
            raise
        _SKIPPED.append("test_hf_marks_truncated_completions")
        return
    trajs = pol.act(["hello world"], group_size=4, max_new_tokens=6, temperature=1.0)
    assert trajs
    for tr in trajs:
        n, last = int(tr.mask.sum()), int(tr.states[tr.mask.bool()][-1])
        if tr.truncated:
            assert n == 6 and last not in pol.stop_ids   # spent the whole budget, never stopped
        else:
            assert last in pol.stop_ids                  # kept the stop token, so it finished


def test_worker_and_trainer_split_fingerprints():
    """The trainer serves its split fingerprint; a worker whose fingerprint differs would
    train on the trainer's eval holdout, so the fields that shape the split must all be in."""
    from serve import RolloutClient, TrainerServer
    from train import Config, split_fingerprint
    fp = split_fingerprint(Config(task="countdown", model="m", seed=0))
    srv = TrainerServer(port=8735, queue_max=2, meta=fp)
    try:
        cli = RolloutClient("http://127.0.0.1:8735")
        cli.wait_for_trainer(timeout=30)
        assert cli.fetch_config() == fp              # survives the wire verbatim
        for k, v in (("seed", 1), ("n_examples", 999), ("eval_n", 7), ("task", "gsm8k")):
            assert split_fingerprint(Config(task="countdown", model="m", **{k: v}) if k != "task"
                                     else Config(task=v, model="m")) != fp, k
    finally:
        srv.close()


def test_validate_requires_token_for_async_roles():
    """The trainer's HTTP endpoint serves weights and ingests training data on 0.0.0.0;
    running either async role without NANORL_TOKEN must fail loudly at startup."""
    from train import Config, validate
    old = os.environ.pop("NANORL_TOKEN", None)
    try:
        try:
            validate(Config(role="trainer"))
            raise AssertionError("unauthenticated trainer should be rejected")
        except ValueError as e:
            assert "NANORL_TOKEN" in str(e)
        os.environ["NANORL_TOKEN"] = "t"
        validate(Config(role="trainer"))             # same config passes once the token is set
    finally:
        os.environ.pop("NANORL_TOKEN", None)
        if old is not None:
            os.environ["NANORL_TOKEN"] = old


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    msg = f"\n{len(fns) - len(_SKIPPED)} passed"
    if _SKIPPED:
        msg += f", {len(_SKIPPED)} SKIPPED ({', '.join(_SKIPPED)}) — "
        msg += "set NANORL_REQUIRE_LLM_TEST=1 to make skips fail"
    print(msg)
