# nanoRL

One RL training loop that scales from a laptop CPU to a GPU cluster. **No GPU required**: the
same disaggregated trainer/worker setup that runs on 16 GPUs runs as two pods on your laptop.
~2,200 lines across 7 files, no Ray, TRL or DeepSpeed. Like
[nanoGPT](https://github.com/karpathy/nanoGPT), a codebase to fork, not a library to import.

```python
loss = -(advantage * logprob).mean()      # + PPO ratio clip, + optional KL
```

REINFORCE, PPO, GRPO and RLOO differ only in how `advantage(...)` is computed
([algos.py](algos.py)); sync vs async only in where batches come from.

![The training loop](assets/loop.svg)

## Quickstart, on CPU

```bash
uv pip install torch gymnasium numpy transformers "datasets<4" peft pyyaml

python train.py --task cartpole --algo reinforce        # ~10 s
python train.py --config configs/cartpole_ppo.yaml      # ~20 s
python train.py --config configs/ifeval_local_cpu.yaml  # Qwen3-0.6B + LoRA, ~86 min
```

The last one runs GRPO on real prompts from
[RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval). Each prompt carries three
formatting constraints, such as "around 40 words" or "never use the words the or and", and a
couple of lines of Python check them. There is no reward model and no LLM judge.

![Local CPU run](assets/local_run.png)

Held-out score goes from 0.448 to 0.604. On a matched before/after over the same 64 held-out
prompts, strict accuracy (all three constraints, which is what IFEval itself reports) goes
from 0.109 to 0.250. The hard constraints move too. "Never use the words the or and" improves
by 0.33, and satisfying it means rewriting the sentence. Four of the nine constraints do not
move at all.

Per-step train reward is left off the plot. At `n_prompts: 2` it mostly reflects which two
prompts were drawn, and it drifts down over the run while the held-out score rises.

Three settings drive that result, and getting any of them wrong flattens the curve:

- Three constraints per prompt, scored as a fraction. With a single binary constraint, 58% of
  groups score identically, because a 0.6B model answers a given prompt much the same way
  every time. A group with no reward variance yields no advantage and no gradient.
- Constraints spread across difficulty. An earlier pool held five the model already satisfied
  (all lowercase scored 1.00), which is free reward. Training started at 0.693 with little
  left to learn. `IFEVAL_CONSTRAINTS` in [tasks.py](tasks.py) carries the measured
  per-constraint rates.
- `inner_epochs: 4`. Generation dominates wall clock, so spending each batch on one update
  wastes it, and at `1` the ratio is exactly 1 so the PPO clip never engages (`clipfrac`
  measured 0.0000). Raising it took the gain from 0.06 to 0.16.

The same run, split across a trainer pod and a rollout pod on a local
[kind](https://kind.sigs.k8s.io/) cluster (Docker Desktop >= 12 GB):

```bash
sky local up && sky jobs launch sky/local_jobgroup.yaml   # sky jobs logs <id> trainer
```

[`configs/countdown_local_cpu.yaml`](configs/countdown_local_cpu.yaml) is the arithmetic
alternative (0.039 to 0.164). Its ceiling is the model, since five configurations all stopped
at 0.172-0.195.

## Scaling up

| Command | Hardware |
|---|---|
| `--config configs/countdown_grpo.yaml` | 1x 24 GB GPU |
| `torchrun --nproc_per_node=8 train.py --config configs/countdown_grpo_gemma4.yaml` | 8x H100 |
| `sky jobs launch sky/jobgroup.yaml` | 2+ GPUs, k8s only |

The last one as it ships: Qwen3-8B + LoRA, one 8x H100 node training while eight L40S generate
with vLLM, wired by hostname via a
[SkyPilot Job Group](https://docs.skypilot.ai/en/latest/examples/job-groups.html). Held-out
0.391 to 0.609 at step 60; 512 sequences/update, 102k in 90 min; staleness mean 1.8, 0 of
1,603 batches dropped. CSV in [results/](results/).

![Training curves](assets/async_run.png)

The trainer recomputes logprobs under the weights that actually sampled each batch. An earlier
run used vLLM's as `old_logp`, the ratio drifted to 1.165 and accuracy fell from 0.500 to
0.188.

![Async health](assets/async_health.png)

## Knobs

All off by default, and a few lines of code each:

| Flag | |
|---|---|
| `--tis-clip 2` | Async only. The trainer recomputes `old_logp`, but vLLM sampled the tokens, so this is the importance weight for the gap that remains |
| `--eps-high 0.28` | Clip-higher: raise the ceiling alone, so unlikely but useful tokens can still grow |
| `--dual-clip 3` | Floor the surrogate where the advantage is negative and PPO's clip is one-sided |
| `--overlong-coef 0.5` | Penalize completions that crowd the token budget |
| `--eval-k 8` | pass@k at temperature 1 instead of greedy pass@1 |

`--skip-zero-adv` is on by default. A group scoring uniformly has zero advantage, so its
forward/backward is skipped. That is a speedup only, since the loss denominators still count
its tokens. Log columns: `entropy` (which falls before the reward does), `clipfrac`, `akl`,
`resp_len`, `trunc`, `dead`.

## Layout

```
algos.py    ~90   advantage estimators: reinforce / baseline / ppo (GAE) / grpo / rloo
core.py    ~120   Trajectory + Batch
utils.py   ~170   masked reductions, returns/GAE, dist plumbing, CSV logger
serve.py   ~250   async transport: weight publishing, rollout queue, staleness rejection
model.py   ~380   MLPPolicy, HFPolicy (train + score), VLLMGenerator (rollout only)
tasks.py   ~450   CartPole, IFEval, Countdown, GSM8K; task = sample + rollout + reward fns
train.py   ~700   the loop: rollout, advantage, clipped update, publish
```

`--role trainer` on one box and `--role rollout` on the others talk over stdlib HTTP;
`max_staleness` is rejection bound, queue depth and snapshot count at once. 65 CPU-only tests:
`python tests/test_nanorl.py`.

A dataset is an `LLMTask` subclass: load rows, write `build_prompt`, pick reward functions
`(prompt, completion, answer) -> float`, register in `make_task`. Any `AutoModelForCausalLM`
works with `--model` (~12B on 48 GB with LoRA). Not included: Megatron-scale parallelism, MoE,
multi-tenant scheduling, dashboards.

MIT
