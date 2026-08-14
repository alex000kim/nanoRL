# nanoRL

An RL training loop that runs CartPole on a laptop and disaggregated async RLVR on a GPU
cluster. About 1,800 lines across 7 files, with no Ray, TRL, or DeepSpeed. Like
[nanoGPT](https://github.com/karpathy/nanoGPT), it's a codebase to fork rather than a
library to import.

```python
loss = -(advantage * logprob).mean()      # + PPO ratio clip, + optional KL
```

REINFORCE, PPO, GRPO and RLOO differ only in how `advantage(...)` is computed, and sync vs
async only in where batches come from.

![The training loop](assets/loop.svg)

## What's in it

[algos.py](algos.py) has the advantage estimators: raw returns, a constant baseline, GAE
with a critic, and the group baselines (GRPO, RLOO). Adding one means writing one function.

Async training splits the same loop across machines. Run `--role trainer` on one box and
`--role rollout` on the others; they talk over stdlib HTTP ([serve.py](serve.py)). If a
worker dies you lose its in-flight batch. `max_staleness` is the only knob, and it sets
three things at once: the rejection bound, the queue depth, and how many weight snapshots
the trainer keeps.

The trainer recomputes `old_logp` under the exact weights that sampled each batch, drops
any batch whose weights it no longer has, and prints the measured gap every step next to
the importance ratio, the staleness, and the queue drops. HFPolicy also neutralizes the
`generation_config.json` knobs the ratio doesn't model (`top_k`, `repetition_penalty`).

Speed comes from vLLM on the workers, separate batch sizes for generation and scoring
(opposite memory profiles; tying them cost 6x in one measured run), adapter-only weight
sync at ~170 MB rather than 16 GB, and a micro-batched backward whose gradient matches one
big backward on any number of GPUs. 44 CPU-only tests cover it: `python tests/test_nanorl.py`.

## Quickstart

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
uv pip install torch gymnasium numpy transformers "datasets<4" peft pyyaml
python train.py --task cartpole --algo reinforce     # solves in ~10 s on CPU
```

| # | Command | Hardware |
|---|---|---|
| 1 | `--task cartpole --algo reinforce` | CPU, seconds |
| 2 | `--config configs/cartpole_ppo.yaml` | CPU, ~20 s |
| 3 | `--config configs/countdown_grpo.yaml` | 1x 24 GB GPU |
| 4 | `torchrun --nproc_per_node=8 train.py --config configs/countdown_grpo_gemma4.yaml` | 8x H100 |
| 5 | `sky jobs launch sky/jobgroup.yaml` | 2+ GPUs, k8s only (workers find the trainer by hostname) |

## The end-to-end run

Row 5, exactly as the YAML ships: Qwen3-8B with LoRA doing GRPO on Countdown. One 8x H100
node trains, eight L40S GPUs generate with vLLM, and a
[SkyPilot Job Group](https://docs.skypilot.ai/en/latest/examples/job-groups.html) wires
them together by hostname.

![Training curves](assets/async_run.png)

| Metric | Value |
|---|---|
| Held-out accuracy (greedy) | 0.391 to 0.609 peak (+22 pp) at step 60; 0.61 again at steps 160–180 |
| Throughput | 512 sequences/update; 200 steps, 102k sequences in 90 min |
| Staleness | mean 1.8 versions, max 2; 0 of 1,603 batches dropped (staleness bound or missing snapshot) |
| Worker vs trainer logprob gap | 0.026 to 0.007, corrected every step |

That last row is why the trainer recomputes logprobs: vLLM computes them with different
kernels than the trainer's HF forward. An earlier run used them as `old_logp` anyway, and
the ratio drifted from 1.006 to 1.165 at zero staleness while accuracy fell from 0.500 to
0.188.

![Async health](assets/async_health.png)

Two GPUs are enough to reproduce (one trainer, one worker), and the run's CSV is in
[results/](results/). On the sync path, row 4 got Gemma-4-12B from 0.312 to 0.688.

## Layout

```
algos.py    ~90   advantage estimators: reinforce / baseline / ppo (GAE) / grpo / rloo
core.py    ~110   Trajectory + Batch
utils.py   ~170   masked reductions, returns/GAE, dist plumbing, CSV logger
serve.py   ~250   async transport: weight publishing, rollout queue, staleness rejection
tasks.py   ~310   CartPole + Countdown/GSM8K; task = sample + rollout + reward fns
model.py   ~330   MLPPolicy, HFPolicy (train + score), VLLMGenerator (rollout only)
train.py   ~560   the loop: rollout, advantage, clipped update, publish
```

## Your model, your dataset

Any `AutoModelForCausalLM` id works with `--model` if it fits on one GPU, which with LoRA
is roughly 12B on 48 GB. LoRA targets `q_proj/k_proj/v_proj/o_proj`; models that name their
projections differently need a one-line change in [model.py](model.py).

A dataset is a subclass of `LLMTask`. Load rows, write `build_prompt`, pick reward
functions, register the class in `make_task` ([tasks.py](tasks.py)), then run with
`--task <name>`:

```python
class MyTask(LLMTask):
    def __init__(self, n_examples=2000, seed=0, n_eval=128, rank=0, world=1, **kw):
        self.rank, self.world = rank, world
        rows = [...]                       # [{"question": ..., "value": ...}, ...]
        self._split(rows, n_eval)          # eval holdout + per-rank sharding
        self.reward_fns = [(gsm8k_reward, 1.0), (format_reward, 0.1)]

    def build_prompt(self, p):
        return p["question"] + "\nPut ONLY the final number in <answer> </answer> tags."
```

A reward function is `(prompt, completion, answer) -> float` and can call anything, from a
regex to an external judge.

Not included: Megatron-scale parallelism, MoE, multi-tenant scheduling, dashboards.

## License

MIT
