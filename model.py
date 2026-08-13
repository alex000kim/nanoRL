"""The policies. All implement the same protocol:
    act(...)         sample actions/tokens + capture old logprobs (rollout, no grad)
    logprobs(batch)  recompute logp under CURRENT params          (update, with grad)
    value(states)    critic head, or None                         (ppo only)
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Control — a NumPy-simple MLP policy (+ optional value head)
# --------------------------------------------------------------------------- #
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, value_head: bool = False):
        super().__init__()

        def trunk():
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())

        self.body = trunk()
        self.pi = nn.Linear(hidden, n_actions)
        # separate critic trunk: a shared body lets the much larger value loss drown the
        # policy gradient
        self.vbody = trunk() if value_head else None
        self.v = nn.Linear(hidden, 1) if value_head else None

    def _logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.pi(self.body(states))

    @torch.no_grad()
    def act(self, states: torch.Tensor, greedy: bool = False):
        """states: [B, obs] -> (action[B], logp[B], value[B] or None)."""
        logits = self._logits(states)
        logp_all = F.log_softmax(logits, dim=-1)
        action = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        logp = logp_all.gather(-1, action[:, None]).squeeze(-1)
        value = self.v(self.vbody(states)).squeeze(-1) if self.v is not None else None
        return action, logp, value

    def entropy(self, batch) -> torch.Tensor:
        """Per-step policy entropy [N,T] — an optional exploration bonus for control."""
        N, T = batch.actions.shape
        logits = self._logits(batch.states.reshape(N * T, -1))
        return torch.distributions.Categorical(logits=logits).entropy().reshape(N, T)

    def grad_groups(self) -> dict:
        """Actor and critic clipped separately, so the critic's large norm cannot rescale
        the policy update."""
        if self.v is None:
            return {"policy": list(self.parameters())}
        critic = {id(p) for p in [*self.vbody.parameters(), *self.v.parameters()]}
        return {"policy": [p for p in self.parameters() if id(p) not in critic],
                "value": [p for p in self.parameters() if id(p) in critic]}

    def logprobs(self, batch) -> torch.Tensor:
        """batch.states [N,T,obs], batch.actions [N,T] -> per-step logp [N,T] (with grad)."""
        N, T = batch.actions.shape
        logits = self._logits(batch.states.reshape(N * T, -1))
        logp_all = F.log_softmax(logits, dim=-1)
        gathered = logp_all.gather(-1, batch.actions.reshape(N * T, 1)).reshape(N, T)
        return gathered * batch.mask

    def value(self, states: torch.Tensor) -> torch.Tensor:
        """states [N,T,obs] -> [N,T] value estimates (from the critic's own trunk)."""
        assert self.v is not None, "value() called on a policy without a value head"
        N, T = states.shape[:2]
        return self.v(self.vbody(states.reshape(N * T, -1))).reshape(N, T)


# --------------------------------------------------------------------------- #
# LLM — a thin wrapper around a small HF causal LM
# --------------------------------------------------------------------------- #
def chat_prompt(tok, user_msg: str, system_msg: str | None, think: bool) -> str:
    """Chat-template glue, shared by HF and vLLM policies so all roles format identically."""
    msgs = []
    if system_msg:
        msgs.append({"role": "system", "content": system_msg})
    msgs.append({"role": "user", "content": user_msg})
    if not tok.chat_template:
        return (f"{system_msg}\n" if system_msg else "") + user_msg
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=think)
    except TypeError:      # template takes no such kwarg (most models)
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class VLLMGenerator:
    """Generation-only policy backed by vLLM, for `--role rollout --vllm`.

    Its logprobs come from different kernels than the trainer's HF forward, so they are never
    used as old_logp (the trainer recomputes); they only feed the logp_gap diagnostic.
    """

    def __init__(self, model_name: str, dtype: str = "bfloat16", lora_r: int = 32,
                 gpu_frac: float = 0.85, max_len: int = 2048, think: bool = False):
        import os as _os
        # spawn, not fork: the parent has already touched CUDA
        _os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # the flashinfer sampler JIT-compiles and needs nvcc, which cluster images lack
        _os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        from transformers import AutoTokenizer
        self.think = think
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.llm = LLM(model=model_name, dtype=dtype, enable_lora=True, max_lora_rank=lora_r,
                       gpu_memory_utilization=gpu_frac, max_model_len=max_len,
                       # eager: torch.compile pulls in flashinfer (broken on py3.10) and
                       # slows engine startup
                       enforce_eager=True)
        self.lora_req = None          # set by set_adapter() on every weight sync
        self.device = "cuda"
        self.v = None

    def format_prompt(self, user_msg: str, system_msg: str | None = None) -> str:
        return chat_prompt(self.tok, user_msg, system_msg, self.think)

    def set_adapter(self, path: str, version: int) -> None:
        """Point subsequent generations at the pulled adapter (version doubles as vLLM's id)."""
        from vllm.lora.request import LoRARequest
        self.lora_req = LoRARequest(f"v{version}", version, path)

    @torch.no_grad()
    def act(self, prompt_texts, group_size: int, max_new_tokens: int = 256,
            temperature: float = 1.0, top_p: float = 1.0):
        from core import Trajectory
        from vllm import SamplingParams
        sp = SamplingParams(n=group_size, temperature=max(temperature, 1e-6), top_p=top_p,
                            max_tokens=max_new_tokens, logprobs=0)
        outs = self.llm.generate(prompt_texts, sp, lora_request=self.lora_req)
        trajs = []
        for out in outs:                       # vLLM keeps the n samples grouped per prompt
            p_ids = list(out.prompt_token_ids)
            for cand in out.outputs:
                c_ids = list(cand.token_ids)
                lp = [0.0] * len(p_ids)
                for pos, tid in enumerate(c_ids):
                    d = cand.logprobs[pos] if cand.logprobs else None
                    lp.append(float(d[tid].logprob) if d and tid in d else 0.0)
                full = torch.tensor(p_ids + c_ids, dtype=torch.long)
                mask = torch.zeros(full.shape[0])
                mask[len(p_ids):] = 1.0
                trajs.append(Trajectory(states=full, actions=full.clone(),
                                        old_logp=torch.tensor(lp, dtype=torch.float32),
                                        rewards=torch.zeros(full.shape[0]), mask=mask))
        return trajs

    def logprobs(self, batch):
        raise NotImplementedError("vLLM policy is generation-only; the trainer scores with HF")


class HFPolicy(nn.Module):
    """Owns three things only: sampling with captured logprobs, logprob recompute, and
    tokenizer/chat-template glue. No TRL, no DeepSpeed."""

    def __init__(self, model_name: str, device: str = "cpu", dtype: str = "bfloat16",
                 lora: bool = False, lora_r: int = 16, micro_batch: int = 4,
                 gen_batch: int = 0, grad_ckpt: bool = False, think: bool = False):
        super().__init__()
        self.think = think        # hybrid-reasoning template toggle (see format_prompt)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        # `dtype` is the COMPUTE dtype (autocast). Trained weights stay fp32: AdamW's ~lr-sized
        # updates round to zero in bf16. The frozen LoRA base can be bf16 (halves its memory).
        self.amp_dtype = {"float32": None, "bfloat16": torch.bfloat16,
                          "float16": torch.float16}[dtype]
        base_dtype = torch.bfloat16 if (lora and self.amp_dtype is not None) else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=base_dtype)
        except TypeError:  # older transformers
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=base_dtype)
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.0,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
            self.model = get_peft_model(self.model, cfg)
            for p in self.model.parameters():   # adapters must be fp32 (see above)
                if p.requires_grad:
                    p.data = p.data.float()
        if grad_ckpt:
            # use_reentrant=False is required to co-exist with frozen (LoRA) params
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.config.use_cache = False       # incompatible with checkpointing
            # HF layers only checkpoint when self.training is True, and from_pretrained
            # returns the model in eval mode — without train() the flag is a silent no-op.
            # Dropout must be pinned to 0 first, or train mode would make old_logp and the
            # update see different networks (breaking the ratio==1 invariant).
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0
            self.model.train()
        self.model.to(device)
        self.pad_id = self.tok.pad_token_id
        self.is_lora = lora
        self.micro_batch = micro_batch      # sequences per SCORING forward (logits-bound)
        self.gen_batch = gen_batch or micro_batch   # sequences per GENERATE call (KV-bound)
        self._warned_topp = False
        # every id that ends a completion; tok.eos_token_id alone misses e.g. Qwen's
        # <|endoftext|>, which would land the terminal reward on a pad token
        gen_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        stops = gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos]
        self.stop_ids = {int(s) for s in [*stops, self.tok.eos_token_id] if s is not None}
        # generate() fills every knob NOT passed explicitly from the model's
        # generation_config.json (Qwen ships top_k=20 + repetition_penalty=1.1; HF's own
        # default is top_k=50). The loss models softmax(logits/T) and nothing else, so any
        # extra warper samples from a distribution the gradient never sees — and no
        # diagnostic catches it: the ratio is 1 on the first epoch by construction.
        gc = self.model.generation_config
        for k, v in dict(temperature=1.0, top_p=1.0, top_k=0, min_p=None, typical_p=1.0,
                         epsilon_cutoff=0.0, eta_cutoff=0.0, repetition_penalty=1.0,
                         no_repeat_ngram_size=0, penalty_alpha=None, num_beams=1).items():
            if hasattr(gc, k):
                setattr(gc, k, v)
        self.v = None  # no critic for RLVR

    def _autocast(self):
        import contextlib
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=dev, dtype=self.amp_dtype)

    @contextlib.contextmanager
    def _eval_mode(self):
        """Generation always runs in eval mode. Under grad_ckpt the model lives in train()
        (HF layers only checkpoint when training), but train mode force-disables the KV cache
        and switches generate() onto a different forward path — measured to corrupt greedy
        decoding outright, not just slow it. The gradient forward keeps train mode."""
        was = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            if was:
                self.model.train()

    # ---- prompt formatting ------------------------------------------------ #
    def format_prompt(self, user_msg: str, system_msg: str | None = None) -> str:
        return chat_prompt(self.tok, user_msg, system_msg, self.think)

    # ---- rollout: sample completions and capture old logprobs ------------- #
    @torch.no_grad()
    def act(self, prompt_texts: list[str], group_size: int, max_new_tokens: int = 256,
            temperature: float = 1.0, top_p: float = 1.0):
        """Sample `group_size` completions per prompt and capture old_logp per token.

        Generation (KV-bound) uses the large `gen_batch`; scoring (logits-bound) uses
        `micro_batch` sub-chunks. old_logp comes from the same `_seq_logprobs` at the same
        temperature the update uses, so the ratio is 1 on inner-epoch 0.
        """
        from core import Trajectory

        temp = temperature if temperature > 0 else 1.0  # passed explicitly; never global state
        if top_p < 1.0 and not self._warned_topp:
            print(f"[warn] top_p={top_p}<1 truncates the sampling dist but is NOT applied when "
                  f"recomputing logprobs (the blog's 'Keep Sampling Mask'); ratio is only "
                  f"approximate. Prefer temperature for exploration.", flush=True)
            self._warned_topp = True

        expanded = [p for p in prompt_texts for _ in range(group_size)]
        trajs = []
        with self._eval_mode():
            return self._act_batches(expanded, max_new_tokens, temperature, temp, top_p, trajs)

    def _act_batches(self, expanded, max_new_tokens, temperature, temp, top_p, trajs):
        """act()'s body, split out so the whole thing (generation AND the old_logp scoring
        forward) runs under _eval_mode."""
        from core import Trajectory
        for i in range(0, len(expanded), self.gen_batch):
            chunk = expanded[i : i + self.gen_batch]
            self.tok.padding_side = "left"
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            in_len = enc["input_ids"].shape[1]
            with self._autocast():
                out = self.model.generate(
                    **enc, do_sample=temperature > 0, temperature=max(temperature, 1e-6),
                    top_p=top_p, max_new_tokens=max_new_tokens, pad_token_id=self.pad_id,
                    use_cache=True)   # grad-checkpointing disables the cache in the config
            gen = out[:, in_len:]

            # assemble each row's full (prompt+completion) sequence, stop-token-truncated
            fulls, masks = [], []
            for b in range(len(chunk)):
                row_prompt = enc["input_ids"][b][enc["attention_mask"][b].bool()]
                comp = gen[b]
                stop = torch.isin(comp, torch.tensor(sorted(self.stop_ids), device=comp.device))
                hit = stop.nonzero()
                comp = comp[: hit[0, 0] + 1] if len(hit) else comp   # keep the stop token
                full = torch.cat([row_prompt, comp])
                m = torch.zeros(full.shape[0])
                m[row_prompt.shape[0] : row_prompt.shape[0] + comp.shape[0]] = 1.0
                fulls.append(full)
                masks.append(m)
            # score the chunk in micro_batch-sized sub-chunks for old_logp
            Lc = max(f.shape[0] for f in fulls)
            padded = torch.full((len(fulls), Lc), self.pad_id, dtype=fulls[0].dtype)
            for b, f in enumerate(fulls):
                padded[b, : f.shape[0]] = f
            logp_chunk = torch.cat([
                self._seq_logprobs(padded[j : j + self.micro_batch].to(self.device), temp).cpu()
                for j in range(0, padded.shape[0], self.micro_batch)], dim=0)
            for b, (full, m) in enumerate(zip(fulls, masks)):
                trajs.append(Trajectory(states=full, actions=full.clone(),
                                        old_logp=logp_chunk[b, : full.shape[0]],
                                        rewards=torch.zeros(full.shape[0]), mask=m))
        return trajs

    @torch.no_grad()
    def generate(self, prompt_texts: list[str], max_new_tokens: int = 256,
                 temperature: float = 0.0, top_p: float = 1.0) -> list[str]:
        """Decode-only (for eval): skips act()'s old_logp forward."""
        outs = []
        with self._eval_mode():
            return self._generate_batches(prompt_texts, max_new_tokens, temperature, top_p, outs)

    def _generate_batches(self, prompt_texts, max_new_tokens, temperature, top_p, outs):
        for i in range(0, len(prompt_texts), self.gen_batch):
            chunk = prompt_texts[i : i + self.gen_batch]
            self.tok.padding_side = "left"
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self._autocast():
                out = self.model.generate(
                    **enc, do_sample=temperature > 0, temperature=max(temperature, 1e-6),
                    top_p=top_p, max_new_tokens=max_new_tokens, pad_token_id=self.pad_id,
                    use_cache=True)   # grad-checkpointing disables the cache in the config
            outs += self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        return outs

    # ---- update: per-token logprobs under current params ------------------ #
    def _seq_logprobs(self, ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """ids [N, L] -> logp [N, L], logp[:,t] = logp(id_t | id_<t) at `temperature`.

        cross_entropy fuses log_softmax without a second [N, L, vocab] fp32 copy (fit vs OOM
        at 151k vocab). Right padding + causal attention means no pad mask is needed.
        """
        ids = ids.to(self.device)
        with self._autocast():
            logits = self.model(ids).logits[:, :-1, :]
        if temperature != 1.0:
            logits = logits / temperature
        N, Lm1, V = logits.shape
        logp = -F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1),
                                reduction="none").view(N, Lm1)
        return F.pad(logp, (1, 0), value=0.0)                          # prepend logp[:,0]=0

    def logprobs(self, batch) -> torch.Tensor:
        """Logprobs for this batch (a micro-batch slice; the caller bounds memory by
        backwarding each chunk before building the next)."""
        return self._seq_logprobs(batch.states, batch.temperature) * batch.mask.to(self.device)

    @torch.no_grad()
    def ref_logprobs(self, batch) -> torch.Tensor:
        """KL reference without a model copy: under LoRA the adapter-disabled base IS the
        reference."""
        if self.is_lora:
            with self.model.disable_adapter():
                return self.logprobs(batch)
        return self.logprobs(batch)

    def value(self, states):
        return None
