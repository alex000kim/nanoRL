"""vLLM + Qwen3.5-9B (hybrid GDN) + LoRA load/generate probe.
Must be a real file AND main-guarded: VLLM_WORKER_MULTIPROC_METHOD=spawn re-imports
the main module in each engine proc."""


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3.5-9B", dtype="bfloat16", enable_lora=True, max_lora_rank=32,
              gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True)
    out = llm.generate(["What is 17*23? Answer with just the number."],
                       SamplingParams(temperature=0.0, max_tokens=64))
    print("VLLM_PROBE_OK:", out[0].outputs[0].text[:100])


if __name__ == "__main__":
    main()
