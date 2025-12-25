"""
Benchmark: vLLM with Qwen3-4B (text-only baseline)

This measures text generation speed to compare with HF.
"""

import time
import torch

def main():
    print("="*60)
    print("vLLM Qwen3-4B Benchmark (text-only)")
    print("="*60)

    from vllm import LLM, SamplingParams

    print("\nLoading vLLM model...")
    load_start = time.time()

    llm = LLM(
        model="Qwen/Qwen3-4B",
        dtype="bfloat16",
        gpu_memory_utilization=0.2,  # Leave room for other processes
        max_model_len=2048,
        enforce_eager=True,  # Disable CUDA graphs to save memory
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Test prompts (similar length to what audio model would produce)
    prompts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nConstruct a query to sum the values of a given column.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nCategorize the following characters into heroes and villains: Harry Potter, Voldemort, Hermione Granger.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nFind a quote related to success.<|im_end|>\n<|im_start|>assistant\n",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
    )

    print(f"\nGenerating {len(prompts)} responses...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tokens_per_sec = total_tokens / gen_time if gen_time > 0 else 0

    print("\n--- Results ---")
    for i, output in enumerate(outputs):
        print(f"\nSample {i+1}:")
        print(f"  Tokens: {len(output.outputs[0].token_ids)}")
        print(f"  Response: {output.outputs[0].text[:100]}...")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model load time: {load_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Total generation time: {gen_time:.2f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")

    # Compare with HF results
    hf_throughput = 32.60  # From previous benchmark
    print(f"\nComparison with HF:")
    print(f"  HF throughput: {hf_throughput:.2f} tokens/s")
    print(f"  vLLM throughput: {tokens_per_sec:.2f} tokens/s")
    if tokens_per_sec > 0:
        print(f"  Speedup: {tokens_per_sec / hf_throughput:.2f}x")


if __name__ == "__main__":
    main()
