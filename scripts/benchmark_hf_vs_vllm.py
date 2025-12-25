"""
Benchmark: Native HuggingFace vs vLLM inference for Borealis

This script compares:
1. Native HF inference (full model)
2. Hybrid: HF audio encoding + vLLM text generation
"""

import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"
# Prevent vLLM from causing issues on import
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import time
import torch
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from datasets import load_dataset, Audio

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Vikhrmodels/Borealis-5b-it"
NUM_SAMPLES = 3
MAX_NEW_TOKENS = 128


def load_test_audio():
    """Load test audio samples."""
    print("Loading test audio...")
    ds = load_dataset("Vikhrmodels/Speech-Instructions", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    samples = []
    for i, sample in enumerate(ds):
        if i >= NUM_SAMPLES:
            break
        audio = torch.tensor(sample["audio"]["array"]).float()
        samples.append({
            "audio": audio,
            "question": sample.get("question", ""),
            "answer": sample.get("answer", "")[:200],
        })

    print(f"Loaded {len(samples)} samples")
    return samples


def benchmark_native_hf(samples):
    """Benchmark native HuggingFace inference."""
    print("\n" + "="*60)
    print("BENCHMARK: Native HuggingFace")
    print("="*60)

    print("Loading model...")
    load_start = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device=DEVICE,
    )
    model.eval()
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    results = []
    total_tokens = 0
    total_time = 0

    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}/{len(samples)}")
        print(f"Question: {sample['question'][:80]}...")

        start = time.time()
        with torch.inference_mode():
            output = model.generate(
                audio=sample["audio"],
                user_prompt="What is being said in this audio? <|start_of_audio|><|end_of_audio|>",
                system_prompt="You are a helpful voice assistant.",
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
            )
        gen_time = time.time() - start

        response = model.decode(output[0])
        num_tokens = output.shape[1]
        total_tokens += num_tokens
        total_time += gen_time

        print(f"Response ({num_tokens} tokens, {gen_time:.2f}s): {response[:100]}...")
        results.append({
            "response": response,
            "tokens": num_tokens,
            "time": gen_time,
        })

    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    print(f"\n--- HF Summary ---")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        "results": results,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_sec": tokens_per_sec,
        "load_time": load_time,
    }


def benchmark_vllm(samples):
    """Benchmark vLLM inference with Qwen3 (text-only, for comparison)."""
    print("\n" + "="*60)
    print("BENCHMARK: vLLM (Qwen3-4B text-only baseline)")
    print("="*60)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed, skipping...")
        return None

    print("Loading vLLM model...")
    load_start = time.time()

    # Load just the LLM part with vLLM
    llm = LLM(
        model="Qwen/Qwen3-4B",
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=MAX_NEW_TOKENS,
    )

    # Create text prompts (simulating what audio would produce)
    prompts = []
    for sample in samples:
        # Use the original question as a proxy for what audio encoding would produce
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)

    print(f"\nGenerating {len(prompts)} responses...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tokens_per_sec = total_tokens / gen_time if gen_time > 0 else 0

    print(f"\n--- vLLM Summary ---")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {gen_time:.2f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")

    for i, output in enumerate(outputs):
        print(f"\nSample {i+1}: {output.outputs[0].text[:100]}...")

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return {
        "total_tokens": total_tokens,
        "total_time": gen_time,
        "tokens_per_sec": tokens_per_sec,
        "load_time": load_time,
    }


def benchmark_hybrid(samples):
    """
    Hybrid approach:
    - HF Borealis for audio encoding (Whisper + Adapter)
    - vLLM for text generation

    This requires pre-computing audio embeddings and passing them to vLLM.
    Note: vLLM doesn't directly support this, so we measure the components separately.
    """
    print("\n" + "="*60)
    print("BENCHMARK: Hybrid (HF Audio Encoding + vLLM Generation)")
    print("="*60)

    # Load HF model for audio encoding only
    print("Loading HF model for audio encoding...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device=DEVICE,
    )
    model.eval()

    # Time audio encoding
    print("\nEncoding audio with Whisper + Adapter...")
    audio_times = []

    for i, sample in enumerate(samples):
        start = time.time()
        with torch.inference_mode():
            # Prepare audio
            mel = model.prepare_audio(sample["audio"])
            # Process through encoder and adapter
            audio_emb, audio_mask, _ = model._process_audio(mel)
        enc_time = time.time() - start
        audio_times.append(enc_time)
        print(f"Sample {i+1}: Audio encoding took {enc_time:.3f}s, shape: {audio_emb.shape}")

    avg_audio_time = sum(audio_times) / len(audio_times)
    print(f"\nAverage audio encoding time: {avg_audio_time:.3f}s")

    # For full hybrid, we would pass audio_emb to vLLM
    # But vLLM's embedding input API is limited, so we just report the breakdown

    del model
    torch.cuda.empty_cache()

    return {
        "avg_audio_encoding_time": avg_audio_time,
        "audio_times": audio_times,
    }


def main():
    print("="*60)
    print("Borealis Inference Benchmark: HF vs vLLM")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_ID}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Max tokens: {MAX_NEW_TOKENS}")

    # Load test data
    samples = load_test_audio()

    # Run benchmarks
    hf_results = benchmark_native_hf(samples)

    hybrid_results = benchmark_hybrid(samples)

    # Skip vLLM for now due to compatibility issues
    vllm_results = None
    try:
        vllm_results = benchmark_vllm(samples)
    except Exception as e:
        print(f"\nvLLM benchmark failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    print(f"\nNative HF:")
    print(f"  Load time: {hf_results['load_time']:.2f}s")
    print(f"  Generation: {hf_results['total_time']:.2f}s for {hf_results['total_tokens']} tokens")
    print(f"  Throughput: {hf_results['tokens_per_sec']:.2f} tokens/s")

    print(f"\nHybrid (Audio encoding only):")
    print(f"  Avg audio encoding: {hybrid_results['avg_audio_encoding_time']:.3f}s")

    if vllm_results:
        print(f"\nvLLM (text-only baseline):")
        print(f"  Load time: {vllm_results['load_time']:.2f}s")
        print(f"  Generation: {vllm_results['total_time']:.2f}s for {vllm_results['total_tokens']} tokens")
        print(f"  Throughput: {vllm_results['tokens_per_sec']:.2f} tokens/s")

        speedup = vllm_results['tokens_per_sec'] / hf_results['tokens_per_sec'] if hf_results['tokens_per_sec'] > 0 else 0
        print(f"\nvLLM speedup over HF: {speedup:.2f}x")


if __name__ == "__main__":
    main()
