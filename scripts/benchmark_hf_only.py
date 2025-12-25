"""
Benchmark: Native HuggingFace inference for Borealis
"""

import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"

import time
import torch
from transformers import AutoModel
from datasets import load_dataset, Audio

DEVICE = "cuda"
MODEL_ID = "Vikhrmodels/Borealis-5b-it"
NUM_SAMPLES = 3
MAX_NEW_TOKENS = 128


def main():
    print("="*60)
    print("Borealis HF Inference Benchmark")
    print("="*60)

    # Load test audio
    print("\nLoading test audio...")
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
        })
    print(f"Loaded {len(samples)} samples")

    # Load model
    print("\nLoading model...")
    load_start = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device=DEVICE,
    )
    model.eval()
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Benchmark
    total_tokens = 0
    total_time = 0

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)} ---")
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

        print(f"Tokens: {num_tokens}, Time: {gen_time:.2f}s")
        print(f"Response: {response[:150]}...")

    # Summary
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model load time: {load_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")
    print(f"Avg time per sample: {total_time/len(samples):.2f}s")


if __name__ == "__main__":
    main()
