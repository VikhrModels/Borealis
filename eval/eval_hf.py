"""
Evaluate Borealis-5b-it on RuASRBenchmark using HuggingFace model.
"""

import os
import json
import string
import torch
from datasets import load_dataset, Audio
from jiwer import wer, cer
from tqdm.auto import tqdm
from transformers import AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clean_text_list(text_list):
    punct = set(string.punctuation)
    cleaned_list = [
        "".join(char for char in text.lower() if char not in punct)
        for text in text_list
    ]
    return cleaned_list


def extract_assistant_content(text: str) -> str:
    if "assistant\n" in text:
        return text.split("assistant\n")[-1].strip()
    return text.strip()


def extract_audio(audio_data):
    """Extract audio tensor from various formats."""
    if hasattr(audio_data, 'get_all_samples'):
        audio = audio_data.get_all_samples().data.squeeze().float()
    elif isinstance(audio_data, dict) and "array" in audio_data:
        import numpy as np
        arr = audio_data["array"]
        if isinstance(arr, np.ndarray):
            audio = torch.from_numpy(arr).float()
        else:
            audio = torch.tensor(arr).float()
    else:
        return None

    # Ensure 1D
    if audio.dim() > 1:
        audio = audio.mean(dim=0)
    return audio


def evaluate_on_benchmark(model, bench_set, device="cuda", batch_size=16, max_samples=1000):
    """Evaluate model on all benchmark splits with batching."""
    results = {}

    for split in bench_set:
        print(f"\nEvaluating on split: {split}")

        dataset = bench_set[split]
        # Limit to max_samples
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"  Limited to {max_samples} samples")
        ground_truth_texts = []
        generated_transcripts = []

        with torch.inference_mode():
            # Process in batches
            for batch_start in tqdm(range(0, len(dataset), batch_size), desc=f"Processing {split}"):
                batch_end = min(batch_start + batch_size, len(dataset))

                # Collect batch
                batch_audios = []
                batch_texts = []

                for i in range(batch_start, batch_end):
                    try:
                        sample = dataset[i]
                        audio = extract_audio(sample["audio"])
                        if audio is not None:
                            batch_audios.append(audio)
                            batch_texts.append(sample["text"])
                    except Exception as e:
                        continue

                if not batch_audios:
                    continue

                try:
                    # Generate for batch
                    outputs = model.generate(
                        audio=batch_audios,
                        user_prompt="Транскрибируй аудио: <|start_of_audio|><|end_of_audio|>",
                        system_prompt="Ты полезный голосовой ассистент. Точно транскрибируй аудио.",
                        max_new_tokens=320,
                        temperature=0.0,
                        do_sample=False,
                    )

                    # Decode outputs
                    for j, output in enumerate(outputs):
                        if hasattr(model, 'decode'):
                            transcript = model.decode(output)
                        else:
                            transcript = output

                        ground_truth_texts.append(batch_texts[j])
                        generated_transcripts.append(extract_assistant_content(str(transcript)))

                except Exception as e:
                    print(f"Batch error at {batch_start}: {e}")
                    continue

        if not ground_truth_texts:
            print(f"  No samples processed for {split}")
            continue

        # Compute metrics
        wer_score = wer(
            clean_text_list(ground_truth_texts),
            clean_text_list(generated_transcripts)
        )
        cer_score = cer(
            clean_text_list(ground_truth_texts),
            clean_text_list(generated_transcripts)
        )

        results[split] = {
            "wer": round(wer_score * 100, 2),
            "cer": round(cer_score * 100, 2),
            "samples": len(ground_truth_texts),
        }

        print(f"  WER: {wer_score*100:.2f}%  CER: {cer_score*100:.2f}%  ({len(ground_truth_texts)} samples)")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Checkpoints to evaluate: 1800, 2400, 2898 (baseline already done)
    checkpoints = [
        ("step-1800", "borealis_instruct_ckpts/checkpoint-1800/pytorch_model.bin"),
        ("step-2400", "borealis_instruct_ckpts/checkpoint-2400/pytorch_model.bin"),
        ("step-2898", "borealis_instruct_ckpts/checkpoint-2898/pytorch_model.bin"),
    ]

    # Load benchmark
    print("Loading RuASRBenchmark...")
    bench_set = load_dataset("Vikhrmodels/RuASRBenchmark", num_proc=4)
    bench_set = bench_set.cast_column("audio", Audio(decode=True, sampling_rate=16000))

    # Load existing results if available
    all_results = {}
    if os.path.exists("eval_results_hf.json"):
        with open("eval_results_hf.json", "r") as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")

    for ckpt_name, ckpt_path in checkpoints:
        print(f"\n{'='*80}")
        print(f"Evaluating checkpoint: {ckpt_name}")
        print(f"{'='*80}")

        # Load model from HuggingFace with local checkpoint
        print(f"Loading model with checkpoint: {ckpt_path}")
        model = AutoModel.from_pretrained(
            "Vikhrmodels/Borealis-5b-it",
            trust_remote_code=True,
            device=device,
            dtype="bfloat16",
        )

        # Load local checkpoint weights
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded local checkpoint: {ckpt_path}")
        else:
            print(f"Checkpoint not found: {ckpt_path}, using HF weights")

        model.eval()

        # Evaluate
        results = evaluate_on_benchmark(model, bench_set, device=device)
        all_results[ckpt_name] = results

        # Free memory
        del model
        torch.cuda.empty_cache()

        # Save intermediate results
        with open("eval_results_hf.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    splits = list(next(iter(all_results.values())).keys())
    print(f"{'Checkpoint':<12}", end="")
    for split in splits:
        print(f"{split[:12]:<14}", end="")
    print()
    print("-" * 100)

    for ckpt_name, results in all_results.items():
        print(f"{ckpt_name:<12}", end="")
        for split in splits:
            if split in results:
                print(f"W:{results[split]['wer']:>5.1f}%    ", end="")
        print()

    print(f"\nResults saved to eval_results_hf.json")


if __name__ == "__main__":
    main()
