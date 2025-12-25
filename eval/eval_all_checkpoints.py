"""
Evaluate all Borealis checkpoints on RuASRBenchmark.
"""

import os
import json
import string
import torch
from pathlib import Path
from datasets import load_dataset, Audio
from jiwer import wer, cer
from tqdm.auto import tqdm
from transformers import AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHECKPOINTS_DIR = Path("/home/alex/Borealis/borealis_ru_it")
BASE_MODEL = "Vikhrmodels/Borealis-5b-it"
RESULTS_FILE = "eval_results_all_checkpoints.json"


def clean_text_list(text_list):
    punct = set(string.punctuation)
    return ["".join(char for char in text.lower() if char not in punct) for text in text_list]


def extract_assistant_content(text: str) -> str:
    if "assistant\n" in text:
        return text.split("assistant\n")[-1].strip()
    return text.strip()


def extract_audio(audio_data):
    """Extract audio tensor from various formats."""
    import numpy as np
    if hasattr(audio_data, 'get_all_samples'):
        audio = audio_data.get_all_samples().data.squeeze().float()
    elif isinstance(audio_data, dict) and "array" in audio_data:
        arr = audio_data["array"]
        if isinstance(arr, np.ndarray):
            audio = torch.from_numpy(arr).float()
        else:
            audio = torch.tensor(arr).float()
    else:
        return None
    if audio.dim() > 1:
        audio = audio.mean(dim=0)
    return audio


def load_checkpoint(ckpt_path, device="cuda"):
    """Load base model and apply checkpoint weights."""
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModel.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        device=device,  # Pass device directly to from_pretrained
    )

    # Load checkpoint weights
    ckpt_file = Path(ckpt_path) / "pytorch_model.bin"
    if ckpt_file.exists():
        print(f"Loading weights from: {ckpt_file}")
        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: No pytorch_model.bin found in {ckpt_path}")

    model.eval()
    return model


def evaluate_checkpoint(model, bench_set, device="cuda", max_samples=1000, batch_size=8):
    """Evaluate a loaded model with batched inference."""
    results = {}

    # Transcription prompt (from model card)
    user_prompt = "Transcribe this audio: <|start_of_audio|><|end_of_audio|>"
    system_prompt = "You are a speech recognition assistant. Accurately transcribe audio to text."

    for split in bench_set:
        print(f"\n  Split: {split}")

        dataset = bench_set[split]
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        ground_truth_texts = []
        generated_transcripts = []

        # Process in batches
        num_batches = (len(dataset) + batch_size - 1) // batch_size

        with torch.inference_mode():
            for batch_idx in tqdm(range(num_batches), desc=f"    {split}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))

                batch_audios = []
                batch_texts = []

                for i in range(start_idx, end_idx):
                    try:
                        sample = dataset[i]
                        audio = extract_audio(sample["audio"])
                        if audio is None:
                            continue
                        batch_audios.append(audio)
                        batch_texts.append(sample["text"])
                    except Exception:
                        continue

                if not batch_audios:
                    continue

                try:
                    # Generate transcriptions for the batch
                    gen_ids = model.generate(
                        audio=batch_audios,
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_new_tokens=256,
                        temperature=0.7,
                    )

                    # Decode all outputs
                    for j, ids in enumerate(gen_ids):
                        transcript = model.decode(ids)
                        generated_transcripts.append(transcript)
                        ground_truth_texts.append(batch_texts[j])
                except Exception as e:
                    # Fallback to single sample processing on error
                    for audio, text in zip(batch_audios, batch_texts):
                        try:
                            gen_ids = model.generate(
                                audio=audio,
                                user_prompt=user_prompt,
                                system_prompt=system_prompt,
                                max_new_tokens=256,
                                temperature=0.7,
                            )
                            transcript = model.decode(gen_ids[0])
                            generated_transcripts.append(transcript)
                            ground_truth_texts.append(text)
                        except Exception:
                            continue

        if not ground_truth_texts:
            continue

        wer_score = wer(clean_text_list(ground_truth_texts), clean_text_list(generated_transcripts))
        cer_score = cer(clean_text_list(ground_truth_texts), clean_text_list(generated_transcripts))

        results[split] = {
            "wer": round(wer_score * 100, 2),
            "cer": round(cer_score * 100, 2),
            "samples": len(ground_truth_texts),
        }
        print(f"    WER: {wer_score*100:.2f}%  CER: {cer_score*100:.2f}%")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load benchmark
    print("Loading RuASRBenchmark...")
    bench_set = load_dataset("Vikhrmodels/RuASRBenchmark", num_proc=4)
    bench_set = bench_set.cast_column("audio", Audio(decode=True, sampling_rate=16000))

    # Find checkpoints
    checkpoints = sorted(CHECKPOINTS_DIR.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    print(f"Found {len(checkpoints)} checkpoints: {[c.name for c in checkpoints]}")

    # Load existing results
    all_results = {}
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, "r") as f:
            all_results = json.load(f)
        print(f"Loaded existing results for: {list(all_results.keys())}")

    # Evaluate each checkpoint
    for ckpt_path in checkpoints:
        ckpt_name = ckpt_path.name

        if ckpt_name in all_results:
            print(f"\nSkipping {ckpt_name} (already evaluated)")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name}")
        print(f"{'='*60}")

        model = load_checkpoint(str(ckpt_path), device=device)
        results = evaluate_checkpoint(model, bench_set, device=device, batch_size=8)
        all_results[ckpt_name] = results

        # Cleanup
        del model
        torch.cuda.empty_cache()

        # Save after each checkpoint
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to {RESULTS_FILE}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Calculate average WER for each checkpoint
    avg_wers = {}
    for ckpt, splits in all_results.items():
        wers = [s["wer"] for s in splits.values()]
        avg_wers[ckpt] = sum(wers) / len(wers) if wers else float('inf')

    # Sort by average WER
    sorted_ckpts = sorted(avg_wers.items(), key=lambda x: x[1])

    print("\nCheckpoints by Average WER:")
    for ckpt, avg in sorted_ckpts:
        print(f"  {ckpt}: {avg:.2f}%")

    best_ckpt = sorted_ckpts[0][0]
    print(f"\nBest checkpoint: {best_ckpt} (avg WER: {sorted_ckpts[0][1]:.2f}%)")

    # Print detailed results for best
    print(f"\nDetailed results for {best_ckpt}:")
    for split, metrics in all_results[best_ckpt].items():
        print(f"  {split}: WER={metrics['wer']:.2f}% CER={metrics['cer']:.2f}%")


if __name__ == "__main__":
    main()
