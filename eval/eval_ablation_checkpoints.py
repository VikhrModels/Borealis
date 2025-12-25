"""
Evaluate ablation checkpoints on RuASRBenchmark.
Usage: CUDA_VISIBLE_DEVICES=2,3 python eval_ablation_checkpoints.py
"""

import os
import json
import string
import argparse
import torch
from pathlib import Path
from datasets import load_dataset, Audio
from jiwer import wer, cer
from tqdm.auto import tqdm
from transformers import AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ABLATIONS_DIR = Path("/home/alex/Borealis/ablations/outputs")
BASE_MODEL = "Vikhrmodels/Borealis-5b-it"


def clean_text_list(text_list):
    punct = set(string.punctuation)
    return ["".join(char for char in text.lower() if char not in punct) for text in text_list]


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
        device=device,
    )

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

    user_prompt = "Transcribe this audio: <|start_of_audio|><|end_of_audio|>"
    system_prompt = "You are a speech recognition assistant. Accurately transcribe audio to text."

    for split in bench_set:
        print(f"\n  Split: {split}")

        dataset = bench_set[split]
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        ground_truth_texts = []
        generated_transcripts = []

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
                    gen_ids = model.generate(
                        audio=batch_audios,
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_new_tokens=256,
                        temperature=0.7,
                    )

                    for j, ids in enumerate(gen_ids):
                        transcript = model.decode(ids)
                        generated_transcripts.append(transcript)
                        ground_truth_texts.append(batch_texts[j])
                except Exception as e:
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


def find_all_checkpoints():
    """Find all checkpoints in ablation experiments."""
    checkpoints = []
    for exp_dir in ABLATIONS_DIR.iterdir():
        if exp_dir.is_dir():
            for ckpt_dir in sorted(exp_dir.glob("checkpoint-*")):
                if (ckpt_dir / "pytorch_model.bin").exists():
                    checkpoints.append({
                        "experiment": exp_dir.name,
                        "checkpoint": ckpt_dir.name,
                        "path": str(ckpt_dir),
                        "step": int(ckpt_dir.name.split("-")[1])
                    })
    return sorted(checkpoints, key=lambda x: (x["experiment"], x["step"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--experiment", type=str, default=None, help="Evaluate only this experiment")
    parser.add_argument("--checkpoint", type=str, default=None, help="Evaluate only this checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load benchmark once
    print("Loading RuASRBenchmark...")
    bench_set = load_dataset("Vikhrmodels/RuASRBenchmark", num_proc=4)
    bench_set = bench_set.cast_column("audio", Audio(decode=True, sampling_rate=16000))

    # Find checkpoints
    all_checkpoints = find_all_checkpoints()

    # Filter if specified
    if args.experiment:
        all_checkpoints = [c for c in all_checkpoints if c["experiment"] == args.experiment]
    if args.checkpoint:
        all_checkpoints = [c for c in all_checkpoints if c["checkpoint"] == args.checkpoint]

    print(f"\nFound {len(all_checkpoints)} checkpoints to evaluate:")
    for c in all_checkpoints:
        print(f"  {c['experiment']}/{c['checkpoint']}")

    all_results = {}
    results_file = ABLATIONS_DIR / "eval_results.json"

    # Load existing results if any
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)

    for ckpt_info in all_checkpoints:
        exp_name = ckpt_info["experiment"]
        ckpt_name = ckpt_info["checkpoint"]
        key = f"{exp_name}/{ckpt_name}"

        # Skip if already evaluated
        if key in all_results:
            print(f"\n{'='*60}")
            print(f"Skipping {key} (already evaluated)")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {key}")
        print(f"{'='*60}")

        try:
            model = load_checkpoint(ckpt_info["path"], device=device)
            results = evaluate_checkpoint(
                model, bench_set, device=device,
                batch_size=args.batch_size, max_samples=args.max_samples
            )

            # Calculate average WER
            wers = [s["wer"] for s in results.values()]
            avg_wer = sum(wers) / len(wers) if wers else float('inf')

            all_results[key] = {
                "experiment": exp_name,
                "checkpoint": ckpt_name,
                "step": ckpt_info["step"],
                "avg_wer": round(avg_wer, 2),
                "splits": results
            }

            print(f"\n>>> {key}: Average WER = {avg_wer:.2f}%")

            # Save after each checkpoint
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {results_file}")

            # Free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {key}: {e}")
            continue

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, res in sorted(all_results.items(), key=lambda x: x[1].get("avg_wer", 999)):
        print(f"{key}: WER = {res['avg_wer']:.2f}%")


if __name__ == "__main__":
    main()
