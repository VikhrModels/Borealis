"""
Evaluate Whisper-large-v3 on RuASRBenchmark.
"""

import os
import json
import string
import torch
from datasets import load_dataset, Audio
from jiwer import wer, cer
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clean_text_list(text_list):
    punct = set(string.punctuation)
    cleaned_list = [
        "".join(char for char in text.lower() if char not in punct)
        for text in text_list
    ]
    return cleaned_list


def extract_audio(audio_data):
    """Extract audio array from various formats."""
    if hasattr(audio_data, 'get_all_samples'):
        audio = audio_data.get_all_samples().data.squeeze().float().numpy()
    elif isinstance(audio_data, dict) and "array" in audio_data:
        audio = audio_data["array"]
    else:
        return None
    return audio


def evaluate_whisper(model, processor, bench_set, device="cuda", max_samples=1000):
    """Evaluate Whisper on all benchmark splits."""
    results = {}

    for split in bench_set:
        print(f"\nEvaluating on split: {split}")

        dataset = bench_set[split]
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"  Limited to {max_samples} samples")

        ground_truth_texts = []
        generated_transcripts = []

        with torch.inference_mode():
            for i in tqdm(range(len(dataset)), desc=f"Processing {split}"):
                try:
                    sample = dataset[i]
                    audio = extract_audio(sample["audio"])
                    if audio is None:
                        continue

                    # Process audio
                    inputs = processor(
                        audio,
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features.to(device, dtype=torch.bfloat16)

                    # Generate
                    generated_ids = model.generate(
                        inputs,
                        max_new_tokens=256,
                        language="ru",
                        task="transcribe",
                    )

                    # Decode
                    transcript = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]

                    ground_truth_texts.append(sample["text"])
                    generated_transcripts.append(transcript)

                except Exception as e:
                    print(f"Error on sample {i}: {e}")
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

    # Load Whisper
    print("Loading Whisper-large-v3...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Load benchmark
    print("Loading RuASRBenchmark...")
    bench_set = load_dataset("Vikhrmodels/RuASRBenchmark", num_proc=4)
    bench_set = bench_set.cast_column("audio", Audio(decode=True, sampling_rate=16000))

    # Evaluate
    results = evaluate_whisper(model, processor, bench_set, device=device, max_samples=1000)

    # Save results
    with open("eval_results_whisper.json", "w") as f:
        json.dump({"whisper-large-v3": results}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to eval_results_whisper.json")

    # Print summary
    print(f"\n{'='*60}")
    print("WHISPER-LARGE-V3 RESULTS")
    print(f"{'='*60}")
    for split, res in results.items():
        print(f"{split}: WER={res['wer']:.2f}% CER={res['cer']:.2f}%")


if __name__ == "__main__":
    main()
