"""
Evaluate all instruct checkpoints on RuASRBenchmark.
Computes WER and CER for each checkpoint and split.
"""

import os
import sys
import json
import string
import torch
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, Qwen2ForCausalLM, AutoTokenizer
from datasets import load_dataset, Audio
from jiwer import wer, cer
from tqdm.auto import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import model
sys.path.insert(0, "/home/alex/Borealis")
from borealis.modeling import BorealisForConditionalGeneration
import numpy as np


class SimpleASRDataset:
    """Simple dataset for ASR evaluation."""
    def __init__(self, hf_dataset, tokenizer, feature_extractor, max_audio_len=30, sampling_rate=16000):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_audio_len = int(max_audio_len * sampling_rate)
        self.sr = sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Extract audio - handle different formats
        audio_data = example["audio"]

        # Handle AudioDecoder (torchcodec)
        if hasattr(audio_data, 'get_all_samples'):
            audio = audio_data.get_all_samples().data.squeeze().float()
        # Handle HuggingFace Audio format (dict with 'array')
        elif isinstance(audio_data, dict) and "array" in audio_data:
            audio = audio_data["array"]
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            else:
                audio = torch.tensor(audio).float()
        elif isinstance(audio_data, np.ndarray):
            audio = torch.from_numpy(audio_data).float()
        else:
            audio = torch.tensor(audio_data).float()

        # Ensure 1D
        if audio.dim() > 1:
            audio = audio.mean(dim=0)

        # Process to mel
        chunks = []
        for i in range(0, len(audio), self.max_audio_len):
            chunk = audio[i:i + self.max_audio_len]
            proc = self.feature_extractor(
                chunk.numpy(),
                sampling_rate=self.sr,
                padding="max_length",
                max_length=self.max_audio_len,
                truncation=True,
                return_attention_mask=False,
                return_tensors="pt",
            )
            chunks.append(proc.input_features.squeeze(0))

        return {
            "mel": chunks,
            "text": example["text"],
        }


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


def load_model(checkpoint_path, whisper_encoder, device="cuda"):
    """Load model with a specific checkpoint."""
    from transformers import AutoModelForCausalLM

    # Load language model
    language_model = AutoModelForCausalLM.from_pretrained(
        "./models/Qwen3-4B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "./models/Qwen3-4B",
        trust_remote_code=True,
    )

    start_audio_token = "<|start_of_audio|>"
    end_audio_token = "<|end_of_audio|>"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [start_audio_token, end_audio_token]}
    )

    # Create model with audio encoder
    model = BorealisForConditionalGeneration(
        audio_encoder=whisper_encoder,
        language_model=language_model,
        tokenizer=tokenizer
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model, tokenizer


def evaluate_on_benchmark(model, tokenizer, whisper_encoder, bench_set, device="cuda"):
    """Evaluate model on all benchmark splits."""
    results = {}

    for split in bench_set:
        print(f"\nEvaluating on split: {split}")

        dataset = SimpleASRDataset(
            hf_dataset=bench_set[split],
            tokenizer=tokenizer,
            feature_extractor=whisper_encoder,
            max_audio_len=30,
        )

        ground_truth_texts = []
        generated_transcripts = []

        with torch.inference_mode():
            for i in tqdm(range(len(dataset)), desc=f"Processing {split}"):
                try:
                    sample = dataset[i]
                    # mel is list of chunks, wrap in list for batch dim
                    mel = [[c.to(device) for c in sample["mel"]]]

                    transcripts = model.generate(
                        mel=mel,
                        max_new_tokens=320,
                        do_sample=False,  # Greedy for evaluation
                    )

                    ground_truth_texts.append(sample["text"])
                    generated_transcripts.extend(transcripts)
                except Exception as e:
                    print(f"Error on sample {i}: {e}")
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

        print(f"  WER: {wer_score*100:.2f}%  CER: {cer_score*100:.2f}%")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Checkpoints to evaluate
    checkpoints = [
        ("baseline", "checkpoints/pytorch_model.bin"),
        ("step-1800", "borealis_instruct_ckpts/checkpoint-1800/pytorch_model.bin"),
        ("step-2100", "borealis_instruct_ckpts/checkpoint-2100/pytorch_model.bin"),
        ("step-2400", "borealis_instruct_ckpts/checkpoint-2400/pytorch_model.bin"),
        ("step-2700", "borealis_instruct_ckpts/checkpoint-2700/pytorch_model.bin"),
        ("step-2898", "borealis_instruct_ckpts/checkpoint-2898/pytorch_model.bin"),
    ]

    # Load benchmark
    print("Loading RuASRBenchmark...")
    bench_set = load_dataset("Vikhrmodels/RuASRBenchmark", num_proc=4)
    bench_set = bench_set.cast_column("audio", Audio(decode=True, sampling_rate=16000))

    # Load whisper feature extractor for audio processing
    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    # Load whisper encoder model
    from transformers import WhisperModel
    print("Loading Whisper encoder...")
    whisper_model = WhisperModel.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.bfloat16,
    ).encoder.to(device)

    all_results = {}

    for name, ckpt_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping...")
            continue

        # Load model
        model, tokenizer = load_model(ckpt_path, whisper_model, device)

        # Evaluate
        results = evaluate_on_benchmark(
            model, tokenizer, whisper_feature_extractor, bench_set,
            device=device
        )

        all_results[name] = results

        # Free memory
        del model
        torch.cuda.empty_cache()

        # Save intermediate results
        with open("eval_results.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Header
    splits = list(next(iter(all_results.values())).keys())
    print(f"{'Checkpoint':<15}", end="")
    for split in splits:
        print(f"{split[:15]:<17}", end="")
    print()

    print("-" * 80)

    for name, results in all_results.items():
        print(f"{name:<15}", end="")
        for split in splits:
            if split in results:
                print(f"W:{results[split]['wer']:>5.1f} C:{results[split]['cer']:>5.1f}  ", end="")
            else:
                print(f"{'N/A':<17}", end="")
        print()

    print(f"\n{'='*80}")
    print("Results saved to eval_results.json")


if __name__ == "__main__":
    main()
