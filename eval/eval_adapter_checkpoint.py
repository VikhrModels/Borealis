"""Evaluate adapter-only checkpoint on RuASRBenchmark."""
import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import string
import json
from transformers import WhisperFeatureExtractor, WhisperModel, AutoTokenizer, Qwen3ForCausalLM
from datasets import load_dataset, Audio
from jiwer import wer, cer
from tqdm import tqdm
import numpy as np

from borealis.modeling import BorealisForConditionalGeneration

CHECKPOINT = "/home/alex/Borealis/borealis_adapter_only/checkpoint-30000/pytorch_model.bin"
DEVICE = "cuda"

def clean_text(text):
    punct = set(string.punctuation)
    return "".join(char for char in text.lower() if char not in punct)

print("Loading benchmark...")
bench = load_dataset("Vikhrmodels/RuASRBenchmark")
bench = bench.cast_column("audio", Audio(sampling_rate=16000))

print("Loading model components...")
whisper_encoder = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
audio_encoder = WhisperModel.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.bfloat16
).encoder.to(DEVICE)

language_model = Qwen3ForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|start_of_audio|>", "<|end_of_audio|>"]})

print("Creating Borealis model...")
model = BorealisForConditionalGeneration(
    audio_encoder=audio_encoder,
    language_model=language_model,
    tokenizer=tokenizer
)

print(f"Loading checkpoint: {CHECKPOINT}")
state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
del state_dict

model = model.to(DEVICE).eval()

results = {}

for split in bench:
    print(f"\n=== Evaluating {split} ({len(bench[split])} samples) ===")

    refs = []
    hyps = []

    for sample in tqdm(bench[split], desc=split):
        audio = sample["audio"]["array"]
        ref = sample["text"]

        # Process audio
        mel = whisper_encoder(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE, dtype=torch.bfloat16)
        mel = [[mel[0]]]

        with torch.inference_mode():
            out = model.generate(
                mel=mel,
                user_prompt="Транскрибируй аудио. /no_think",
                system_prompt="Транскрибируй аудио в текст.",
                max_new_tokens=256,
                do_sample=False,
            )

        hyp = tokenizer.decode(out[0], skip_special_tokens=True)
        hyp = hyp.replace("<think>", "").replace("</think>", "").strip()

        refs.append(ref)
        hyps.append(hyp)

    # Calculate metrics
    refs_clean = [clean_text(r) for r in refs]
    hyps_clean = [clean_text(h) for h in hyps]

    wer_score = wer(refs_clean, hyps_clean) * 100
    cer_score = cer(refs_clean, hyps_clean) * 100

    results[split] = {"wer": round(wer_score, 2), "cer": round(cer_score, 2), "samples": len(refs)}
    print(f"  WER: {wer_score:.2f}%  CER: {cer_score:.2f}%")

    # Print some examples
    print("\n  Examples:")
    for i in range(min(3, len(refs))):
        print(f"    REF: {refs[i][:80]}")
        print(f"    HYP: {hyps[i][:80]}")
        print()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for split, res in results.items():
    print(f"{split:20} WER: {res['wer']:6.2f}%  CER: {res['cer']:6.2f}%  ({res['samples']} samples)")

with open("eval_adapter_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to eval_adapter_results.json")
