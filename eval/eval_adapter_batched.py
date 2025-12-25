"""Evaluate adapter-only checkpoint on RuASRBenchmark with batched inference."""
import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import string
import json
from transformers import WhisperFeatureExtractor, WhisperModel, AutoTokenizer, Qwen3ForCausalLM
from datasets import load_dataset, Audio
from jiwer import wer, cer
from tqdm import tqdm

from borealis.modeling import BorealisForConditionalGeneration

if len(sys.argv) < 2:
    print("Usage: python eval_adapter_batched.py <checkpoint_path> [batch_size]")
    sys.exit(1)

CHECKPOINT = sys.argv[1]
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 8
DEVICE = "cuda"

ckpt_name = CHECKPOINT.split("/")[-2] if CHECKPOINT.endswith(".bin") else CHECKPOINT.split("/")[-1]

def clean_text(text):
    punct = set(string.punctuation)
    return "".join(char for char in text.lower() if char not in punct)

print(f"Loading benchmark for {ckpt_name} (batch_size={BATCH_SIZE})...")
bench = load_dataset("Vikhrmodels/RuASRBenchmark")
bench = bench.cast_column("audio", Audio(sampling_rate=16000))

print("Loading model components...")
whisper_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
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

# Compile for faster inference
print("Compiling model...")
# model.llm = torch.compile(model.llm, mode="reduce-overhead")

results = {}

for split in bench:
    print(f"\n=== [{ckpt_name}] Evaluating {split} ({len(bench[split])} samples) ===")

    refs = []
    hyps = []

    samples = list(bench[split])
    num_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc=f"{ckpt_name}/{split}"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(samples))
        batch_samples = samples[start_idx:end_idx]

        # Process batch
        batch_mels = []
        batch_refs = []

        for sample in batch_samples:
            audio = sample["audio"]["array"]
            ref = sample["text"]
            batch_refs.append(ref)

            mel = whisper_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(DEVICE, dtype=torch.bfloat16)
            batch_mels.append([mel[0]])

        with torch.inference_mode():
            out = model.generate(
                mel=batch_mels,
                user_prompt="Транскрибируй аудио. /no_think",
                system_prompt="Транскрибируй аудио в текст.",
                max_new_tokens=256,
                do_sample=False,
            )

        for i, gen_ids in enumerate(out):
            hyp = tokenizer.decode(gen_ids, skip_special_tokens=True)
            hyp = hyp.replace("<think>", "").replace("</think>", "").strip()
            hyps.append(hyp)

        refs.extend(batch_refs)

    # Calculate metrics
    refs_clean = [clean_text(r) for r in refs]
    hyps_clean = [clean_text(h) for h in hyps]

    wer_score = wer(refs_clean, hyps_clean) * 100
    cer_score = cer(refs_clean, hyps_clean) * 100

    results[split] = {"wer": round(wer_score, 2), "cer": round(cer_score, 2), "samples": len(refs)}
    print(f"  [{ckpt_name}] WER: {wer_score:.2f}%  CER: {cer_score:.2f}%")

    # Print some examples
    print("\n  Examples:")
    for i in range(min(3, len(refs))):
        print(f"    REF: {refs[i][:80]}")
        print(f"    HYP: {hyps[i][:80]}")
        print()

print("\n" + "="*60)
print(f"[{ckpt_name}] SUMMARY")
print("="*60)
for split, res in results.items():
    print(f"{split:20} WER: {res['wer']:6.2f}%  CER: {res['cer']:6.2f}%  ({res['samples']} samples)")

output_file = f"eval_{ckpt_name}_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {output_file}")
