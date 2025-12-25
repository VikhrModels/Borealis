"""Quick test of text-only data loading from GrandMaster-PRO-MAX."""
import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"

from datasets import load_dataset
from transformers import AutoTokenizer, WhisperFeatureExtractor
from borealis.dataset import BorealisTextOnlyDataset

# Load a small sample
print("Loading GrandMaster-PRO-MAX dataset...")
ds = load_dataset("Vikhrmodels/GrandMaster-PRO-MAX", split="train", streaming=True)

# Get first 5 samples
samples = []
for i, s in enumerate(ds):
    if i >= 5:
        break
    samples.append(s)
    print(f"Sample {i}: {s.keys()}")
    if 'conversation' in s:
        print(f"  Conversation: {s['conversation'][:2] if len(s['conversation']) >= 2 else s['conversation']}")
    if 'prompt_lang' in s:
        print(f"  Language: prompt={s.get('prompt_lang')}, answer={s.get('answer_lang')}")

print("\nTesting BorealisTextOnlyDataset...")
# Need to load non-streaming for dataset
ds_full = load_dataset("Vikhrmodels/GrandMaster-PRO-MAX", split="train")
ds_small = ds_full.select(range(min(100, len(ds_full))))

# Filter Russian
ds_ru = ds_small.filter(lambda x: x.get("prompt_lang") == "ru" or x.get("answer_lang") == "ru")
print(f"Filtered to {len(ds_ru)} Russian samples out of {len(ds_small)}")

# Initialize tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-4B", trust_remote_code=True)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

# Create dataset
text_ds = BorealisTextOnlyDataset(
    hf_dataset=ds_ru,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    max_text_len=1024,
)

print(f"TextOnlyDataset length: {len(text_ds)}")

# Test getting a sample
sample = text_ds[0]
print(f"\nSample keys: {sample.keys()}")
print(f"Mel shape: {sample['mel'][0].shape}")
print(f"Labels shape: {sample['labels'].shape}")
print(f"Is text only: {sample.get('is_text_only', False)}")

# Decode labels to see content
decoded = tokenizer.decode(sample['labels'], skip_special_tokens=False)
print(f"\nDecoded (first 500 chars):\n{decoded[:500]}")

print("\nText-only data loading test PASSED!")
