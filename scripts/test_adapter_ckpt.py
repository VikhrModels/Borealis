"""Test adapter-only checkpoint on validation samples with IT prompts."""
import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"

import torch
from datasets import load_dataset, Audio
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel, Qwen3ForCausalLM
from borealis.modeling import BorealisForConditionalGeneration

# Load validation dataset
print("Loading validation dataset...")
val_ds = load_dataset("Vikhrmodels/Speech-Instructions", split="train")
val_ds = val_ds.cast_column("audio", Audio(sampling_rate=16000))
val_ds = val_ds.select(range(10))

# Load model components
print("Loading model components...")
whisper_encoder = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
audio_encoder = WhisperModel.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.bfloat16,
).encoder

language_model = Qwen3ForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|start_of_audio|>", "<|end_of_audio|>"]})

# Create model
print("Creating Borealis model...")
model = BorealisForConditionalGeneration(
    audio_encoder=audio_encoder,
    language_model=language_model,
    tokenizer=tokenizer
)

# Load checkpoint
checkpoint_path = "/home/alex/Borealis/borealis_adapter_only/checkpoint-2000/pytorch_model.bin"
print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
del checkpoint

model = model.cuda().eval()
device = next(model.parameters()).device

# Test prompts (IT-style)
test_prompts = [
    "What is being said in this audio?",
    "Summarize the main point of this audio.",
    "What emotion does the speaker convey?",
    "Answer in JSON format: {\"topic\": ..., \"sentiment\": ...}",
    "Translate this to English.",
    "What question is the speaker asking?",
    "Describe the speaker's tone.",
    "What is the key information in this audio?",
    "Is the speaker happy or sad? Explain.",
    "Extract the main idea from this audio.",
]

print("\n" + "="*80)
print("TESTING ADAPTER-ONLY CHECKPOINT (step 2000)")
print("="*80)

for i, sample in enumerate(val_ds):
    audio = sample["audio"]["array"]
    original_question = sample.get("question", "N/A")
    original_answer = sample.get("answer", "N/A")

    # Process audio - mel shape is (1, 128, 3000), need [[chunk]] format
    mel_features = whisper_encoder(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device, dtype=torch.bfloat16)
    # mel[b] should be list of chunks, each chunk is (128, 3000)
    mel = [[mel_features[0]]]  # [[chunk]] - single sample, single chunk

    # Use test prompt
    prompt = test_prompts[i % len(test_prompts)]

    print(f"\n--- Sample {i+1} ---")
    print(f"Original Q: {original_question[:100]}...")
    print(f"Original A: {original_answer[:100]}...")
    print(f"Test prompt: {prompt}")

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            mel=mel,
            user_prompt=prompt,
            system_prompt="You are a helpful voice assistant. Listen to the audio and respond appropriately.",
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Model response: {response[:300]}...")
    print("-" * 40)

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
