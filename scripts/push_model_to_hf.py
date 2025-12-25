"""Push Borealis checkpoint to HuggingFace Hub."""
import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, upload_file
from transformers import AutoTokenizer, WhisperModel, Qwen3ForCausalLM
from borealis.modeling import BorealisForConditionalGeneration

CHECKPOINT_PATH = "/home/alex/Borealis/borealis_adapter_only/checkpoint-26000/pytorch_model.bin"
HF_REPO = "Vikhrmodels/Borealis-5b-it"
OUTPUT_SAFETENSORS = "/tmp/borealis_model.safetensors"

print("Loading model components...")
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

print("Creating Borealis model...")
model = BorealisForConditionalGeneration(
    audio_encoder=audio_encoder,
    language_model=language_model,
    tokenizer=tokenizer
)

print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
del checkpoint

print("Converting to safetensors...")
state_dict = model.state_dict()
# Convert to float16 for smaller size
state_dict_fp16 = {k: v.half() if v.dtype == torch.bfloat16 else v for k, v in state_dict.items()}
save_file(state_dict_fp16, OUTPUT_SAFETENSORS)
print(f"Saved to {OUTPUT_SAFETENSORS}")

# Get file size
size_gb = os.path.getsize(OUTPUT_SAFETENSORS) / (1024**3)
print(f"File size: {size_gb:.2f} GB")

print(f"\nUploading to {HF_REPO}...")
api = HfApi()
api.upload_file(
    path_or_fileobj=OUTPUT_SAFETENSORS,
    path_in_repo="model.safetensors",
    repo_id=HF_REPO,
    repo_type="model",
)
print("Model uploaded!")

# Clean up
os.remove(OUTPUT_SAFETENSORS)
print("Done!")
