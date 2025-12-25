import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"

import torch
from datasets import load_dataset, Audio
from transformers import WhisperModel, WhisperFeatureExtractor, AutoTokenizer, Qwen3ForCausalLM
from borealis.modeling import BorealisForConditionalGeneration

# Config
CHECKPOINT_PATH = "borealis_instruct_ckpts/checkpoint-2607"
WHISPER_PATH = "openai/whisper-large-v3"
LLM_PATH = "./models/Qwen3-4B"
DEVICE = "cuda:0"

print("Loading models...")

# Load whisper
whisper_encoder = WhisperModel.from_pretrained(
    WHISPER_PATH, torch_dtype=torch.bfloat16
).encoder.to(DEVICE)
feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_PATH)

# Load LLM
llm = Qwen3ForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|start_of_audio|>", "<|end_of_audio|>"]
})

# Create model
model = BorealisForConditionalGeneration(
    audio_encoder=whisper_encoder,
    language_model=llm,
    tokenizer=tokenizer
)

# Load checkpoint
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
ckpt = torch.load(f"{CHECKPOINT_PATH}/pytorch_model.bin", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt, strict=False)
model = model.to(DEVICE)
model.eval()

print("Loading audio sample from dataset...")
ds = load_dataset("Vikhrmodels/Speech-Instructions", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# Get first sample
sample = next(iter(ds))
audio_array = sample["audio"]["array"]
original_question = sample["question"]
original_answer = sample["answer"]

print(f"\nOriginal question: {original_question}")
print(f"Original answer: {original_answer[:200]}...")

# Process audio
audio_tensor = torch.tensor(audio_array).float()
mel = feature_extractor(
    audio_tensor.numpy(),
    sampling_rate=16000,
    return_tensors="pt",
    padding="max_length",
    max_length=30 * 16000,
    truncation=True,
).input_features.to(DEVICE)

# Custom prompt
custom_prompt = "Перескажи что происходит. <|start_of_audio|><|end_of_audio|>"
system_prompt = "Ты полезный голосовой ассистент."

print(f"\nCustom prompt: {custom_prompt}")
print("\nGenerating response...")

with torch.inference_mode():
    output = model.generate(
        mel=[mel.squeeze(0)],
        user_prompt=custom_prompt,
        system_prompt=system_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nModel response: {response}")
