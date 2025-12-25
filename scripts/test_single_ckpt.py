import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"
import sys
import gc

import torch
from datasets import load_dataset, Audio
from transformers import WhisperModel, WhisperFeatureExtractor, AutoTokenizer, Qwen3ForCausalLM
from borealis.modeling import BorealisForConditionalGeneration

# Get checkpoint from args
CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else "borealis_instruct_ckpts/checkpoint-300"
WHISPER_PATH = "openai/whisper-large-v3"
LLM_PATH = "./models/Qwen3-4B"
DEVICE = "cuda:7"  # Using GPU 7 which has more free memory

# Different prompts to test
PROMPTS = [
    ("Перескажи содержание аудио. <|start_of_audio|><|end_of_audio|>", "Ты полезный голосовой ассистент."),
    ("О чём говорится в этой аудиозаписи? <|start_of_audio|><|end_of_audio|>", "Ты полезный голосовой ассистент."),
    ("Сделай краткое резюме услышанного. <|start_of_audio|><|end_of_audio|>", "Ты полезный голосовой ассистент."),
    ("Какая основная тема обсуждается? <|start_of_audio|><|end_of_audio|>", "Ты эксперт по анализу аудио."),
    ("Ответь на вопрос по аудио: о чём идёт речь? <|start_of_audio|><|end_of_audio|>", "Ты внимательный слушатель."),
]

print(f"Testing checkpoint: {CHECKPOINT_PATH}")
print(f"Using device: {DEVICE}")
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
original_question = sample.get("question", "N/A")
original_answer = sample.get("answer", "N/A")

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

print("\n" + "="*80)
print(f"CHECKPOINT: {CHECKPOINT_PATH}")
print("="*80)

for prompt_idx, (user_prompt, system_prompt) in enumerate(PROMPTS):
    print(f"\n--- Prompt {prompt_idx + 1} ---")
    print(f"User: {user_prompt.replace('<|start_of_audio|><|end_of_audio|>', '[AUDIO]')}")
    print(f"System: {system_prompt}")

    torch.cuda.empty_cache()
    gc.collect()

    with torch.inference_mode():
        output = model.generate(
            mel=[mel.squeeze(0)],
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Response: {response}")

print("\n" + "="*80)
print("DONE")
print("="*80)
