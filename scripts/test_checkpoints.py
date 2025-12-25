import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"

import torch
from datasets import load_dataset, Audio
from transformers import WhisperModel, WhisperFeatureExtractor, AutoTokenizer, Qwen3ForCausalLM
from borealis.modeling import BorealisForConditionalGeneration

# Config
CHECKPOINTS = [
    "/home/alex/Borealis/borealis_instruct_ckpts/checkpoint-2100",
    "/home/alex/Borealis/borealis_instruct_ckpts/checkpoint-2700",
    "/home/alex/Borealis/borealis_instruct_ckpts/checkpoint-2898",
]
WHISPER_PATH = "openai/whisper-large-v3"
LLM_PATH = "./models/Qwen3-4B"
DEVICE = "cuda:0"

# Different prompts to test
PROMPTS = [
    ("Перескажи содержание аудио. <|start_of_audio|><|end_of_audio|>", "Ты полезный голосовой ассистент."),
    ("О чём говорится в этой аудиозаписи? <|start_of_audio|><|end_of_audio|>", "Ты полезный голосовой ассистент."),
    ("Сделай краткое резюме услышанного. <|start_of_audio|><|end_of_audio|>", "Ты полезный голосовой ассистент."),
    ("Какая основная тема обсуждается? <|start_of_audio|><|end_of_audio|>", "Ты эксперт по анализу аудио."),
    ("Расскажи подробно, что ты слышишь. <|start_of_audio|><|end_of_audio|>", "Ты внимательный слушатель."),
]

print("Loading base models...")

# Load whisper
whisper_encoder = WhisperModel.from_pretrained(
    WHISPER_PATH, torch_dtype=torch.bfloat16
).encoder.to(DEVICE)
feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_PATH)

# Load LLM
llm_base = Qwen3ForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|start_of_audio|>", "<|end_of_audio|>"]
})

print("Loading audio samples from dataset...")
ds = load_dataset("Vikhrmodels/Speech-Instructions", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# Get a few samples
samples = []
for i, sample in enumerate(ds):
    if i >= 3:
        break
    samples.append(sample)

print(f"Loaded {len(samples)} audio samples")

# Process audio for all samples
processed_samples = []
for idx, sample in enumerate(samples):
    audio_array = sample["audio"]["array"]
    audio_tensor = torch.tensor(audio_array).float()
    mel = feature_extractor(
        audio_tensor.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        max_length=30 * 16000,
        truncation=True,
    ).input_features.to(DEVICE)
    processed_samples.append({
        "mel": mel,
        "question": sample.get("question", "N/A"),
        "answer": sample.get("answer", "N/A")[:200]
    })

# Test each checkpoint
for ckpt_path in CHECKPOINTS:
    print("\n" + "="*80)
    print(f"TESTING CHECKPOINT: {ckpt_path}")
    print("="*80)

    # Reload LLM for fresh state
    llm = Qwen3ForCausalLM.from_pretrained(
        LLM_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Create model
    model = BorealisForConditionalGeneration(
        audio_encoder=whisper_encoder,
        language_model=llm,
        tokenizer=tokenizer
    )

    # Load checkpoint
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(f"{ckpt_path}/pytorch_model.bin", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(DEVICE)
    model.eval()

    # Test with first sample and all prompts
    sample_data = processed_samples[0]
    print(f"\nOriginal question: {sample_data['question']}")
    print(f"Original answer: {sample_data['answer']}...")

    for prompt_idx, (user_prompt, system_prompt) in enumerate(PROMPTS):
        print(f"\n--- Prompt {prompt_idx + 1}: {user_prompt[:50]}...")

        with torch.inference_mode():
            output = model.generate(
                mel=[sample_data["mel"].squeeze(0)],
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Response: {response}")

    # Clean up
    del model
    del llm
    torch.cuda.empty_cache()

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
