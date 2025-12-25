"""Evaluate Borealis on BigBenchAudio benchmark."""
import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import json
import re
from transformers import WhisperFeatureExtractor, WhisperModel, AutoTokenizer, Qwen3ForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

from borealis.modeling import BorealisForConditionalGeneration

if len(sys.argv) < 2:
    print("Usage: python eval_bigbench_audio.py <checkpoint_path>")
    sys.exit(1)

CHECKPOINT = sys.argv[1]
DEVICE = "cuda"

ckpt_name = CHECKPOINT.split("/")[-2] if CHECKPOINT.endswith(".bin") else CHECKPOINT.split("/")[-1]

print(f"Loading BigBenchAudio for {ckpt_name}...")
dataset = load_dataset("ArtificialAnalysis/big_bench_audio", split="train")

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

# System prompts per category
CATEGORY_PROMPTS = {
    "formal_fallacies": "Listen to the logical argument and determine if it is valid or invalid. Answer with only 'valid' or 'invalid'.",
    "navigate": "Listen to the navigation instructions and determine the final position. Answer with only 'Yes' or 'No' for whether the agent returns to the starting point.",
    "object_counting": "Listen to the description and count the specified objects. Answer with only the number.",
    "web_of_lies": "Listen to the statements about truth-telling and lying. Determine if the final person tells the truth. Answer with only 'Yes' or 'No'.",
}

USER_PROMPT = "Listen carefully and answer the question. /no_think"

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    # Remove punctuation and extra spaces
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = ' '.join(answer.split())

    # Common normalizations
    if answer in ['yes', 'true', 'correct']:
        return 'yes'
    if answer in ['no', 'false', 'incorrect']:
        return 'no'
    if answer in ['valid', 'the argument is valid']:
        return 'valid'
    if answer in ['invalid', 'the argument is invalid']:
        return 'invalid'

    # Try to extract number
    numbers = re.findall(r'\b(\d+)\b', answer)
    if numbers:
        return numbers[-1]  # Take last number mentioned

    return answer

def extract_answer(response: str, category: str) -> str:
    """Extract the answer from model response."""
    response = response.replace("<think>", "").replace("</think>", "").strip()

    # Try to find answer after common patterns
    patterns = [
        r"(?:the answer is|answer:|final answer:?)\s*[:\-]?\s*(.+?)(?:\.|$)",
        r"(?:therefore|thus|so|hence)[,\s]+(.+?)(?:\.|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            return normalize_answer(match.group(1))

    # For short responses, use the whole thing
    if len(response.split()) <= 5:
        return normalize_answer(response)

    # Take the last sentence/phrase
    sentences = re.split(r'[.!?]', response)
    if sentences:
        return normalize_answer(sentences[-1] if sentences[-1].strip() else sentences[-2] if len(sentences) > 1 else response)

    return normalize_answer(response)

results = defaultdict(lambda: {"correct": 0, "total": 0, "examples": []})

print(f"\n=== [{ckpt_name}] Evaluating BigBenchAudio ({len(dataset)} samples) ===\n")

for sample in tqdm(dataset, desc=f"{ckpt_name}/BigBenchAudio"):
    category = sample["category"]
    official_answer = sample["official_answer"].lower().strip()
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    # Resample if needed
    if sr != 16000:
        import torchaudio
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
        audio = audio_tensor.squeeze(0).numpy()

    # Process audio
    mel = whisper_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(DEVICE, dtype=torch.bfloat16)
    mel = [[mel[0]]]

    system_prompt = CATEGORY_PROMPTS.get(category, "Listen and answer the question accurately.")

    with torch.inference_mode():
        out = model.generate(
            mel=mel,
            user_prompt=USER_PROMPT,
            system_prompt=system_prompt,
            max_new_tokens=64,
            do_sample=False,
        )

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    response = response.replace("<think>", "").replace("</think>", "").strip()

    predicted = extract_answer(response, category)

    # Exact match only (avoid substring issues like "valid" in "invalid")
    is_correct = predicted == official_answer

    results[category]["total"] += 1
    if is_correct:
        results[category]["correct"] += 1

    # Save some examples
    if len(results[category]["examples"]) < 3:
        results[category]["examples"].append({
            "id": sample["id"],
            "official": official_answer,
            "predicted": predicted,
            "response": response[:200],
            "correct": is_correct
        })

# Print results
print("\n" + "="*60)
print(f"[{ckpt_name}] BigBenchAudio RESULTS")
print("="*60)

total_correct = 0
total_samples = 0

for category, data in sorted(results.items()):
    acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
    total_correct += data["correct"]
    total_samples += data["total"]
    print(f"\n{category}:")
    print(f"  Accuracy: {acc:.1f}% ({data['correct']}/{data['total']})")
    print("  Examples:")
    for ex in data["examples"]:
        status = "✓" if ex["correct"] else "✗"
        print(f"    {status} Official: {ex['official']}, Predicted: {ex['predicted']}")
        print(f"      Response: {ex['response'][:100]}...")

overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {overall_acc:.1f}% ({total_correct}/{total_samples})")
print("="*60)

# Save results
output = {
    "checkpoint": ckpt_name,
    "overall_accuracy": round(overall_acc, 2),
    "total_correct": total_correct,
    "total_samples": total_samples,
    "categories": {
        cat: {
            "accuracy": round(data["correct"] / data["total"] * 100, 2) if data["total"] > 0 else 0,
            "correct": data["correct"],
            "total": data["total"]
        }
        for cat, data in results.items()
    }
}

output_file = f"eval_{ckpt_name}_bigbench_audio.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {output_file}")
