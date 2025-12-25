"""Evaluate Voxtral-Mini-3B on BigBenchAudio benchmark."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
import re
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

DEVICE = "cuda"
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"

print(f"Loading Voxtral-Mini-3B...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = VoxtralForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE
)

print("Loading BigBenchAudio...")
dataset = load_dataset("ArtificialAnalysis/big_bench_audio", split="train")

# System prompts per category
CATEGORY_PROMPTS = {
    "formal_fallacies": "Listen to the logical argument and determine if it is valid or invalid. Answer with only 'valid' or 'invalid'.",
    "navigate": "Listen to the navigation instructions and determine the final position. Answer with only 'Yes' or 'No' for whether the agent returns to the starting point.",
    "object_counting": "Listen to the description and count the specified objects. Answer with only the number.",
    "web_of_lies": "Listen to the statements about truth-telling and lying. Determine if the final person tells the truth. Answer with only 'Yes' or 'No'.",
}

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = ' '.join(answer.split())

    if answer in ['yes', 'true', 'correct']:
        return 'yes'
    if answer in ['no', 'false', 'incorrect']:
        return 'no'
    if answer in ['valid', 'the argument is valid']:
        return 'valid'
    if answer in ['invalid', 'the argument is invalid']:
        return 'invalid'

    numbers = re.findall(r'\b(\d+)\b', answer)
    if numbers:
        return numbers[-1]

    return answer

def extract_answer(response: str, category: str) -> str:
    """Extract the answer from model response."""
    response = response.strip()

    patterns = [
        r"(?:the answer is|answer:|final answer:?)\s*[:\-]?\s*(.+?)(?:\.|$)",
        r"(?:therefore|thus|so|hence)[,\s]+(.+?)(?:\.|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            return normalize_answer(match.group(1))

    if len(response.split()) <= 5:
        return normalize_answer(response)

    sentences = re.split(r'[.!?]', response)
    if sentences:
        return normalize_answer(sentences[-1] if sentences[-1].strip() else sentences[-2] if len(sentences) > 1 else response)

    return normalize_answer(response)

results = defaultdict(lambda: {"correct": 0, "total": 0, "examples": []})

print(f"\n=== Evaluating Voxtral-Mini-3B on BigBenchAudio ({len(dataset)} samples) ===\n")

import tempfile
import soundfile as sf

for sample in tqdm(dataset, desc="Voxtral/BigBenchAudio"):
    category = sample["category"]
    official_answer = sample["official_answer"].lower().strip()
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    # Save audio to temp file (Voxtral expects file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        audio_path = f.name

    system_prompt = CATEGORY_PROMPTS.get(category, "Listen and answer the question accurately.")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": f"{system_prompt}\n\nAnswer:"},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(DEVICE, dtype=torch.bfloat16)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.0, do_sample=False)

        response = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
    except Exception as e:
        print(f"Error: {e}")
        response = ""
    finally:
        os.unlink(audio_path)

    predicted = extract_answer(response, category)
    is_correct = predicted == official_answer

    results[category]["total"] += 1
    if is_correct:
        results[category]["correct"] += 1

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
print("Voxtral-Mini-3B BigBenchAudio RESULTS")
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

output = {
    "model": "Voxtral-Mini-3B-2507",
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

with open("eval_voxtral_bigbench_audio.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to eval_voxtral_bigbench_audio.json")
