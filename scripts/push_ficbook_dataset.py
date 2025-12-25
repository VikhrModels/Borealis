"""
Push Ficbook Instruct dataset to HuggingFace.
"""

import json
import os
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import HfApi

# Configuration
DATASET_DIR = Path("ficbook_instruct_10k")
HF_REPO = "Vikhrmodels/Ficbook-Audio-Instruct-10K"


def load_dataset_from_files():
    """Load dataset from generated files."""
    # Load instructions
    instructions_file = DATASET_DIR / "instructions.json"
    with open(instructions_file, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    # Create mapping by idx
    instructions_map = {item["idx"]: item for item in instructions}

    # Load TTS metadata
    tts_file = DATASET_DIR / "tts_results.json"
    with open(tts_file, "r", encoding="utf-8") as f:
        tts_results = json.load(f)

    # Merge data
    samples = []
    for tts in tts_results:
        if not tts.get("success"):
            continue

        idx = tts["idx"]
        if idx not in instructions_map:
            continue

        instr = instructions_map[idx]
        audio_path = tts["audio_path"]

        if not Path(audio_path).exists():
            continue

        samples.append({
            "audio": audio_path,
            "text": instr["text"],
            "question": instr["question"],
            "answer": instr["answer"],
            "task_type": instr["task_type"],
            "voice": tts.get("voice", "unknown"),
        })

    return samples


def generate_task_distribution_chart(samples):
    """Generate task distribution pie chart."""
    task_counts = Counter(s["task_type"] for s in samples)

    # Sort by count
    sorted_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in sorted_tasks]
    sizes = [t[1] for t in sorted_tasks]

    # Colors
    colors = plt.cm.tab20.colors[:len(labels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax1.set_title('Task Type Distribution', fontsize=14, fontweight='bold')

    # Bar chart
    bars = ax2.barh(labels[::-1], sizes[::-1], color=colors[::-1])
    ax2.set_xlabel('Number of Samples')
    ax2.set_title('Task Type Counts', fontsize=14, fontweight='bold')

    # Add count labels
    for bar, count in zip(bars, sizes[::-1]):
        ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                 str(count), va='center', fontsize=10)

    plt.tight_layout()

    chart_path = DATASET_DIR / "task_distribution.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Task distribution chart saved to {chart_path}")
    return chart_path


def generate_voice_distribution_chart(samples):
    """Generate voice distribution chart."""
    voice_counts = Counter(s["voice"] for s in samples)

    sorted_voices = sorted(voice_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [v[0] for v in sorted_voices]
    sizes = [v[1] for v in sorted_voices]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Pastel1.colors[:len(labels)]
    bars = ax.bar(labels, sizes, color=colors)

    ax.set_xlabel('Voice')
    ax.set_ylabel('Number of Samples')
    ax.set_title('TTS Voice Distribution', fontsize=14, fontweight='bold')

    # Add count labels
    for bar, count in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    chart_path = DATASET_DIR / "voice_distribution.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Voice distribution chart saved to {chart_path}")
    return chart_path


def create_readme(samples, task_chart_path, voice_chart_path):
    """Create README.md for the dataset."""

    task_counts = Counter(s["task_type"] for s in samples)
    voice_counts = Counter(s["voice"] for s in samples)

    # Task prompts from the generation script
    task_prompts = {
        "classification": [
            "Определи жанр текста из аудио",
            "Классифицируй стиль повествования",
            "Определи тип текста (диалог, описание, повествование)",
        ],
        "summarization": [
            "Кратко перескажи содержание аудио",
            "Сделай краткое резюме услышанного",
            "О чём говорится в этой записи?",
        ],
        "ner": [
            "Извлеки именованные сущности из аудио в формате JSON",
            "Найди все имена персонажей и места в аудио. Ответь JSON",
        ],
        "json_extraction": [
            'Извлеки информацию из аудио в JSON: {"тема": "", "персонажи": [], "место": "", "время": ""}',
            "Преобразуй содержание аудио в структурированный JSON с полями: summary, characters, mood",
        ],
        "json_structure": [
            'Представь содержание аудио как JSON-схему событий: [{"event": "", "actor": "", "result": ""}]',
            'Создай JSON-массив диалогов из аудио: [{"speaker": "", "text": ""}]',
        ],
        "json_analysis": [
            'Проанализируй аудио и верни JSON: {"sentiment": "", "confidence": 0.0, "reasons": []}',
            'Оцени текст из аудио, ответь JSON: {"quality": 1-10, "criteria": {"plot": 0, "style": 0}}',
        ],
        "translation": [
            "Переведи содержание аудио на английский язык",
            "Translate the audio content to English",
        ],
        "question_answering": [
            "Ответь на вопрос по содержанию аудио: кто главный герой?",
            "Что происходит в аудио? Ответь подробно",
        ],
        "sentiment": [
            "Определи эмоциональную окраску аудио",
            "Какое настроение передаёт этот фрагмент?",
        ],
        "keywords": [
            "Выдели ключевые слова из аудио",
            "Назови 5-7 главных понятий из аудио",
        ],
        "paraphrase": [
            "Перефразируй содержание аудио другими словами",
            "Передай смысл аудио, используя другие выражения",
        ],
        "continuation": [
            "Продолжи историю из аудио",
            "Что могло бы произойти дальше?",
        ],
    }

    # Build task table
    task_table = "| Task Type | Count | Percentage |\n|-----------|-------|------------|\n"
    total = len(samples)
    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100
        task_table += f"| {task} | {count} | {pct:.1f}% |\n"

    # Build prompts section
    prompts_section = ""
    for task, prompts in task_prompts.items():
        prompts_section += f"\n### {task}\n"
        for p in prompts:
            prompts_section += f"- `{p}`\n"

    readme = f"""---
license: cc-by-4.0
language:
- ru
tags:
- audio
- speech
- asr
- instruction-following
- tts
- russian
- fiction
size_categories:
- 1K<n<10K
task_categories:
- automatic-speech-recognition
- text-generation
- question-answering
---

# Ficbook Audio Instruct 10K

Synthetic audio instruction dataset for training Russian audio-language models.
Contains ~10K samples of fiction text voiced with OpenAI TTS and paired with diverse instruction tasks.

## Dataset Description

This dataset was created for training and evaluating audio-language models on Russian fiction content.
Each sample contains:
- **Audio**: Fiction text voiced using OpenAI's `gpt-4o-mini-tts` model
- **Text**: Original text from ficbook stories
- **Question**: Instruction/task for the model
- **Answer**: Expected response generated by Gemini 2.5 Flash
- **Task Type**: One of 12 task categories

## Data Collection Pipeline

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Ficbook Dataset    │────▶│  OpenAI TTS API  │────▶│   Audio Files       │
│  (preprocessed)     │     │  gpt-4o-mini-tts │     │   (MP3, 16kHz)      │
└─────────────────────┘     └──────────────────┘     └─────────────────────┘
         │                                                    │
         │                                                    │
         ▼                                                    ▼
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Text + Task Type   │────▶│  Gemini 2.5 Flash│────▶│  Question + Answer  │
│  (random selection) │     │  (via OpenRouter)│     │                     │
└─────────────────────┘     └──────────────────┘     └─────────────────────┘
         │                                                    │
         │                                                    │
         ▼                                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Final Dataset Sample                              │
│  {{audio, text, question, answer, task_type, voice}}                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Statistics

- **Total samples**: {len(samples):,}
- **Audio format**: MP3, mono, generated at 16kHz
- **TTS model**: OpenAI gpt-4o-mini-tts
- **TTS voices**: {len(voice_counts)} different voices
- **Task types**: {len(task_counts)} categories
- **Instruction model**: Gemini 2.5 Flash (via OpenRouter)

## Task Distribution

![Task Distribution](task_distribution.png)

{task_table}

## Voice Distribution

![Voice Distribution](voice_distribution.png)

**Voices used**: {', '.join(sorted(voice_counts.keys()))}

## Task Prompts

Examples of prompts used for each task type:
{prompts_section}

## Dataset Structure

```python
{{
    "audio": Audio(sampling_rate=16000),  # Audio file
    "text": str,                           # Original text
    "question": str,                       # Task instruction
    "answer": str,                         # Expected response
    "task_type": str,                      # Task category
    "voice": str,                          # TTS voice used
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("Vikhrmodels/Ficbook-Audio-Instruct-10K")

# Access a sample
sample = dataset["train"][0]
print(f"Task: {{sample['task_type']}}")
print(f"Question: {{sample['question']}}")
print(f"Answer: {{sample['answer']}}")

# Play audio (in notebook)
from IPython.display import Audio
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```

## Source Data

- **Text source**: [Vikhrmodels/ficbook_preprocessed](https://huggingface.co/datasets/Vikhrmodels/ficbook_preprocessed)
- First 10,000 samples from the preprocessed ficbook stories

## Generation Scripts

The dataset was generated using the following script:
- `generate_ficbook_dataset.py` - Main generation pipeline

## License

CC-BY-4.0

## Citation

```bibtex
@dataset{{ficbook_audio_instruct_10k,
  title={{Ficbook Audio Instruct 10K}},
  author={{VikhrModels}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/Vikhrmodels/Ficbook-Audio-Instruct-10K}}
}}
```
"""

    readme_path = DATASET_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"README saved to {readme_path}")
    return readme_path


def push_to_huggingface(samples):
    """Push dataset to HuggingFace."""
    print(f"Preparing dataset with {len(samples)} samples...")

    # Create HF dataset
    df = pd.DataFrame(samples)
    dataset = Dataset.from_pandas(df)

    # Cast audio column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Pushing to {HF_REPO}...")
    dataset.push_to_hub(
        HF_REPO,
        private=False,
    )

    # Upload additional files
    api = HfApi()

    # Upload charts
    for chart in ["task_distribution.png", "voice_distribution.png"]:
        chart_path = DATASET_DIR / chart
        if chart_path.exists():
            api.upload_file(
                path_or_fileobj=str(chart_path),
                path_in_repo=chart,
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            print(f"Uploaded {chart}")

    # Upload generation script
    script_path = Path("generate_ficbook_dataset.py")
    if script_path.exists():
        api.upload_file(
            path_or_fileobj=str(script_path),
            path_in_repo="scripts/generate_ficbook_dataset.py",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print("Uploaded generation script")

    print(f"\nDataset pushed to: https://huggingface.co/datasets/{HF_REPO}")


def main():
    print("Loading dataset...")
    samples = load_dataset_from_files()
    print(f"Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("No samples found! Make sure generation is complete.")
        return

    print("\nGenerating charts...")
    task_chart = generate_task_distribution_chart(samples)
    voice_chart = generate_voice_distribution_chart(samples)

    print("\nCreating README...")
    create_readme(samples, task_chart, voice_chart)

    print("\nPushing to HuggingFace...")
    push_to_huggingface(samples)

    print("\nDone!")


if __name__ == "__main__":
    main()
