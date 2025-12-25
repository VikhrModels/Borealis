"""
Push Ficbook Audio Instruct dataset to HuggingFace.
"""

import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_from_disk, Audio
from huggingface_hub import HfApi

HF_REPO = "Vikhrmodels/Ficbook-Audio-Instruct-10K"
DATASET_DIR = Path("ficbook_instruct_10k")


def generate_charts():
    """Generate distribution charts."""
    # Load data
    with open("ficbook_instruct_10k.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    task_counts = Counter(s["task_type"] for s in data)
    voice_counts = Counter(s["voice"] for s in data)

    # Task distribution chart
    sorted_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in sorted_tasks]
    sizes = [t[1] for t in sorted_tasks]
    colors = plt.cm.tab20.colors[:len(labels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Task Type Distribution', fontsize=14, fontweight='bold')

    bars = ax2.barh(labels[::-1], sizes[::-1], color=colors[::-1])
    ax2.set_xlabel('Number of Samples')
    ax2.set_title('Task Type Counts', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, sizes[::-1]):
        ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, str(count), va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig("task_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved task_distribution.png")

    # Voice distribution chart
    sorted_voices = sorted(voice_counts.items(), key=lambda x: x[1], reverse=True)
    vlabels = [v[0] for v in sorted_voices]
    vsizes = [v[1] for v in sorted_voices]

    fig, ax = plt.subplots(figsize=(12, 6))
    vcolors = plt.cm.Pastel1.colors[:len(vlabels)]
    bars = ax.bar(vlabels, vsizes, color=vcolors)
    ax.set_xlabel('Voice')
    ax.set_ylabel('Number of Samples')
    ax.set_title('TTS Voice Distribution', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, vsizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(count), ha='center', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("voice_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved voice_distribution.png")

    return task_counts, voice_counts


def create_readme(task_counts, voice_counts, total):
    """Create README.md."""

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

    task_table = "| Task Type | Count | Percentage |\n|-----------|-------|------------|\n"
    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100
        task_table += f"| {task} | {count} | {pct:.1f}% |\n"

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

- **Total samples**: {total:,}
- **Audio format**: MP3, mono, generated at variable rates
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
    "audio": Audio(),           # Audio file path
    "text": str,                # Original text
    "question": str,            # Task instruction
    "answer": str,              # Expected response
    "task_type": str,           # Task category
    "voice": str,               # TTS voice used
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

    with open("README_dataset.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print("Saved README_dataset.md")


def main():
    print("Generating charts...")
    task_counts, voice_counts = generate_charts()

    total = sum(task_counts.values())
    print(f"\nTotal samples: {total}")

    print("\nCreating README...")
    create_readme(task_counts, voice_counts, total)

    print("\nLoading dataset...")
    dataset = load_from_disk(str(DATASET_DIR))

    # Cast audio column
    dataset = dataset.cast_column("audio", Audio())

    print(f"Dataset: {dataset}")
    print(f"\nPushing to {HF_REPO}...")

    dataset.push_to_hub(HF_REPO, private=False)

    # Upload additional files
    api = HfApi()

    for fname in ["task_distribution.png", "voice_distribution.png", "README_dataset.md", "generate_ficbook_dataset.py"]:
        if Path(fname).exists():
            dest = fname if fname != "README_dataset.md" else "README.md"
            api.upload_file(
                path_or_fileobj=fname,
                path_in_repo=dest,
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            print(f"Uploaded {fname} -> {dest}")

    print(f"\nDone! Dataset: https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
