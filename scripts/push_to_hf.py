"""
Push generated instruction dataset to HuggingFace Hub.
Format: audio + chatml conversations
"""

import json
import os
from datasets import Dataset, Audio, Features, Value, Sequence
from huggingface_hub import HfApi, create_repo
import pandas as pd
from tqdm import tqdm

# Config
HF_REPO = "Vikhrmodels/AudioBooksInstructGemini2.5"
CHECKPOINT_PATTERN = "instruct_dataset_checkpoint_*.json"
TONEBOOKS_PARQUET_PATTERN = "~/.cache/huggingface/hub/datasets--Vikhrmodels--ToneBooks/snapshots/*/data/train-*.parquet"

def load_latest_checkpoint():
    """Load the latest checkpoint file."""
    import glob
    import re

    files = glob.glob("instruct_dataset_checkpoint_*.json")
    if not files:
        raise FileNotFoundError("No checkpoint files found")

    # Sort by number in filename
    def get_num(f):
        match = re.search(r'checkpoint_(\d+)\.json', f)
        return int(match.group(1)) if match else 0

    files = sorted(files, key=get_num)
    latest = files[-1]
    print(f"Loading checkpoint: {latest}")
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)

def create_chatml_conversation(question, answer, system_prompt="Ты полезный голосовой ассистент."):
    """Create ChatML format conversation."""
    # Remove audio tags from question for conversation format
    clean_question = question.replace("<|start_of_audio|><|end_of_audio|}}", "").strip()
    clean_question = question.replace("<|start_of_audio|><|end_of_audio|>", "[AUDIO]").strip()

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": clean_question},
        {"role": "assistant", "content": answer}
    ]
    return conversation

def get_system_prompt(task_type):
    """Get appropriate system prompt for task type."""
    prompts = {
        "classification": "Ты эксперт по анализу аудио. Классифицируй аудио по заданному критерию.",
        "summarization": "Ты полезный голосовой ассистент. Кратко и точно передай суть услышанного.",
        "ner": "Ты NER-система для аудио. Извлекай сущности и отвечай строго в JSON формате.",
        "instruction_following": "Ты полезный голосовой ассистент. Внимательно слушай аудио и выполняй инструкции пользователя.",
    }
    return prompts.get(task_type, "Ты полезный голосовой ассистент.")

def main():
    print("Loading generated data...")
    data = load_latest_checkpoint()
    print(f"Loaded {len(data)} samples")

    # Load ToneBooks parquet to get audio
    print("\nLoading ToneBooks audio data...")
    import glob
    from huggingface_hub import hf_hub_download

    # Download parquet files
    parquet_files = []
    for i in range(6):
        path = hf_hub_download(
            repo_id="Vikhrmodels/ToneBooks",
            filename=f"data/train-0000{i}-of-00006.parquet",
            repo_type="dataset"
        )
        parquet_files.append(path)

    # Read parquets with audio column
    print("Reading parquet files with audio...")
    dfs = []
    for path in parquet_files[:2]:  # First 2 files have our indices
        df = pd.read_parquet(path)
        dfs.append(df)
    audio_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(audio_df)} audio samples")

    # Prepare dataset
    print("\nPreparing dataset...")
    processed_data = []

    for item in tqdm(data, desc="Processing"):
        audio_idx = item.get("audio_idx", 0)

        if audio_idx >= len(audio_df):
            continue

        # Get audio from original dataset
        audio_row = audio_df.iloc[audio_idx]

        # Create ChatML conversation
        system_prompt = get_system_prompt(item.get("task_type", "instruction_following"))
        conversation = create_chatml_conversation(
            item["question"],
            item.get("answer", ""),
            system_prompt
        )

        processed_data.append({
            "audio": audio_row["audio"],
            "text": item["text"],
            "question": item["question"],
            "answer": item.get("answer", ""),
            "task_type": item.get("task_type", "unknown"),
            "conversations": json.dumps(conversation, ensure_ascii=False),
            "voice_name": item.get("voice_name", ""),
        })

    print(f"\nProcessed {len(processed_data)} samples")

    # Create HF dataset
    print("\nCreating HuggingFace dataset...")

    # Extract audio bytes/paths
    hf_data = {
        "audio": [d["audio"] for d in processed_data],
        "text": [d["text"] for d in processed_data],
        "question": [d["question"] for d in processed_data],
        "answer": [d["answer"] for d in processed_data],
        "task_type": [d["task_type"] for d in processed_data],
        "conversations": [d["conversations"] for d in processed_data],
        "voice_name": [d["voice_name"] for d in processed_data],
    }

    dataset = Dataset.from_dict(hf_data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(dataset)}")
    task_counts = {}
    for item in processed_data:
        t = item["task_type"]
        task_counts[t] = task_counts.get(t, 0) + 1
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} ({100*count/len(processed_data):.1f}%)")

    # Create repo and push
    print(f"\nPushing to {HF_REPO}...")

    try:
        create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    dataset.push_to_hub(HF_REPO, private=False)
    print(f"Dataset pushed to: https://huggingface.co/datasets/{HF_REPO}")

    # Create README
    readme_content = """# AudioBooksInstructGemini2.5

Датасет инструкций для обучения аудио-языковых моделей, сгенерированный с помощью Gemini 2.5 Flash на основе [ToneBooks](https://huggingface.co/datasets/Vikhrmodels/ToneBooks).

## Описание

Датасет содержит аудиозаписи из русскоязычных аудиокниг с автоматически сгенерированными вопросами и ответами для обучения моделей следовать инструкциям.

## Типы задач

| Задача | Описание | Примерный % |
|--------|----------|-------------|
| **instruction_following** | Общие инструкции по работе с аудио (перевод, перефразирование, анализ) | ~25% |
| **summarization** | Краткий пересказ содержания аудио | ~25% |
| **classification** | Классификация эмоций, стиля, жанра | ~25% |
| **ner** | Извлечение именованных сущностей в JSON формате | ~25% |

## Формат данных

```python
{
    "audio": Audio(sampling_rate=16000),  # Аудиозапись
    "text": str,                          # Оригинальный текст аудио
    "question": str,                      # Вопрос/инструкция с тегом [AUDIO]
    "answer": str,                        # Сгенерированный ответ
    "task_type": str,                     # Тип задачи
    "conversations": str,                 # ChatML формат (JSON)
    "voice_name": str,                    # Имя диктора
}
```

## ChatML формат

Поле `conversations` содержит JSON с диалогом в формате ChatML:

```json
[
    {"role": "system", "content": "Ты полезный голосовой ассистент."},
    {"role": "user", "content": "Перескажи содержание аудио. [AUDIO]"},
    {"role": "assistant", "content": "В аудио говорится о..."}
]
```

## Использование

```python
from datasets import load_dataset

ds = load_dataset("Vikhrmodels/AudioBooksInstructGemini2.5")

# Пример
sample = ds["train"][0]
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
print(f"Task: {sample['task_type']}")

# Для ChatML
import json
conv = json.loads(sample['conversations'])
for msg in conv:
    print(f"{msg['role']}: {msg['content']}")
```

## Генерация

- **Базовый датасет**: [Vikhrmodels/ToneBooks](https://huggingface.co/datasets/Vikhrmodels/ToneBooks)
- **Модель генерации**: Google Gemini 2.5 Flash Lite (via OpenRouter)
- **Процесс**:
  1. Для каждого аудиофрагмента случайно выбирается тип задачи
  2. Gemini генерирует естественный вопрос на русском языке
  3. Gemini генерирует ответ, перефразируя исходный текст

## Лицензия

Следует лицензии базового датасета ToneBooks.

## Цитирование

```bibtex
@dataset{audiobooksinstructgemini2025,
    title={AudioBooksInstructGemini2.5},
    author={VikhrModels},
    year={2025},
    publisher={HuggingFace},
    url={https://huggingface.co/datasets/Vikhrmodels/AudioBooksInstructGemini2.5}
}
```
"""

    # Push README
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print("README uploaded!")

    print("\n=== DONE ===")
    print(f"Dataset: https://huggingface.co/datasets/{HF_REPO}")

if __name__ == "__main__":
    main()
