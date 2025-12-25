"""
Generate instruction dataset from ToneBooks using Gemini Flash 2.5 via OpenRouter.
Tasks: classification, summarization, NER (JSON entities)
"""

import os
import json
import random
import asyncio
import aiohttp
from datasets import load_dataset, Dataset, Audio
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import time

# OpenRouter API config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

# Task types
TASK_TYPES = ["classification", "summarization", "ner", "instruction_following"]

# Task descriptions and question templates
TASK_TEMPLATES = {
    "classification": {
        "descriptions": [
            "Определи эмоциональную окраску аудио",
            "Классифицируй настроение говорящего",
            "Определи тип речи в аудио",
            "Какой жанр у этого аудиофрагмента?",
            "Определи стиль повествования",
        ],
        "system": "Ты эксперт по анализу аудио. Классифицируй аудио по заданному критерию.",
    },
    "summarization": {
        "descriptions": [
            "Кратко перескажи содержание аудио",
            "Сделай резюме услышанного",
            "О чём говорится в этой записи?",
            "Передай основную мысль аудио",
            "Суммаризуй содержание аудиофрагмента",
        ],
        "system": "Ты полезный голосовой ассистент. Кратко и точно передай суть услышанного.",
    },
    "ner": {
        "descriptions": [
            "Извлеки именованные сущности из аудио в формате JSON",
            "Найди все имена, места и организации в аудио. Ответь JSON",
            "Выдели ключевые сущности из аудио в JSON формате",
            "Определи персонажей и локации в аудио. Формат: JSON",
            "Извлеки структурированную информацию из аудио в JSON",
        ],
        "system": "Ты NER-система для аудио. Извлекай сущности и отвечай строго в JSON формате.",
    },
    "instruction_following": {
        "descriptions": [
            "Выполни задание на основе аудио",
            "Ответь на вопрос по содержанию аудио",
            "Проанализируй аудио и выполни инструкцию",
            "Сделай то, что просят, используя информацию из аудио",
            "Следуй инструкции и используй данные из аудиозаписи",
            "Переведи текст из аудио на английский",
            "Перефразируй услышанное другими словами",
            "Объясни смысл сказанного в аудио",
            "Продолжи мысль из аудио",
            "Задай вопрос по содержанию аудио",
            "Найди главную идею в аудио",
            "Опиши ситуацию из аудио",
            "Дай совет на основе услышанного",
            "Прокомментируй содержание аудио",
        ],
        "system": "Ты полезный голосовой ассистент. Внимательно слушай аудио и выполняй инструкции пользователя.",
    },
}


async def call_gemini(session, prompt, system_prompt, semaphore, max_retries=3):
    """Call Gemini via OpenRouter with rate limiting and retries."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Borealis",
        }

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.7,
        }

        for attempt in range(max_retries):
            try:
                async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
                    if resp.status == 429:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue

                    if resp.status != 200:
                        text = await resp.text()
                        print(f"API error {resp.status}: {text[:200]}")
                        await asyncio.sleep(1)
                        continue

                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(1)

        return None


async def generate_question(session, text, text_description, task_type, semaphore):
    """Generate a question for the given task type."""
    template = random.choice(TASK_TEMPLATES[task_type]["descriptions"])

    prompt = f"""Сгенерируй вопрос/задание для голосового ассистента.

Тип задачи: {task_type}
Шаблон задания: "{template}"

Сгенерируй ОДИН конкретный вопрос или задание на русском языке.

ВАЖНО:
- НЕ включай текст или содержание аудио в вопрос!
- Вопрос должен быть ОБЩИМ, без цитирования конкретного текста
- Модель должна сама извлечь информацию из аудио

Примеры ПРАВИЛЬНЫХ вопросов:
- "Извлеки именованные сущности из аудио в формате JSON"
- "Кратко перескажи что говорится в аудио"
- "Определи эмоциональную окраску речи"
- "О чём идёт речь в этой записи?"

Примеры НЕПРАВИЛЬНЫХ вопросов (НЕ делай так!):
- "Извлеки из текста 'Маша пошла в лес' именованные сущности" - НЕТ, не цитируй текст!
- "Перескажи историю про Ивана" - НЕТ, не упоминай конкретные имена из аудио!

Ответь ТОЛЬКО вопросом, без пояснений."""

    system = "Ты генератор вопросов для обучения голосового ассистента. Создавай общие вопросы БЕЗ цитирования содержимого аудио."

    result = await call_gemini(session, prompt, system, semaphore)
    if result:
        # Clean up the question
        result = result.strip().strip('"').strip("'")
        # Remove any quoted text that might have slipped through
        import re
        result = re.sub(r'["\'].*?["\']', '', result).strip()
        # Add audio placeholder
        if "<|start_of_audio|>" not in result:
            result = result + " <|start_of_audio|><|end_of_audio|>"
    return result


async def generate_answer(session, text, text_description, question, task_type, semaphore):
    """Generate an answer based on the audio text."""

    if task_type == "ner":
        prompt = f"""На основе аудио с текстом:
"{text}"

И вопроса: "{question}"

Извлеки именованные сущности и ответь СТРОГО в JSON формате:
{{"persons": [...], "locations": [...], "organizations": [...], "other_entities": [...]}}

Если сущностей нет - используй пустые списки. Перефразируй найденные сущности своими словами."""

    elif task_type == "classification":
        prompt = f"""На основе аудио с текстом:
"{text}"
Характеристики голоса: {text_description}

И вопроса: "{question}"

Дай краткий ответ-классификацию (1-2 предложения). Используй информацию из характеристик голоса.
Перефразируй своими словами, не копируй текст напрямую."""

    elif task_type == "instruction_following":
        prompt = f"""На основе аудио с текстом:
"{text}"
Характеристики голоса: {text_description}

Пользователь дал инструкцию: "{question}"

Выполни инструкцию пользователя, используя информацию из аудио.
Дай полезный, информативный ответ (2-5 предложений).
ВАЖНО: Перефразируй информацию своими словами, НЕ копируй текст напрямую.
Отвечай на русском языке (если не просят перевести)."""

    else:  # summarization
        prompt = f"""На основе аудио с текстом:
"{text}"

И вопроса: "{question}"

Дай краткий пересказ/резюме (2-4 предложения).
ВАЖНО: Перефразируй текст своими словами, НЕ копируй оригинал.
Сохрани смысл но измени формулировки."""

    system = TASK_TEMPLATES[task_type]["system"]
    return await call_gemini(session, prompt, system, semaphore)


async def process_sample(session, sample, semaphore):
    """Process a single sample: generate question and answer."""
    text = sample["text"]
    text_description = sample.get("text_description", "")

    # Skip very short texts
    if len(text) < 10:
        return None

    # Random task type
    task_type = random.choice(TASK_TYPES)

    # Generate question
    question = await generate_question(session, text, text_description, task_type, semaphore)
    if not question:
        return None

    # Generate answer
    answer = await generate_answer(session, text, text_description, question, task_type, semaphore)
    if not answer:
        return None

    return {
        "audio_idx": sample.get("audio_idx"),
        "text": text,
        "question": question,
        "answer": answer,
        "task_type": task_type,
        "voice_name": sample.get("voice_name", ""),
    }


async def process_batch(samples, batch_size=50):
    """Process samples in batches with concurrency control."""
    semaphore = asyncio.Semaphore(100)  # Max 100 concurrent requests

    connector = aiohttp.TCPConnector(limit=150)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_sample(session, s, semaphore) for s in samples]
        results = []

        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing"):
            result = await coro
            if result:
                results.append(result)

    return results


def main():
    print("Loading ToneBooks dataset (parquet via pandas)...")
    import pandas as pd
    import glob as glob_module
    from huggingface_hub import hf_hub_download

    # Check for existing checkpoint to resume
    checkpoint_files = sorted(glob_module.glob("instruct_dataset_checkpoint_*.json"))
    start_from = 0
    all_results = []
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print(f"Found checkpoint: {latest_checkpoint}")
        with open(latest_checkpoint, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        # Extract starting chunk from number of results
        start_from = (len(all_results) // 1000) * 1000
        print(f"Resuming from sample {start_from}, loaded {len(all_results)} results")

    # Download and read parquet directly with pandas (no audio decoding)
    print("Downloading parquet files...")
    parquet_files = []
    for i in range(6):  # 6 train files
        path = hf_hub_download(
            repo_id="Vikhrmodels/ToneBooks",
            filename=f"data/train-0000{i}-of-00006.parquet",
            repo_type="dataset"
        )
        parquet_files.append(path)
        print(f"  Downloaded: train-0000{i}-of-00006.parquet")

    print("Reading parquet files...")
    # Read all parquet files for 90k dataset
    dfs = []
    for path in parquet_files:  # All 6 files
        df = pd.read_parquet(path, columns=["text", "text_description", "voice_name"])
        dfs.append(df)
        print(f"  Loaded {len(df)} rows")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # Take 90k samples
    print("Collecting 90,000 samples (text only)...")
    samples = []
    for i in tqdm(range(min(90000, len(df))), desc="Loading"):
        row = df.iloc[i]
        samples.append({
            "audio_idx": i,  # Store index to map back to audio later
            "text": str(row["text"]) if row["text"] else "",
            "text_description": str(row["text_description"]) if row["text_description"] else "",
            "voice_name": str(row["voice_name"]) if row["voice_name"] else "",
        })

    print(f"Collected {len(samples)} samples")

    # Process in chunks to save progress
    chunk_size = 1000
    # all_results already loaded from checkpoint above

    for i in range(start_from, len(samples), chunk_size):
        chunk = samples[i:i+chunk_size]
        print(f"\nProcessing chunk {i//chunk_size + 1}/{len(samples)//chunk_size + 1}")

        results = asyncio.run(process_batch(chunk))
        all_results.extend(results)

        # Save intermediate results
        print(f"Total processed: {len(all_results)}")

        # Save checkpoint
        if len(all_results) > 0:
            checkpoint_path = f"instruct_dataset_checkpoint_{len(all_results)}.json"
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Create final dataset
    print("\nCreating HuggingFace dataset...")

    # Prepare data for HF format
    hf_data = {
        "audio_idx": [r["audio_idx"] for r in all_results],
        "text": [r["text"] for r in all_results],
        "question": [r["question"] for r in all_results],
        "answer": [r["answer"] for r in all_results],
        "task_type": [r["task_type"] for r in all_results],
    }

    final_dataset = Dataset.from_dict(hf_data)

    # Save locally
    output_path = "tonebooks_instruct_90k"
    final_dataset.save_to_disk(output_path)
    print(f"Dataset saved to: {output_path}")

    # Also save as JSON for easier inspection
    with open(f"{output_path}.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {output_path}.json")

    # Save parquet with audio references
    final_dataset.to_parquet(f"{output_path}.parquet")
    print(f"Parquet saved: {output_path}.parquet")
    print("\nNote: audio_idx maps to row index in original ToneBooks dataset")

    print(f"\nDone! Generated {len(all_results)} instruction samples")
    print(f"Task distribution:")
    for task in TASK_TYPES:
        count = sum(1 for r in all_results if r["task_type"] == task)
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
