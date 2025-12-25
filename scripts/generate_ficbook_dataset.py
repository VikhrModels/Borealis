"""
Generate instruction dataset from Ficbook with OpenAI TTS audio synthesis.
Extended tasks: classification, summarization, NER, JSON extraction, translation, Q&A, etc.
"""

import os
import json
import random
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from datasets import load_dataset, Dataset, Audio
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import time
from openai import AsyncOpenAI

# API Keys (from environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

# OpenAI TTS voices
TTS_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]

# Voice instructions for different styles
VOICE_INSTRUCTIONS = [
    "Speak in a calm, neutral tone.",
    "Speak clearly and expressively.",
    "Read with emotion appropriate to the content.",
    "Speak naturally as if telling a story.",
    "Use a warm and engaging voice.",
    "Read at a moderate pace with good articulation.",
    "Speak with a gentle, soothing tone.",
    "Read expressively with varied intonation.",
]

# Output directory for audio files
AUDIO_OUTPUT_DIR = Path("ficbook_audio")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

# Extended task types with more JSON tasks
TASK_TYPES = [
    "classification",
    "summarization",
    "ner",
    "json_extraction",
    "json_structure",
    "json_analysis",
    "translation",
    "question_answering",
    "sentiment",
    "keywords",
    "paraphrase",
    "continuation",
]

TASK_TEMPLATES = {
    "classification": {
        "descriptions": [
            "Определи жанр текста из аудио",
            "Классифицируй стиль повествования",
            "Определи тип текста (диалог, описание, повествование)",
            "Какой литературный жанр представлен в аудио?",
            "Определи характер текста: художественный или нехудожественный",
        ],
        "system": "Ты эксперт по анализу текстов. Классифицируй аудио по заданному критерию.",
    },
    "summarization": {
        "descriptions": [
            "Кратко перескажи содержание аудио",
            "Сделай краткое резюме услышанного",
            "О чём говорится в этой записи?",
            "Передай основную мысль аудио в 2-3 предложениях",
            "Суммаризуй сюжет аудиофрагмента",
        ],
        "system": "Ты голосовой ассистент. Кратко и точно передай суть услышанного.",
    },
    "ner": {
        "descriptions": [
            "Извлеки именованные сущности из аудио в формате JSON",
            "Найди все имена персонажей и места в аудио. Ответь JSON",
            "Выдели ключевые сущности из аудио в JSON формате",
            "Определи всех персонажей и локации. Формат ответа: JSON",
            "Извлеки структурированную информацию о персонажах из аудио в JSON",
        ],
        "system": "Ты NER-система. Извлекай сущности и отвечай строго в JSON формате.",
    },
    "json_extraction": {
        "descriptions": [
            "Извлеки информацию из аудио в JSON: {\"тема\": \"\", \"персонажи\": [], \"место\": \"\", \"время\": \"\"}",
            "Преобразуй содержание аудио в структурированный JSON с полями: summary, characters, mood",
            "Создай JSON-объект с ключевой информацией из аудио",
            "Извлеки метаданные из аудио в формате JSON: {\"genre\": \"\", \"tone\": \"\", \"keywords\": []}",
            "Сформируй JSON с анализом аудио: события, персонажи, эмоции",
        ],
        "system": "Ты система извлечения данных. Преобразуй аудио в структурированный JSON.",
    },
    "json_structure": {
        "descriptions": [
            "Представь содержание аудио как JSON-схему событий: [{\"event\": \"\", \"actor\": \"\", \"result\": \"\"}]",
            "Создай JSON-массив диалогов из аудио: [{\"speaker\": \"\", \"text\": \"\"}]",
            "Структурируй информацию из аудио в JSON: {\"context\": {}, \"content\": {}, \"conclusion\": {}}",
            "Преврати аудио в JSON-граф связей персонажей",
            "Сгенерируй JSON-timeline событий из аудио",
        ],
        "system": "Ты конвертер текста в JSON. Создавай сложные структурированные JSON-объекты.",
    },
    "json_analysis": {
        "descriptions": [
            "Проанализируй аудио и верни JSON: {\"sentiment\": \"\", \"confidence\": 0.0, \"reasons\": []}",
            "Оцени текст из аудио, ответь JSON: {\"quality\": 1-10, \"criteria\": {\"plot\": 0, \"style\": 0, \"characters\": 0}}",
            "Сделай лингвистический анализ в JSON: {\"pos_tags\": {}, \"syntax\": \"\", \"complexity\": \"\"}",
            "Верни JSON с оценкой читабельности аудио: {\"level\": \"\", \"audience\": \"\", \"score\": 0}",
            "Анализ эмоций в JSON: {\"emotions\": [{\"emotion\": \"\", \"intensity\": 0.0, \"trigger\": \"\"}]}",
        ],
        "system": "Ты аналитическая система. Проводи анализ и возвращай результаты в JSON.",
    },
    "translation": {
        "descriptions": [
            "Переведи содержание аудио на английский язык",
            "Translate the audio content to English",
            "Сделай перевод текста из аудио на английский",
            "Переведи услышанное на English, сохраняя стиль",
            "Дай английский перевод содержимого аудио",
        ],
        "system": "Ты профессиональный переводчик. Переводи точно, сохраняя стиль оригинала.",
    },
    "question_answering": {
        "descriptions": [
            "Ответь на вопрос по содержанию аудио: кто главный герой?",
            "Что происходит в аудио? Ответь подробно",
            "Какие действия совершают персонажи в аудио?",
            "Где и когда происходит действие в аудио?",
            "Какова мотивация персонажей в аудио?",
            "Что чувствуют герои в этом фрагменте?",
        ],
        "system": "Ты внимательный слушатель. Отвечай на вопросы по содержанию аудио.",
    },
    "sentiment": {
        "descriptions": [
            "Определи эмоциональную окраску аудио",
            "Какое настроение передаёт этот фрагмент?",
            "Оцени тональность текста: позитивная, негативная или нейтральная",
            "Какие эмоции вызывает это аудио?",
            "Определи эмоциональный фон повествования",
        ],
        "system": "Ты эксперт по анализу эмоций. Определяй настроение и тональность текста.",
    },
    "keywords": {
        "descriptions": [
            "Выдели ключевые слова из аудио",
            "Назови 5-7 главных понятий из аудио",
            "Какие слова лучше всего описывают содержание аудио?",
            "Извлеки теги для этого аудиофрагмента",
            "Определи ключевые темы аудио",
        ],
        "system": "Ты система извлечения ключевых слов. Выделяй главные понятия из текста.",
    },
    "paraphrase": {
        "descriptions": [
            "Перефразируй содержание аудио другими словами",
            "Передай смысл аудио, используя другие выражения",
            "Перескажи услышанное своими словами",
            "Изложи содержание аудио в другом стиле",
            "Переформулируй текст из аудио",
        ],
        "system": "Ты эксперт по перефразированию. Передавай смысл другими словами.",
    },
    "continuation": {
        "descriptions": [
            "Продолжи историю из аудио",
            "Что могло бы произойти дальше?",
            "Напиши продолжение этого фрагмента",
            "Как может развиться сюжет дальше?",
            "Придумай, что случится после событий в аудио",
        ],
        "system": "Ты писатель. Продолжай истории в том же стиле.",
    },
}


async def generate_tts_audio(client: AsyncOpenAI, text: str, idx: int, semaphore: asyncio.Semaphore):
    """Generate TTS audio using OpenAI API."""
    async with semaphore:
        try:
            voice = random.choice(TTS_VOICES)
            instruction = random.choice(VOICE_INSTRUCTIONS)

            # Truncate text if too long (max ~4096 chars for TTS)
            if len(text) > 4000:
                text = text[:4000] + "..."

            audio_path = AUDIO_OUTPUT_DIR / f"audio_{idx:06d}.mp3"

            # Skip if already exists
            if audio_path.exists():
                return {
                    "idx": idx,
                    "audio_path": str(audio_path),
                    "voice": voice,
                    "success": True,
                }

            async with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                instructions=instruction,
            ) as response:
                async with aiofiles.open(audio_path, "wb") as f:
                    async for chunk in response.iter_bytes():
                        await f.write(chunk)

            return {
                "idx": idx,
                "audio_path": str(audio_path),
                "voice": voice,
                "success": True,
            }

        except Exception as e:
            print(f"TTS error for idx {idx}: {e}")
            return {
                "idx": idx,
                "audio_path": None,
                "voice": None,
                "success": False,
                "error": str(e),
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
            "max_tokens": 1024,
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
                print(f"Gemini error: {e}")
                await asyncio.sleep(1)

        return None


async def generate_question(session, text, task_type, semaphore):
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

Ответь ТОЛЬКО вопросом, без пояснений."""

    system = "Ты генератор вопросов. Создавай общие вопросы БЕЗ цитирования содержимого аудио."

    result = await call_gemini(session, prompt, system, semaphore)
    if result:
        result = result.strip().strip('"').strip("'")
        import re
        result = re.sub(r'["\'].*?["\']', '', result).strip()
        if "<|start_of_audio|>" not in result:
            result = result + " <|start_of_audio|><|end_of_audio|>"
    return result


async def generate_answer(session, text, question, task_type, semaphore):
    """Generate an answer based on the audio text."""

    if task_type in ["ner", "json_extraction", "json_structure", "json_analysis"]:
        prompt = f"""На основе аудио с текстом:
"{text}"

И вопроса: "{question}"

Ответь СТРОГО в JSON формате. Создай валидный JSON с релевантной информацией.
Если данных нет - используй пустые значения. Перефразируй найденную информацию."""

    elif task_type == "translation":
        prompt = f"""На основе аудио с текстом:
"{text}"

И вопроса: "{question}"

Переведи содержание на английский язык. Сохрани стиль и настроение оригинала."""

    elif task_type in ["classification", "sentiment"]:
        prompt = f"""На основе аудио с текстом:
"{text}"

И вопроса: "{question}"

Дай краткий ответ-классификацию (1-2 предложения)."""

    elif task_type == "continuation":
        prompt = f"""На основе аудио с текстом:
"{text}"

И вопроса: "{question}"

Продолжи историю в том же стиле (3-5 предложений). Сохрани атмосферу и характеры персонажей."""

    elif task_type == "keywords":
        prompt = f"""На основе аудио с текстом:
"{text}"

И вопроса: "{question}"

Выдели 5-7 ключевых слов или фраз. Перечисли их через запятую."""

    else:  # summarization, paraphrase, question_answering
        prompt = f"""На основе аудио с текстом:
"{text}"

И вопроса: "{question}"

Дай информативный ответ (2-5 предложений).
ВАЖНО: Перефразируй информацию своими словами, НЕ копируй текст напрямую."""

    system = TASK_TEMPLATES[task_type]["system"]
    return await call_gemini(session, prompt, system, semaphore)


async def process_sample(session, sample, semaphore):
    """Process a single sample: generate question and answer."""
    text = sample["text"]
    idx = sample["idx"]

    if len(text) < 20:
        return None

    task_type = random.choice(TASK_TYPES)

    question = await generate_question(session, text, task_type, semaphore)
    if not question:
        return None

    answer = await generate_answer(session, text, question, task_type, semaphore)
    if not answer:
        return None

    return {
        "idx": idx,
        "text": text,
        "question": question,
        "answer": answer,
        "task_type": task_type,
    }


async def generate_tts_batch(texts_with_idx, max_concurrent=20):
    """Generate TTS for a batch of texts."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        generate_tts_audio(client, item["text"], item["idx"], semaphore)
        for item in texts_with_idx
    ]

    results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating TTS"):
        result = await coro
        results.append(result)

    return results


async def generate_instructions_batch(samples, max_concurrent=100):
    """Generate instructions for a batch of samples."""
    semaphore = asyncio.Semaphore(max_concurrent)

    connector = aiohttp.TCPConnector(limit=150)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_sample(session, s, semaphore) for s in samples]
        results = []

        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating instructions"):
            result = await coro
            if result:
                results.append(result)

    return results


def main():
    print("=" * 60)
    print("Ficbook TTS + Instruction Dataset Generator")
    print("=" * 60)

    # Load dataset
    print("\nLoading ficbook_preprocessed dataset...")
    dataset = load_dataset("Vikhrmodels/ficbook_preprocessed", split="train")

    # Take first 10k samples
    num_samples = 10000
    print(f"Selecting first {num_samples} samples...")

    samples = []
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Preparing samples"):
        # Ficbook dataset has 'conversation' array where second item is the generated text
        conv = dataset[i].get("conversation", [])
        if len(conv) >= 2:
            text = conv[1].get("content", "")
        else:
            continue
        if text and len(text) > 20:
            samples.append({
                "idx": i,
                "text": text,
            })

    print(f"Prepared {len(samples)} valid samples")

    # Check for existing checkpoints
    checkpoint_file = "ficbook_dataset_checkpoint.json"
    tts_checkpoint_file = "ficbook_tts_checkpoint.json"

    tts_results = []
    instruction_results = []

    if os.path.exists(tts_checkpoint_file):
        print(f"\nLoading TTS checkpoint: {tts_checkpoint_file}")
        with open(tts_checkpoint_file, "r", encoding="utf-8") as f:
            tts_results = json.load(f)
        print(f"Loaded {len(tts_results)} TTS results")

    if os.path.exists(checkpoint_file):
        print(f"\nLoading instruction checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            instruction_results = json.load(f)
        print(f"Loaded {len(instruction_results)} instruction results")

    # Step 1: Generate TTS audio
    completed_tts_idx = {r["idx"] for r in tts_results if r.get("success")}
    samples_needing_tts = [s for s in samples if s["idx"] not in completed_tts_idx]

    if samples_needing_tts:
        print(f"\n=== Step 1: Generating TTS for {len(samples_needing_tts)} samples ===")

        # Process in batches
        batch_size = 500
        for i in range(0, len(samples_needing_tts), batch_size):
            batch = samples_needing_tts[i:i+batch_size]
            print(f"\nTTS Batch {i//batch_size + 1}/{(len(samples_needing_tts)-1)//batch_size + 1}")

            batch_results = asyncio.run(generate_tts_batch(batch, max_concurrent=20))
            tts_results.extend(batch_results)

            # Save checkpoint
            with open(tts_checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(tts_results, f, ensure_ascii=False, indent=2)
            print(f"TTS checkpoint saved: {len(tts_results)} results")
    else:
        print("\n=== Step 1: TTS already complete ===")

    # Step 2: Generate instructions
    completed_inst_idx = {r["idx"] for r in instruction_results}
    samples_needing_inst = [s for s in samples if s["idx"] not in completed_inst_idx]

    if samples_needing_inst:
        print(f"\n=== Step 2: Generating instructions for {len(samples_needing_inst)} samples ===")

        batch_size = 1000
        for i in range(0, len(samples_needing_inst), batch_size):
            batch = samples_needing_inst[i:i+batch_size]
            print(f"\nInstruction Batch {i//batch_size + 1}/{(len(samples_needing_inst)-1)//batch_size + 1}")

            batch_results = asyncio.run(generate_instructions_batch(batch, max_concurrent=100))
            instruction_results.extend(batch_results)

            # Save checkpoint
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(instruction_results, f, ensure_ascii=False, indent=2)
            print(f"Instruction checkpoint saved: {len(instruction_results)} results")
    else:
        print("\n=== Step 2: Instructions already complete ===")

    # Step 3: Merge results and create final dataset
    print("\n=== Step 3: Creating final dataset ===")

    # Create lookup for TTS results
    tts_lookup = {r["idx"]: r for r in tts_results if r.get("success")}

    # Merge
    final_data = []
    for inst in instruction_results:
        idx = inst["idx"]
        if idx in tts_lookup:
            tts = tts_lookup[idx]
            final_data.append({
                "idx": idx,
                "text": inst["text"],
                "audio_path": tts["audio_path"],
                "voice": tts["voice"],
                "question": inst["question"],
                "answer": inst["answer"],
                "task_type": inst["task_type"],
            })

    print(f"Final dataset size: {len(final_data)}")

    # Create HuggingFace dataset
    hf_data = {
        "idx": [d["idx"] for d in final_data],
        "text": [d["text"] for d in final_data],
        "audio_path": [d["audio_path"] for d in final_data],
        "voice": [d["voice"] for d in final_data],
        "question": [d["question"] for d in final_data],
        "answer": [d["answer"] for d in final_data],
        "task_type": [d["task_type"] for d in final_data],
    }

    final_dataset = Dataset.from_dict(hf_data)

    # Save
    output_path = "ficbook_instruct_10k"
    final_dataset.save_to_disk(output_path)
    print(f"Dataset saved to: {output_path}")

    with open(f"{output_path}.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {output_path}.json")

    final_dataset.to_parquet(f"{output_path}.parquet")
    print(f"Parquet saved: {output_path}.parquet")

    # Print statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(final_data)}")
    print(f"\nTask distribution:")
    task_counts = {}
    for d in final_data:
        task_counts[d["task_type"]] = task_counts.get(d["task_type"], 0) + 1
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count} ({100*count/len(final_data):.1f}%)")

    print(f"\nVoice distribution:")
    voice_counts = {}
    for d in final_data:
        voice_counts[d["voice"]] = voice_counts.get(d["voice"], 0) + 1
    for voice, count in sorted(voice_counts.items(), key=lambda x: -x[1]):
        print(f"  {voice}: {count}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"Audio files: {AUDIO_OUTPUT_DIR}")
    print(f"Dataset: {output_path}")


if __name__ == "__main__":
    main()
