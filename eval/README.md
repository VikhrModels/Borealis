# Evaluation Scripts

Скрипты для оценки моделей Borealis и baseline-моделей на различных бенчмарках.

## Бенчмарки

### RuASRBenchmark
Русский ASR бенчмарк для оценки качества транскрипции. Метрики: WER (Word Error Rate), CER (Character Error Rate).

### BigBenchAudio
Мультизадачный аудио-бенчмарк с категориями:
- `formal_fallacies` — определение валидности логических аргументов
- `navigate` — определение финальной позиции по инструкциям навигации
- `object_counting` — подсчёт объектов по описанию
- `web_of_lies` — определение истинности утверждений

## Скрипты

### Borealis

| Скрипт | Описание |
|--------|----------|
| `eval_checkpoints.py` | Оценка всех instruct-чекпоинтов на RuASRBenchmark |
| `eval_single_checkpoint.py` | Оценка одного чекпоинта |
| `eval_all_checkpoints.py` | Пакетная оценка чекпоинтов |
| `eval_ablation_checkpoints.py` | Оценка ablation-экспериментов |
| `eval_adapter_checkpoint.py` | Оценка адаптерных чекпоинтов |
| `eval_adapter_batched.py` | Батчевая оценка адаптеров |
| `eval_adapter_checkpoint_param.py` | Оценка с параметром чекпоинта |
| `eval_bigbench_audio.py` | Оценка на BigBenchAudio |
| `eval_hf.py` | Оценка через HuggingFace AutoModel |

### Baseline модели

| Скрипт | Модель |
|--------|--------|
| `eval_whisper.py` | Whisper-large-v3 (OpenAI) |
| `eval_voxtral.py` | Voxtral-Mini-3B (Mistral) |
| `eval_voxtral_bigbench.py` | Voxtral на BigBenchAudio |

## Использование

### Оценка чекпоинта на RuASRBenchmark:
```bash
python eval/eval_single_checkpoint.py
```

### Оценка на BigBenchAudio:
```bash
python eval/eval_bigbench_audio.py <path_to_checkpoint>
```

### Оценка baseline:
```bash
python eval/eval_whisper.py
python eval/eval_voxtral.py
```

## Результаты

Результаты сохраняются в JSON-файлы:
- `eval_results_*.json` — метрики по сплитам
- `eval_*_log.txt` — логи выполнения

Формат результатов:
```json
{
  "checkpoint_name": {
    "split_name": {
      "wer": 12.34,
      "cer": 5.67,
      "samples": 1000
    }
  }
}
```

## Зависимости

- `torch`, `transformers` — модели
- `datasets` — загрузка бенчмарков
- `jiwer` — вычисление WER/CER
- `tqdm` — прогресс-бар
