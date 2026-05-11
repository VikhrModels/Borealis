---
title: Borealis Interactive Architecture Blog
emoji: 🐻‍❄️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Borealis — interactive architecture blog

Интерактивный блог, который проводит по архитектуре и ablation-исследованиям
[**Borealis**](https://github.com/VikhrModels/Borealis) — аудио-LLM для русского языка.

## Что внутри

- **Архитектура** — пайплайн `Whisper-large-v3 → downsample → Adapter → Qwen3`
  и встроенный [transformer-xray](https://huggingface.co/spaces/AlexWortega/transformer-xray)
  для визуализации графа модели без скачивания весов.
- **Adapter ablation** — интерактивный калькулятор параметров для
  Simple/Deep адаптера vs разные backbone'ы.
- **Размер LLM** — Qwen3-0.6B / 1.7B / 4B (конфиги `Borealis_1.5B/2.4B/5B`).
- **Train mode** — full fine-tune vs adapter-only.
- **Аугментации** — расписание `AugmentationScheduler`.
- **Результаты** — WER/CER на RuASRBenchmark и BigBenchAudio (Borealis vs Whisper vs Voxtral).
- **Конфиги** — сводная таблица всех ablation-конфигов из репо.

## Запуск локально

```bash
pip install -r requirements.txt
python app.py
```

## Развёртывание

Этот каталог — готовый HF Space. Загружается на Hugging Face одним пушем
в репозиторий Space с тем же содержимым (`app.py`, `requirements.txt`, `README.md`).

## Данные

Все цифры в блоге — из этого репозитория:

- `borealis/modeling.py` — формулы размеров адаптеров
- `borealis/augmentations.py` — таблица расписания аугментаций
- `configs/*.yaml` — гиперпараметры обучения
- `eval/eval_results_*.json` — реальные метрики на RuASRBenchmark и BigBenchAudio
