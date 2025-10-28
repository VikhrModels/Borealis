# Borealis

## Обзор
Borealis — аудио LLM для русского языка. Есть в двух вариантах - `0.6B` и `1.7B`. 

## Как запустить обучение модели
0. **Установка uv** 
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
1. **Установка зависимостей**  
   ```bash
   uv sync 
   ```

2. **Проброска ключей**  
   ```bash
   wandb login

   hf auth login
   ```

3. **Запуск обучения**  
   ```bash
   accelerate config

   accelerate launch train.py
   ```
   Сначала нужно задать конфиг под конкретную спецификацию сервера. Если обучение будет на одной видеокарте, то будет достаточно команды
   ```bash
   python train.py
   ```

4. **Инференс**  
   Для быстрой проверки используйте ноутбук [test_model.ipynb](test_model.ipynb). Он загружает сохранённую модель и вычисляет метрики $WER$ и $CER$

## Структура проекта
- [borealis/](borealis/)
  - [`__init__.py`](borealis/__init__.py) — экспорт основных компонентов
  - [`augmentations.py`](borealis/augmentations.py) — расписание аугментаций, миксинга шумов, реверберации и голосовых эффекторных слоёв
  - [`dataset.py`](borealis/dataset.py) — класс датасета
  - [`modeling.py`](borealis/modeling.py) — кастомный `BorealisForConditionalGeneration` и обвязка поверх `Qwen3ForCausalLM`
  - [`utils.py`](borealis/utils.py) — коллатор, нормализация текстов (`clean_dataset`), загрузка аудио
- [train.py](train.py) — главный файл для обучения
- [configs/](configs/) — шаблоны конфигов Hydra (добавятся позже)
