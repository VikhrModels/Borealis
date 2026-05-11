"""Borealis: Interactive Architecture Blog (HF Space).

A Gradio app that walks through the architecture and ablations of Borealis,
a Russian audio LLM built on Whisper-large-v3 + an adapter + Qwen3.
Architecture exploration is delegated to AlexWortega/transformer-xray via iframe.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

XRAY_URL = "https://alexwortega-transformer-xray.hf.space/"
REPO_URL = "https://github.com/VikhrModels/Borealis"

# Whisper-large-v3 encoder hidden dim
WHISPER_D = 1280
DOWNSAMPLE = 4

# Qwen3 hidden sizes (sourced from HF model configs)
QWEN3 = {
    "Qwen3-0.6B": {"hidden": 1024, "layers": 28, "heads": 16, "kv_heads": 8, "params_b": 0.6},
    "Qwen3-1.7B": {"hidden": 2048, "layers": 28, "heads": 16, "kv_heads": 8, "params_b": 1.7},
    "Qwen3-4B":   {"hidden": 2560, "layers": 36, "heads": 32, "kv_heads": 8, "params_b": 4.0},
}


def adapter_params_simple(whisper_d: int, downsample: int, llm_hidden: int) -> int:
    hidden_in = whisper_d * downsample
    return hidden_in * llm_hidden + llm_hidden * llm_hidden


def adapter_params_deep(
    whisper_d: int,
    downsample: int,
    llm_hidden: int,
    num_layers: int,
    expansion: float,
) -> int:
    hidden_in = whisper_d * downsample
    hd = int(llm_hidden * expansion)
    input_proj = hidden_in * hd + 2 * hd
    per_layer = 2 * hd + hd * (2 * hd) + (2 * hd) * hd
    layers = num_layers * per_layer
    output_proj = 2 * hd + hd * llm_hidden
    return input_proj + layers + output_proj


def adapter_calc(adapter_type: str, llm_name: str, num_layers: int, expansion: float):
    llm_hidden = QWEN3[llm_name]["hidden"]
    if adapter_type == "Simple":
        params = adapter_params_simple(WHISPER_D, DOWNSAMPLE, llm_hidden)
        shape = (
            f"Linear({WHISPER_D * DOWNSAMPLE} → {llm_hidden}) "
            f"→ GELU → Linear({llm_hidden} → {llm_hidden})"
        )
    else:
        params = adapter_params_deep(WHISPER_D, DOWNSAMPLE, llm_hidden, num_layers, expansion)
        hd = int(llm_hidden * expansion)
        shape = (
            f"InputProj({WHISPER_D * DOWNSAMPLE} → {hd}) "
            f"→ {num_layers} × [LN + MLP({hd} → {2 * hd} → {hd}) + residual] "
            f"→ OutputProj({hd} → {llm_hidden})"
        )
    total_llm = QWEN3[llm_name]["params_b"] * 1e9
    pct = params / (total_llm + params) * 100
    return (
        f"### Адаптер: **{params:,}** параметров ({params/1e6:.1f}M)\n\n"
        f"- Доля от полной модели: **{pct:.2f}%**\n"
        f"- Размер слоёв: {shape}\n"
        f"- Whisper-large-v3 d_model: {WHISPER_D}, downsample: ×{DOWNSAMPLE} → вход {WHISPER_D * DOWNSAMPLE}\n"
        f"- Hidden Qwen3 ({llm_name}): {llm_hidden}\n"
    )


def adapter_compare_plot():
    rows = []
    for llm in QWEN3:
        h = QWEN3[llm]["hidden"]
        simple = adapter_params_simple(WHISPER_D, DOWNSAMPLE, h)
        deep_15 = adapter_params_deep(WHISPER_D, DOWNSAMPLE, h, 3, 1.5)
        deep_20 = adapter_params_deep(WHISPER_D, DOWNSAMPLE, h, 4, 2.0)
        rows.append({"LLM": llm, "Variant": "Simple (×1.0, 0L)", "Params (M)": simple / 1e6})
        rows.append({"LLM": llm, "Variant": "Deep (×1.5, 3L)", "Params (M)": deep_15 / 1e6})
        rows.append({"LLM": llm, "Variant": "Deep (×2.0, 4L)", "Params (M)": deep_20 / 1e6})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="LLM", y="Params (M)", color="Variant",
        barmode="group",
        title="Размер адаптера vs LLM backbone × тип адаптера",
        text_auto=".1f",
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return fig


# --- Eval results (from eval/eval_results_*.json) ---------------------------
HF_EVAL = {
    "Borealis-HF baseline": {
        "Russian_LibriSpeech": {"wer": 6.63, "cer": 3.49},
        "Common_Voice_22":     {"wer": 8.88, "cer": 5.04},
        "Tone_Webinars":       {"wer": 56.87, "cer": 52.47},
        "Tone_Books":          {"wer": 6.03, "cer": 3.75},
        "Tone_Speak":          {"wer": 4.63, "cer": 3.38},
        "Sova_RuDevices":      {"wer": 17.28, "cer": 8.03},
    },
    "Borealis step-2898":   {
        "Russian_LibriSpeech": {"wer": 5.64, "cer": 2.59},
        "Common_Voice_22":     {"wer": 12.67, "cer": 8.59},
        "Tone_Webinars":       {"wer": 60.55, "cer": 54.20},
        "Tone_Books":          {"wer": 5.25, "cer": 2.75},
        "Tone_Speak":          {"wer": 6.49, "cer": 5.19},
        "Sova_RuDevices":      {"wer": 21.57, "cer": 11.48},
    },
    "Whisper-large-v3":     {
        "Russian_LibriSpeech": {"wer": 11.68, "cer": 2.72},
        "Common_Voice_22":     {"wer": 12.23, "cer": 4.75},
        "Tone_Webinars":       {"wer":  7.77, "cer": 5.36},
        "Tone_Books":          {"wer": 11.95, "cer": 3.41},
        "Tone_Speak":          {"wer":  2.68, "cer": 0.46},
        "Sova_RuDevices":      {"wer": 19.87, "cer": 9.36},
    },
}

BIGBENCH = {
    "Borealis-5B (ckpt-32000)": {
        "overall": 48.9,
        "formal_fallacies": 47.2,
        "navigate": 61.2,
        "object_counting": 36.8,
        "web_of_lies": 50.4,
    },
    "Voxtral-Mini-3B": {
        "overall": 49.2,
        "formal_fallacies": 49.2,
        "navigate": 56.0,
        "object_counting": 44.4,
        "web_of_lies": 47.2,
    },
}

# Training-trajectory data (from eval_results_aggregated.json) — note
# Common_Voice has degraded WER ≥180% on earlier checkpoints, so we plot
# per-split rather than the average.
TRAJECTORY = {
    1800: {"RuLS": 15.19, "CV22": 229.10, "Webinars": 99.67, "Books": 64.70, "Speak": 26.43, "RuDev": 46.02},
    2100: {"RuLS": 14.06, "CV22": 223.92, "Webinars": 99.89, "Books": 54.27, "Speak": 22.32, "RuDev": 46.59},
    2400: {"RuLS": 14.85, "CV22": 181.09, "Webinars": 100.67, "Books": 44.83, "Speak": 24.83, "RuDev": 42.81},
    2700: {"RuLS": 14.79, "CV22": 208.86, "Webinars": 103.22, "Books": 58.09, "Speak": 25.71, "RuDev": 40.63},
    2964: {"RuLS": 12.60, "CV22": 243.67, "Webinars": 100.20, "Books": 57.55, "Speak": 28.32, "RuDev": 41.54},
}


def hf_table_df(metric: str = "wer") -> pd.DataFrame:
    splits = ["Russian_LibriSpeech", "Common_Voice_22", "Tone_Webinars",
              "Tone_Books", "Tone_Speak", "Sova_RuDevices"]
    rows = []
    for model, splits_d in HF_EVAL.items():
        row = {"model": model}
        for s in splits:
            row[s] = splits_d[s][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def hf_bar_plot(metric: str):
    df = hf_table_df(metric).melt(id_vars="model", var_name="split", value_name=metric.upper())
    fig = px.bar(
        df, x="split", y=metric.upper(), color="model",
        barmode="group",
        title=f"{metric.upper()} per split — Borealis vs Whisper",
        text_auto=".1f",
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.4), height=480)
    return fig


def bigbench_plot():
    cats = ["formal_fallacies", "navigate", "object_counting", "web_of_lies"]
    rows = []
    for m, d in BIGBENCH.items():
        for c in cats:
            rows.append({"model": m, "category": c, "accuracy": d[c]})
        rows.append({"model": m, "category": "overall", "accuracy": d["overall"]})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="category", y="accuracy", color="model",
        barmode="group",
        title="BigBenchAudio (1000 samples per benchmark)",
        text_auto=".1f",
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.25), height=460)
    return fig


def trajectory_plot():
    fig = go.Figure()
    steps = sorted(TRAJECTORY.keys())
    splits = ["RuLS", "Books", "Speak", "RuDev", "Webinars", "CV22"]
    for s in splits:
        fig.add_trace(go.Scatter(
            x=steps,
            y=[TRAJECTORY[k][s] for k in steps],
            mode="lines+markers",
            name=s,
        ))
    fig.update_layout(
        title="WER траектория во время обучения (5B instruct, ckpt 1800–2964)",
        xaxis_title="step",
        yaxis_title="WER %",
        height=460,
    )
    return fig


# --- Augmentation curriculum (mirrors borealis/augmentations.py) ------------
AUG_STAGES = [
    {"epoch": 0, "overall_p": 0.00, "description": "Тёплый старт без аугментаций",
     "noise": 0.00, "ir": 0.00, "telephony": 0.00, "codec": 0.00, "specaug": 0.00},
    {"epoch": 1, "overall_p": 0.40, "description": "Лёгкие шумы и эквализация",
     "noise": 0.55, "ir": 0.20, "telephony": 0.10, "codec": 0.05, "specaug": 0.20},
    {"epoch": 3, "overall_p": 0.65, "description": "Интенсивные шумы и телефония",
     "noise": 0.70, "ir": 0.30, "telephony": 0.35, "codec": 0.20, "specaug": 0.40},
    {"epoch": 4, "overall_p": 0.72, "description": "Фокус на телефонию и сильные искажения",
     "noise": 0.75, "ir": 0.40, "telephony": 0.50, "codec": 0.35, "specaug": 0.50},
]


def aug_table_df():
    return pd.DataFrame(AUG_STAGES)


def aug_plot():
    df = aug_table_df()
    fig = go.Figure()
    for col in ["overall_p", "noise", "ir", "telephony", "codec", "specaug"]:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col], mode="lines+markers", name=col,
        ))
    fig.update_layout(
        title="Curriculum аугментаций (вероятности vs эпоха)",
        xaxis_title="эпоха",
        yaxis_title="probability",
        height=420,
    )
    return fig


# --- Static markdown sections ------------------------------------------------
INTRO_MD = """
# Borealis: интерактивный архитектурный блог

**Borealis** — это семейство аудио-LLM для русского языка от VikhrModels.
Под капотом — связка из замороженного аудио-энкодера **Whisper-large-v3**,
обучаемого **адаптера** и языковой модели **Qwen3** (0.6B / 1.7B / 4B).

Этот блог — интерактивная экскурсия по архитектуре и ключевым ablation-исследованиям:
- какие адаптеры пробовали,
- что даёт full fine-tune против обучения только адаптера,
- как влияет размер LLM,
- как устроен curriculum аугментаций,
- и что получилось на RuASRBenchmark / BigBenchAudio.

Архитектуру можно покрутить во встроенном **transformer-xray** —
введите туда `Qwen/Qwen3-4B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-0.6B`
или `openai/whisper-large-v3`, чтобы увидеть граф модели слой за слоем.

🔗 Код: [VikhrModels/Borealis]({repo})  ·  🔬 X-ray: [AlexWortega/transformer-xray]({xray})
""".format(repo=REPO_URL, xray=XRAY_URL)


ARCH_MD = """
## Архитектура пайплайна

```
audio (16kHz)
   │
   ▼
WhisperFeatureExtractor → mel-spectrogram (chunks of 3000 frames)
   │
   ▼
WhisperEncoder  (large-v3, 1280-d, FROZEN)
   │  chunked along time, then concatenated
   ▼
Downsample ×4   (reshape: T,D → T/4, D*4 = 5120-d)
   │
   ▼
AudioLanguageAdapter   ← обучаемое звено №1
   │  выход совпадает по размерности с hidden_size LLM
   ▼
Qwen3 input embeddings ← склейка с текстом по позициям
   │            <|im_start|>system…<|im_start|>user…
   │            <|start_of_audio|>  ← сюда вставляются эмбеддинги аудио
   │            <|end_of_audio|>    ← после них идёт текст вопроса/ответа
   ▼
Qwen3ForCausalLM (обучаемое звено №2, опционально)
   │
   ▼
текст ответа
```

**Ключевые числа:**
- WhisperEncoder выходит в 1280-мерное пространство. После ×4 downsample получается **5120-мерный** вход адаптера.
- Адаптер мапит **5120 → hidden_size LLM** (1024 / 2048 / 2560 для Qwen3-0.6B/1.7B/4B).
- Whisper всегда заморожен — обучаются только адаптер и (опционально) LLM.
- Специальные токены `<|start_of_audio|>` / `<|end_of_audio|>` маркируют слот для аудио-эмбеддингов в текстовом потоке.

### Как посмотреть отдельные блоки
Скопируйте один из идентификаторов в поле модели в X-ray ниже:
- `openai/whisper-large-v3` — аудио-энкодер
- `Qwen/Qwen3-0.6B` · `Qwen/Qwen3-1.7B` · `Qwen/Qwen3-4B` — LLM-варианты
"""


ADAPTER_MD = """
## Адаптер: simple vs deep

В `borealis/modeling.py` живёт два варианта:

**`AudioLanguageAdapter` (simple, default ~31M)** — две Linear-проекции без bias и GELU между ними:
```python
w_in:  Linear(whisper_d * downsample, llm_hidden, bias=False)
gelu:  GELU
w_out: Linear(llm_hidden, llm_hidden, bias=False)
```

**`AudioLanguageAdapterDeep` (deep, ~80–150M)** — глубокий с резидуалами:
```python
input_proj:  Linear(in, hidden_dim) + LN + GELU
N layers:    [LN → MLP(hidden_dim → 2*hidden_dim → hidden_dim) → +residual]
output_proj: LN + Linear(hidden_dim, llm_hidden)
```
где `hidden_dim = int(llm_hidden * expansion)`.

Deep-вариант используется в `configs/Borealis_combined_adapter_8gpu.yaml`
(num_layers=3, expansion=1.5, dropout=0.1) для adapter-only тренировки на Qwen3-4B.

Поиграйтесь с конфигурацией ниже:
"""


TRAIN_MODE_MD = """
## Режим обучения: full fine-tune vs adapter-only

| Параметр | Full fine-tune (`Borealis_5B`) | Adapter-only (`Borealis_IT_fresh` / `combined_adapter_8gpu`) |
|---|---|---|
| Замороженные веса | только Whisper | Whisper **и** Qwen3 |
| Обучаемые веса | адаптер + вся Qwen3 | только адаптер |
| Объём обучения (Qwen3-4B) | ~4B + 31M | ~80M |
| learning rate | 3e-4 (новая) / 2e-5 (resume) | 1e-4 |
| Память GPU | большая (full FT 4B) | существенно меньше |
| Где применяется | ASR-пайплайн (5B/2.4B/1.5B) | Speech-Instructions, Combined adapter |

**Why adapter-only?** Идея в том, чтобы получить из чистого Qwen3 голосового
ассистента, не «забыв» текстовое поведение. Поскольку Qwen3 не трогаем,
текстовые способности сохраняются — а адаптер делает аудио «языком,
который Qwen3 понимает».

**Why full FT?** Для ASR-задачи нужно глубокое выравнивание: модель учится
не только воспринимать звук, но и генерировать строго транскрипционный текст
по русски, что меняет распределение вывода.

### Расписание learning rate
- `Borealis_5B.yaml`: cosine, warmup 5%, lr 3e-4, 5 эпох, batch 16
- `Borealis_5B_instruct.yaml`: cosine, warmup 3%, lr 2e-5 (resume), 3 эпохи, eff. batch 64
- `Borealis_IT_fresh.yaml`: cosine, warmup 3%, lr **1e-4** (выше для adapter-only), eff. batch 64
"""


AUG_MD = """
## Curriculum аугментаций

`AugmentationScheduler` (см. `borealis/augmentations.py`) — `TrainerCallback`,
который меняет аугментации в зависимости от эпохи. Это четырёхступенчатый
curriculum: модель сначала видит чистый звук, потом постепенно — всё более
агрессивные искажения.

В каждом stage задана `overall_p` (вероятность применить хоть что-то)
и набор индивидуальных вероятностей конкретных аугментаций:
- **noise** — фоновый шум из MUSAN (с SNR 10–32 dB)
- **ir** — impulse response (реверберация)
- **telephony** — даунсэмпл до 6–12 kHz + полосовой фильтр
- **codec** — симуляция компрессии (lowpass + шум)
- **specaug** — frequency + time masking

Плюс есть всегда-возможные `gaussian_noise`, `eq`, `gain`, `bandpass`,
`pitch_shift`, `speed`, `clipping`.
"""


RESULTS_MD = """
## Результаты

### RuASRBenchmark — WER / CER по сплитам

Сравнение релизной версии Borealis (HF) с inference-чекпоинтом `step-2898`
и baseline Whisper-large-v3 (см. `eval/eval_results_*.json`):
"""


TRAINING_TRAJECTORY_MD = """
### Траектория обучения 5B-instruct

В `eval_results_aggregated.json` лежат WER по 5 чекпоинтам инструкт-стадии.
Сплит `CV22` ушёл в галлюцинацию (WER ≥ 180%) — это известный паттерн,
когда инструкт-модель «договаривает» вместо точной транскрипции. На
сплитах с речью без отвлекающих факторов (RuLS, Books, Speak, RuDev)
тренд снижения видно отчётливо.
"""


BIGBENCH_MD = """
### BigBenchAudio — аудио-reasoning

`Borealis-5B` сравнивается с `Voxtral-Mini-3B-2507` на 4 категориях
(`formal_fallacies`, `navigate`, `object_counting`, `web_of_lies` — по 250
сэмплов). Borealis заметно лучше на `navigate`, чуть хуже на
`object_counting` — модель умеет следовать инструкциям по аудио,
но хуже считает объекты.
"""


# --- Inference demo (lazy, optional) ---------------------------------------
def make_app():
    with gr.Blocks(
        title="Borealis — interactive architecture blog",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    ) as demo:
        gr.Markdown(INTRO_MD)

        with gr.Tabs():
            with gr.Tab("Архитектура"):
                gr.Markdown(ARCH_MD)
                gr.HTML(
                    f"""
                    <div style="border:1px solid #d6d6e0;border-radius:6px;overflow:hidden;
                                margin-top:12px;background:#fff;">
                        <iframe
                            src="{XRAY_URL}"
                            width="100%"
                            height="780"
                            style="border:0;display:block;"
                            allow="clipboard-write; fullscreen"
                            loading="lazy">
                        </iframe>
                    </div>
                    <p style="font-size:13px;color:#666;margin-top:6px;">
                        ↑ Это <a href="{XRAY_URL}" target="_blank">AlexWortega/transformer-xray</a>:
                        вставьте в его поле модели один из id из списка выше,
                        чтобы увидеть граф Whisper или Qwen3 поблочно.
                    </p>
                    """
                )

            with gr.Tab("Adapter"):
                gr.Markdown(ADAPTER_MD)
                with gr.Row():
                    with gr.Column(scale=1):
                        adapter_type = gr.Radio(
                            choices=["Simple", "Deep"],
                            value="Deep",
                            label="Тип адаптера",
                        )
                        llm_pick = gr.Radio(
                            choices=list(QWEN3.keys()),
                            value="Qwen3-4B",
                            label="LLM backbone",
                        )
                        num_layers = gr.Slider(1, 6, value=3, step=1, label="num_layers (только для deep)")
                        expansion = gr.Slider(1.0, 3.0, value=1.5, step=0.25, label="expansion factor (только для deep)")
                    with gr.Column(scale=2):
                        adapter_out = gr.Markdown(adapter_calc("Deep", "Qwen3-4B", 3, 1.5))
                for inp in (adapter_type, llm_pick, num_layers, expansion):
                    inp.change(
                        adapter_calc,
                        inputs=[adapter_type, llm_pick, num_layers, expansion],
                        outputs=adapter_out,
                    )
                gr.Plot(adapter_compare_plot(), label="Сравнение размеров адаптеров")

            with gr.Tab("Размер LLM"):
                gr.Markdown("""
## Ablation: размер LLM-backbone

В `configs/Borealis_{1.5B,2.4B,5B}.yaml` зашиты три варианта (имена — это
суммарный размер с энкодером и адаптером, не размер LLM):

| Config | LLM | Hidden | Layers | Heads (Q/KV) | Полная модель ~ |
|---|---|---|---|---|---|
| `Borealis_1.5B` | `Unsloth/Qwen3-0.6B` | 1024 | 28 | 16 / 8 | 0.6B LLM + 31M adapter + 0.8B Whisper |
| `Borealis_2.4B` | `Unsloth/Qwen3-1.7B` | 2048 | 28 | 16 / 8 | 1.7B + 41M + 0.8B |
| `Borealis_5B`   | `Qwen/Qwen3-4B-Instruct-2507` | 2560 | 36 | 32 / 8 | 4B + 51M + 0.8B |

Все три обучаются с одинаковыми гиперпараметрами (`bs=16`, `lr=3e-4`,
`cosine`, `5 epochs`, full fine-tune) на одной и той же ASR-смеси
(`ToneBooksPlus`, `ToneSpeak`, `ToneRuLS`, `ToneRuDevices`,
`ToneSlavic[ru]`, `ToneGolosOpus`, `bond005/podlodka_speech` и т.д.).

**Что именно меняется**: только backbone. Это позволяет напрямую померить
выгоду от размера LLM при фиксированном остальном пайплайне.
""")
                gr.DataFrame(
                    pd.DataFrame([
                        {"config": k, **v} for k, v in QWEN3.items()
                    ]),
                    label="Конфигурации Qwen3",
                )
                gr.Markdown(
                    "_Hint:_ можно открыть вкладку **Архитектура** и сравнить две модели "
                    "side-by-side в transformer-xray (там есть compare-режим)."
                )

            with gr.Tab("Train mode"):
                gr.Markdown(TRAIN_MODE_MD)

            with gr.Tab("Аугментации"):
                gr.Markdown(AUG_MD)
                gr.Plot(aug_plot(), label="Вероятности по стадиям")
                gr.DataFrame(aug_table_df(), label="Stages")
                gr.Markdown("""
**Особенности расписания**:
- эпоха 0 — фактически как Whisper-инициализация, чтобы адаптер успел «прижиться»;
- с эпохи 1 включаются мягкие искажения (фон, EQ, лёгкая телефония);
- с эпохи 3 — агрессивный фон, телефония 35%, codec 20%;
- эпоха 4 — упор на телефонию (до 50%) и сильные искажения, имитируя сложные домены типа `Tone_Webinars`.

В коде: расписание реализовано через `TrainerCallback` (`AugmentationScheduler`),
который меняет `dataset.augmentations` на новый `AugmentationPipeline` в начале каждой эпохи.
""")

            with gr.Tab("Результаты"):
                gr.Markdown(RESULTS_MD)
                with gr.Row():
                    metric = gr.Radio(
                        choices=["wer", "cer"],
                        value="wer",
                        label="Метрика",
                    )
                wer_table = gr.DataFrame(hf_table_df("wer"), label="WER %")
                wer_plot = gr.Plot(hf_bar_plot("wer"))

                def _refresh(m):
                    return hf_table_df(m), hf_bar_plot(m)

                metric.change(_refresh, inputs=metric, outputs=[wer_table, wer_plot])

                gr.Markdown(TRAINING_TRAJECTORY_MD)
                gr.Plot(trajectory_plot())

                gr.Markdown(BIGBENCH_MD)
                gr.Plot(bigbench_plot())

                gr.Markdown("""
**Как читать**:
- Borealis-HF baseline — то, что в релизе на HF, лучшая точка усреднения по сплитам.
- На `Tone_Webinars` Whisper-large-v3 **сильно** лучше (7.77 vs 56-60 WER) —
  это говорит о gap'е, который ещё предстоит закрыть для домена «длинная речь
  с реверберацией».
- На `Russian_LibriSpeech` и `Tone_Books` Borealis уже **обгоняет Whisper-large-v3**
  (5.64 vs 11.68 WER на RuLS, 5.25 vs 11.95 на Books).
""")

            with gr.Tab("Конфиги"):
                gr.Markdown("""
## Что в каких конфигах

В репо `configs/` лежат yaml-шаблоны под `train.py` / `train_instruct.py`.
Ниже — сводная таблица ablation-измерений:
""")
                gr.DataFrame(
                    pd.DataFrame([
                        {"config": "Borealis_1.5B.yaml",
                         "llm": "Qwen3-0.6B", "mode": "full FT", "adapter": "simple",
                         "max_text_len": 512, "lr": "3e-4", "epochs": 5,
                         "data": "ASR mix (11 sets)"},
                        {"config": "Borealis_2.4B.yaml",
                         "llm": "Qwen3-1.7B", "mode": "full FT", "adapter": "simple",
                         "max_text_len": 512, "lr": "3e-4", "epochs": 5,
                         "data": "ASR mix (11 sets)"},
                        {"config": "Borealis_5B.yaml",
                         "llm": "Qwen3-4B-Instruct", "mode": "full FT", "adapter": "simple",
                         "max_text_len": 512, "lr": "3e-4", "epochs": 5,
                         "data": "ASR mix (11 sets)"},
                        {"config": "Borealis_5B_instruct.yaml",
                         "llm": "Qwen3-4B", "mode": "full FT (resume)", "adapter": "simple",
                         "max_text_len": 1024, "lr": "2e-5", "epochs": 3,
                         "data": "Speech-Instructions + Speech-Describe + Books"},
                        {"config": "Borealis_5B_ficbook.yaml",
                         "llm": "Qwen3-4B", "mode": "full FT", "adapter": "simple",
                         "max_text_len": 1024, "lr": "—", "epochs": "—",
                         "data": "Ficbook audiobooks"},
                        {"config": "Borealis_IT_fresh.yaml",
                         "llm": "Qwen3-4B", "mode": "adapter-only", "adapter": "simple",
                         "max_text_len": 1024, "lr": "1e-4", "epochs": 3,
                         "data": "Speech-Instructions + ASR (350k) + GrandMaster (100k ru text)"},
                        {"config": "Borealis_combined_adapter_8gpu.yaml",
                         "llm": "Qwen3-4B", "mode": "adapter-only", "adapter": "deep (3L, ×1.5)",
                         "max_text_len": 2048, "lr": "1e-4", "epochs": 3,
                         "data": "Long-audio IT + Speech-Instructions + ASR + emotion classification + GrandMaster"},
                    ]),
                    wrap=True,
                )
                gr.Markdown("""
**Ablation-оси, которые получаются из этих конфигов:**
1. **Размер LLM** — 1.5B vs 2.4B vs 5B (при прочих равных).
2. **Mode** — full FT (5B/5B_instruct) vs adapter-only (IT_fresh / combined_adapter_8gpu).
3. **Adapter type** — simple (везде, кроме combined_adapter_8gpu) vs deep (combined_adapter_8gpu).
4. **Data mix** — чистый ASR / instruct / ASR+instruct+text-only / +classification.
5. **Context length** — 512 (ASR) / 1024 (instruct) / 2048 (long-audio combined).
""")

            with gr.Tab("Про блог"):
                gr.Markdown(f"""
## О блоге

Это Hugging Face Space, который собирает архитектуру Borealis и её ablation-исследования
в одном месте. Все цифры — из реальных файлов в репо:

- `borealis/modeling.py` — формулы размеров адаптеров
- `borealis/augmentations.py` — таблица расписания аугментаций
- `configs/*.yaml` — гиперпараметры обучения
- `eval/eval_results_*.json` — числа метрик

Для интерактивной визуализации внутренней структуры моделей блог встраивает
[**AlexWortega/transformer-xray**]({XRAY_URL}) через iframe. Этот сервис умеет
строить граф любого HF-модельного `config.json` без подгрузки весов и делает
side-by-side сравнение двух моделей.

🔗 [VikhrModels/Borealis на GitHub]({REPO_URL})
🔗 [Все скрипты эвала]({REPO_URL}/tree/main/eval)

**TODO/идеи для расширения:**
- Live-инференс (требует GPU Space + публичных весов).
- Подключить deep-link к transformer-xray, когда там появится `?model=...`.
- Добавить heatmap attention'а адаптера на куске аудио (нужно ckpt + хостинг весов).
""")

    return demo


if __name__ == "__main__":
    make_app().launch()
