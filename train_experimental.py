import os

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

from unsloth import FastModel
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    WhisperFeatureExtractor,
    Qwen2ForCausalLM,
)
from borealis.dataset import BorealisBaseDataset
from borealis.utils import AudioCollator
from borealis.modeling import BorealisForConditionalGeneration
from transformers import TrainingArguments, Trainer
import jiwer
import numpy as np
import torch
import random

from audiomentations import (
    Compose,
    AddBackgroundNoise,
    AddGaussianNoise,
    ApplyImpulseResponse,
    PitchShift,
    Gain,
    Mp3Compression,
    ClippingDistortion,
    BandPassFilter,
)


ds_one = load_dataset("Vikhrmodels/ToneBooksPlus", num_proc=8)
ds_two = load_dataset("Vikhrmodels/ToneSpeak", num_proc=8)
ds_three = load_dataset("Vikhrmodels/ToneWebinars", num_proc=8)
ds_four = load_dataset("Vikhrmodels/ToneRuLS", num_proc=8)
ds_five = load_dataset("Vikhrmodels/ToneSlavic", num_proc=8)

ds_five = ds_five.rename_column("sentence", "text")

ds_one["train"] = ds_one["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_two["train"] = ds_two["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_three["train"] = ds_three["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_four["train"] = ds_four["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_five["train"] = ds_five["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)

train_ds_list = [
    ds_one["train"].select_columns(["audio", "text"]),
    ds_two["train"].select_columns(["audio", "text"]),
    ds_three["train"].select_columns(["audio", "text"]),
    ds_four["train"].select_columns(["audio", "text"]),
    ds_five["train"].select_columns(["audio", "text"]),
]

combined_train = concatenate_datasets(train_ds_list)
combined_train = combined_train.cast_column(
    "audio", Audio(decode=True, sampling_rate=16_000)
)

ds_one["validation"] = ds_one["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_two["validation"] = ds_two["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_three["validation"] = ds_three["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_four["validation"] = ds_four["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_five["validation"] = ds_five["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)

val_ds_one = ds_one["validation"].select_columns(["audio", "text"]).select(range(79))
val_ds_two = ds_two["validation"].select_columns(["audio", "text"]).select(range(79))
val_ds_three = (
    ds_three["validation"].select_columns(["audio", "text"]).select(range(79))
)
val_ds_four = ds_four["validation"].select_columns(["audio", "text"]).select(range(79))
val_ds_five = ds_five["validation"].select_columns(["audio", "text"]).select(range(79))

combined_val = concatenate_datasets(
    [
        val_ds_one,
        val_ds_two,
        val_ds_three,
        val_ds_four,
        val_ds_five,
    ]
)
combined_val = combined_val.cast_column(
    "audio", Audio(decode=True, sampling_rate=16_000)
)


whisper_encoder = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")


language_model, tokenizer = FastModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dtype=None,
    auto_model=Qwen2ForCausalLM,
    full_finetuning=True,
)


start_audio_token = "<|start_of_audio|>"
end_audio_token = "<|end_of_audio|>"

tokenizer.add_special_tokens(
    {"additional_special_tokens": [start_audio_token, end_audio_token]}
)

augment = Compose(
    [
        AddBackgroundNoise(
            sounds_path="/home/alexw/Project_Audio/Borealis/data_for_augs/musan/flattened_16khz/",
            min_snr_db=3.0,
            max_snr_db=30.0,
            p=0.5,
        ),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        ApplyImpulseResponse(ir_path="/home/alexw/Project_Audio/Borealis/data_for_augs/EchoThiefImpulseResponseLibrary/flattened_16khz/", p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        Mp3Compression(min_bitrate=48, max_bitrate=320, p=0.3),
        ClippingDistortion(
            min_percentile_threshold=20, max_percentile_threshold=40, p=0.3
        ),
        BandPassFilter(
            min_center_freq=1500, max_center_freq=4000, min_bandwidth_fraction=1.0, max_bandwidth_fraction=1.99, p=0.3
        ),
    ],
    shuffle=True,
)


train_dataset = BorealisBaseDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    hf_ds=combined_train,
    max_text_len=320,
    augmentations=augment,
)

eval_dataset = BorealisBaseDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    hf_ds=combined_val,
    max_text_len=320,
    augmentations=None,
)


collator = AudioCollator()

model = BorealisForConditionalGeneration(
    language_model=language_model, tokenizer=tokenizer
)

training_args = TrainingArguments(
    output_dir="./asr_qwen_ckpts",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_total_limit=7,
    dataloader_num_workers=16,
    num_train_epochs=3,
    warmup_steps=3000,
    learning_rate=3e-4,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1500,
    save_steps=10000,
    logging_steps=50,
    report_to="wandb",
    save_safetensors=False,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)


class CustomTrainer(Trainer):
    def __init__(self, *args, gen_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 320,
            "do_sample": False,
            "num_beams": 5,
            "early_stopping": True,
            "repetition_penalty": 1.2,
        }

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )

        inputs = self._prepare_inputs(inputs)

        has_labels = "labels" in inputs
        labels = inputs["labels"] if has_labels else None

        with torch.inference_mode():
            if has_labels:
                outputs = model(**inputs)
                loss = outputs[0]
            else:
                loss = None

            gen_inputs = {
                k: v
                for k, v in inputs.items()
                if k != "labels" and k != "text_att_mask"
            }
            gen_inputs["att_mask"] = gen_inputs.pop("audio_att_mask")

            generated_ids = model.generate(
                **gen_inputs, return_tokens=True, **self.gen_kwargs
            )

        return (loss, generated_ids, labels)


def extract_assistant_content(text: str) -> str:
    if "assistant\n" in text:
        return text.split("assistant\n")[-1].strip()
    return text.strip()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    print(f"Min/Max predictions: {predictions.min()}, {predictions.max()}")

    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)

    predictions = np.clip(predictions, 0, len(tokenizer) - 1)

    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_preds = [extract_assistant_content(pred).lower() for pred in decoded_preds]

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [
        extract_assistant_content(label).lower() for label in decoded_labels
    ]

    if len(decoded_preds) > 1:
        indices = random.sample(range(len(decoded_preds)), 2)
        for i in indices:
            print(f"Reference: {decoded_labels[i]}\nGenerated: {decoded_preds[i]}\n")

    wer_score = jiwer.wer(decoded_labels, decoded_preds)
    cer_score = jiwer.cer(decoded_labels, decoded_preds)

    return {"wer": wer_score, "cer": cer_score}


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)


trainer.train()
