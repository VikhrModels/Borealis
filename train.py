import os

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_ENTITY"] = "vikhr-audio"
os.environ["WANDB_PROJECT"] = "Borealis"

from unsloth import FastModel
import random

import jiwer
import numpy as np
import torch
from datasets import Audio, concatenate_datasets, load_dataset
from transformers import (
    Qwen3ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperModel,
)

from borealis.augmentations import (
    AugmentationScheduler,
    default_augmentation_stages,
)
from borealis.dataset import BorealisPretrainDataset
from borealis.modeling import BorealisForConditionalGeneration
from borealis.utils import AudioCollator, clean_dataset

import re
import wandb


torch.backends.cudnn.benchmark = True

NOISE_PATH = "/home/alexw/Project_Audio/Borealis/data_for_augs/musan/flattened_16khz/"
IR_PATH = "/home/alexw/Project_Audio/Borealis/data_for_augs/EchoThiefImpulseResponseLibrary/flattened_16khz/"

ds_one = load_dataset(
    "Vikhrmodels/ToneBooksPlus", columns=["audio", "text"], num_proc=8
)
ds_two = load_dataset("Vikhrmodels/ToneSpeak", columns=["audio", "text"], num_proc=8)
ds_three = load_dataset(
    "Vikhrmodels/ReadyFormatDF2", columns=["audio", "text"], num_proc=8
)
ds_four = load_dataset("Vikhrmodels/ToneRuLS", columns=["audio", "text"], num_proc=8)
ds_five = load_dataset(
    "Vikhrmodels/ToneSlavic", columns=["audio", "sentence"], num_proc=8
)
ds_six = load_dataset(
    "Vikhrmodels/ToneRuDevices", columns=["audio", "text"], num_proc=8
)
ds_seven = load_dataset(
    "Vikhrmodels/ReadyFormatDF", columns=["audio", "text"], num_proc=8
)
ds_eight = load_dataset(
    "Vikhrmodels/ToneRuDevicesAudiobooks", columns=["audio", "text"], num_proc=8
)
ds_nine = load_dataset(
    "bond005/podlodka_speech", columns=["audio", "transcription"], num_proc=8
)
ds_ten = load_dataset(
    "Vikhrmodels/ToneGolosOpus",
    "Crowd",
    columns=["audio", "text"],
    num_proc=8,
)
ds_eleven = load_dataset(
    "Vikhrmodels/ToneGolosOpus",
    "Farfield",
    columns=["audio", "text"],
    num_proc=8,
)

ds_five = ds_five.filter(
    lambda ex: ex.get("locale") is not None and "ru" in str(ex["locale"]).lower(),
    num_proc=20,
)
ds_five = ds_five.rename_column("sentence", "text")

ds_nine = ds_nine.rename_column("transcription", "text")

train_ds_list = [
    ds_one["train"],
    ds_two["train"],
    ds_three["train"],
    ds_four["train"],
    ds_five["train"],
    ds_six["train"],
    ds_seven["train"],
    ds_eight["train"],
    ds_nine["train"],
    ds_ten["train"],
    ds_eleven["train"],
]

for i in range(len(train_ds_list)):
    train_ds_list[i] = train_ds_list[i].cast_column(
        "audio", Audio(sampling_rate=16_000)
    )

combined_train = concatenate_datasets(train_ds_list)


val_ds_list = [
    ds_one["validation"].select(range(279)),
    ds_two["validation"].select(range(279)),
    ds_three["validation"].select(range(279)),
    ds_four["validation"].select(range(279)),
    ds_five["validation"].select(range(279)),
    ds_six["validation"].select(range(279)),
    ds_seven["validation"].select(range(279)),
    ds_eight["validation"].select(range(279)),
    ds_nine["validation"].select(range(20)),
    ds_ten["validation"].select(range(279)),
    ds_eleven["validation"].select(range(279)),
]

for i in range(len(val_ds_list)):
    val_ds_list[i] = val_ds_list[i].cast_column("audio", Audio(sampling_rate=16_000))

combined_val = concatenate_datasets(val_ds_list)

combined_train = clean_dataset(combined_train)
combined_val = clean_dataset(combined_val)

whisper_encoder = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

language_model, tokenizer = FastModel.from_pretrained(
    model_name="Unsloth/Qwen3-1.7B",
    dtype=None,
    auto_model=Qwen3ForCausalLM,
    full_finetuning=True,
)

start_audio_token = "<|start_of_audio|>"
end_audio_token = "<|end_of_audio|>"

tokenizer.add_special_tokens(
    {"additional_special_tokens": [start_audio_token, end_audio_token]}
)

AUGMENTATION_STAGES = default_augmentation_stages(sample_rate=16_000)

train_dataset = BorealisPretrainDataset(
    hf_dataset=combined_train,
    tokenizer=tokenizer,
    feature_extractor=whisper_encoder,
    max_text_len=512,
    augmentations=None,
)

eval_dataset = BorealisPretrainDataset(
    hf_dataset=combined_val,
    tokenizer=tokenizer,
    feature_extractor=whisper_encoder,
    max_text_len=512,
    augmentations=None,
)

collator = AudioCollator()

audio_encoder = WhisperModel.from_pretrained(
    "openai/whisper-large-v3", dtype=torch.bfloat16
).encoder

model = BorealisForConditionalGeneration(
    audio_encoder=audio_encoder, language_model=language_model, tokenizer=tokenizer
)

training_args = TrainingArguments(
    output_dir="./asr_qwen_ckpts",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    dataloader_num_workers=16,
    save_total_limit=7,
    num_train_epochs=5,
    warmup_ratio=0.05,
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
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": 5,
            "early_stopping": True,
            "repetition_penalty": 1.2,
            "temperature": 0.79,
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

            generated_ids = model.generate(mel=gen_inputs["mel"], **self.gen_kwargs)

        return (loss, generated_ids, labels)


def extract_assistant_content(text: str) -> str:
    assistant_match = re.search(
        r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL
    )
    if not assistant_match:
        return text.strip()

    assistant_block = assistant_match.group(1).strip()

    think_match = re.search(r"<think>\n\n</think>\n\n", assistant_block, re.DOTALL)
    if think_match:
        assistant_block = re.sub(
            r"<think>.*?</think>", "", assistant_block, flags=re.DOTALL
        ).strip()

    return assistant_block


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    print(f"Min/Max predictions: {predictions.min()}, {predictions.max()}")

    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
    predictions = np.clip(predictions, 0, len(tokenizer) - 1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_preds = [extract_assistant_content(pred).lower() for pred in decoded_preds]

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [
        extract_assistant_content(label).lower() for label in decoded_labels
    ]

    if len(decoded_preds) > 1:
        indices = random.sample(range(len(decoded_preds)), 5)
        for i in indices:
            print(f"Reference: {decoded_labels[i]}\nGenerated: {decoded_preds[i]}\n")

    wer_score = jiwer.wer(decoded_labels, decoded_preds)
    cer_score = jiwer.cer(decoded_labels, decoded_preds)

    return {"wer": wer_score, "cer": cer_score}


class LoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        num_samples = 5
        indices = random.sample(range(len(eval_dataset)), num_samples)

        table = wandb.Table(columns=["audio", "reference", "generated"])

        for idx in indices:
            example = combined_val[idx]
            audio = example["audio"].get_all_samples().data.squeeze()
            reference = example["text"].lower()

            item = eval_dataset[idx]
            inputs = collator([item])
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = model.generate(mel=inputs["mel"], **trainer.gen_kwargs)
                generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                generated = extract_assistant_content(generated).lower()

            table.add_data(wandb.Audio(audio, sample_rate=16000), reference, generated)

        wandb.log({"eval_samples": table}, step=state.global_step)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.add_callback(
    AugmentationScheduler(
        dataset=train_dataset,
        noise_path=NOISE_PATH,
        ir_path=IR_PATH,
        stages=AUGMENTATION_STAGES,
        sample_rate=16_000,
    )
)

trainer.add_callback(LoggingCallback())

trainer.train()
