from unsloth import FastModel
import os
import yaml
import argparse
import random
import re
import jiwer
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import (
    Qwen3ForCausalLM,
    Trainer,
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
from borealis.utils import (
    AudioCollator,
    clean_dataset,
    load_and_process_dataset,
    convert_numeric_strings,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/Borealis_1.5B.yaml",
    help="Path to the config file.",
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

config = convert_numeric_strings(config)

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_ENTITY"] = config["wandb"]["entity"]
os.environ["WANDB_PROJECT"] = config["wandb"]["project"]

torch.backends.cudnn.benchmark = True


noise_dataset = load_dataset(
    config["datasets"]["noise"]["name"],
    split=config["datasets"]["noise"]["split"],
    num_proc=config["datasets"]["num_proc"],
)
ir_dataset = load_dataset(
    config["datasets"]["ir"]["name"],
    split=config["datasets"]["ir"]["split"],
    num_proc=config["datasets"]["num_proc"],
)

train_ds_list = []
for ds_config in config["datasets"]["train"]:
    split_key = ds_config["split"]
    config_name = ds_config.get("config", None)
    ds = load_and_process_dataset(
        name=ds_config["name"],
        config_name=config_name,
        target_split="train",
        columns=ds_config["columns"],
        num_proc=config["datasets"]["num_proc"],
        sampling_rate=config["datasets"]["sampling_rate"],
        rename_text=ds_config.get("rename_text"),
        rename_audio=ds_config.get("rename_audio"),
        filter_locale_ru=ds_config.get("filter", {}).get("locale_ru", False),
    )
    train_ds_list.append(ds)

combined_train = concatenate_datasets(train_ds_list)

val_ds_list = []
for ds_config in config["datasets"]["val"]:
    split_key = ds_config["split"]
    config_name = ds_config.get("config", None)
    ds = load_and_process_dataset(
        name=ds_config["name"],
        config_name=config_name,
        target_split="validation",
        columns=ds_config["columns"],
        num_proc=config["datasets"]["num_proc"],
        sampling_rate=config["datasets"]["sampling_rate"],
        select_range=ds_config.get("select_range"),
        rename_text=ds_config.get("rename_text"),
        rename_audio=ds_config.get("rename_audio"),
        filter_locale_ru=ds_config.get("filter", {}).get("locale_ru", False),
    )
    val_ds_list.append(ds)

combined_val = concatenate_datasets(val_ds_list)

if config["datasets"]["clean_dataset"]:
    combined_train = clean_dataset(combined_train)
    combined_val = clean_dataset(combined_val)

whisper_encoder = WhisperFeatureExtractor.from_pretrained(
    config["model"]["whisper"]["pretrained"]
)

language_model, tokenizer = FastModel.from_pretrained(
    model_name=config["model"]["language_model"]["pretrained"],
    dtype=None,
    auto_model=Qwen3ForCausalLM,
    full_finetuning=config["model"]["language_model"]["full_finetuning"],
)

start_audio_token = config["model"]["special_tokens"]["start_audio"]
end_audio_token = config["model"]["special_tokens"]["end_audio"]

tokenizer.add_special_tokens(
    {"additional_special_tokens": [start_audio_token, end_audio_token]}
)

if config["augmentation"]["stages"] == "default":
    AUGMENTATION_STAGES = default_augmentation_stages(
        sample_rate=config["augmentation"]["sample_rate"]
    )
else:
    AUGMENTATION_STAGES = config["augmentation"]["stages"]

train_dataset = BorealisPretrainDataset(
    hf_dataset=combined_train,
    tokenizer=tokenizer,
    feature_extractor=whisper_encoder,
    max_text_len=config["model"]["max_text_len"],
    augmentations=None,
)

eval_dataset = BorealisPretrainDataset(
    hf_dataset=combined_val,
    tokenizer=tokenizer,
    feature_extractor=whisper_encoder,
    max_text_len=config["model"]["max_text_len"],
    augmentations=None,
)

collator = AudioCollator()

audio_encoder = WhisperModel.from_pretrained(
    config["model"]["whisper"]["pretrained"],
    dtype=getattr(torch, config["model"]["whisper"]["dtype"]),
).encoder

model = BorealisForConditionalGeneration(
    audio_encoder=audio_encoder, language_model=language_model, tokenizer=tokenizer
)

training_args = TrainingArguments(**config["training"])


class CustomTrainer(Trainer):
    def __init__(self, *args, gen_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_kwargs = gen_kwargs or config["generation"]

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
        indices = random.sample(
            range(len(decoded_preds)), config["metrics"]["random_samples"]
        )
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

trainer.add_callback(
    AugmentationScheduler(
        dataset=train_dataset,
        noise_hf_set=noise_dataset,
        ir_hf_set=ir_dataset,
        stages=AUGMENTATION_STAGES,
        sample_rate=config["augmentation"]["sample_rate"],
    )
)


trainer.train()
