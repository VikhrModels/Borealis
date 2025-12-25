# from unsloth import FastModel  # Disabled for multi-GPU
import os

# Disable torchcodec, use soundfile instead
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"
import yaml
import argparse
import random
import re
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk, Audio
from huggingface_hub import hf_hub_download
import os
from transformers import (
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperModel,
)
from liger_kernel.transformers import apply_liger_kernel_to_qwen2

# Apply Liger Kernel optimizations for Qwen
apply_liger_kernel_to_qwen2()

# Augmentations disabled
# from borealis.augmentations import (
#     AugmentationScheduler,
#     default_augmentation_stages,
# )
from borealis.dataset import BorealisInstructDataset, BorealisTextOnlyDataset
from borealis.modeling import BorealisForConditionalGeneration
from borealis.utils import AudioCollator, convert_numeric_strings
# from unsloth.chat_templates import get_chat_template  # Disabled for multi-GPU

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/Borealis_5B_instruct.yaml",
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


def load_instruct_dataset(ds_config, num_proc, sampling_rate):
    """Load a single instruct dataset from HuggingFace Hub or local disk."""
    name = ds_config["name"]
    split_key = ds_config["split"]

    # Check if it's a local path
    if os.path.exists(name):
        print(f"    Loading from local path: {name}")
        ds = load_from_disk(name)
    else:
        ds = load_dataset(
            name,
            split=split_key,
            num_proc=num_proc,
        )

    # Cast audio column if it exists and isn't already cast
    audio_col = ds_config.get("audio_column", "audio")
    if audio_col in ds.column_names:
        ds = ds.cast_column(audio_col, Audio(sampling_rate=sampling_rate))

    return ds, ds_config


# Noise/IR datasets disabled (no augmentations)
# noise_dataset = load_dataset(
#     config["datasets"]["noise"]["name"],
#     split=config["datasets"]["noise"]["split"],
#     num_proc=config["datasets"]["num_proc"],
# )
# ir_dataset = load_dataset(
#     config["datasets"]["ir"]["name"],
#     split=config["datasets"]["ir"]["split"],
#     num_proc=config["datasets"]["num_proc"],
# )

print("Loading training datasets...")
train_ds_list = []
train_ds_configs = []
for ds_config in config["datasets"]["train"]:
    print(f"  Loading {ds_config['name']} ({ds_config['split']})...")
    ds, cfg = load_instruct_dataset(
        ds_config,
        config["datasets"]["num_proc"],
        config["datasets"]["sampling_rate"],
    )
    if "select_range" in ds_config:
        ds = ds.select(range(min(ds_config["select_range"], len(ds))))
    train_ds_list.append(ds)
    train_ds_configs.append(cfg)
    print(f"    Loaded {len(ds)} examples")

print("Loading validation datasets...")
val_ds_list = []
val_ds_configs = []
for ds_config in config["datasets"]["val"]:
    print(f"  Loading {ds_config['name']} ({ds_config['split']})...")
    ds, cfg = load_instruct_dataset(
        ds_config,
        config["datasets"]["num_proc"],
        config["datasets"]["sampling_rate"],
    )
    if "select_range" in ds_config:
        ds = ds.select(range(min(ds_config["select_range"], len(ds))))
    val_ds_list.append(ds)
    val_ds_configs.append(cfg)
    print(f"    Loaded {len(ds)} examples")

# Load text-only datasets if configured
text_ds_list = []
text_ds_configs = []
if "text_train" in config["datasets"]:
    print("Loading text-only datasets...")
    for ds_config in config["datasets"]["text_train"]:
        print(f"  Loading {ds_config['name']} ({ds_config.get('split', 'train')})...")
        ds = load_dataset(
            ds_config["name"],
            split=ds_config.get("split", "train"),
            num_proc=config["datasets"]["num_proc"],
        )
        # Filter by language if specified
        if "filter_lang" in ds_config:
            lang = ds_config["filter_lang"]
            print(f"    Filtering by language: {lang}")
            ds = ds.filter(lambda x: x.get("prompt_lang") == lang or x.get("answer_lang") == lang)
        if "select_range" in ds_config:
            ds = ds.select(range(min(ds_config["select_range"], len(ds))))
        text_ds_list.append(ds)
        text_ds_configs.append(ds_config)
        print(f"    Loaded {len(ds)} text examples")

whisper_encoder = WhisperFeatureExtractor.from_pretrained(
    config["model"]["whisper"]["pretrained"]
)

from transformers import AutoTokenizer

language_model = Qwen3ForCausalLM.from_pretrained(
    config["model"]["language_model"]["pretrained"],
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

tokenizer = AutoTokenizer.from_pretrained(
    config["model"]["language_model"]["pretrained"],
    trust_remote_code=True,
)

start_audio_token = config["model"]["special_tokens"]["start_audio"]
end_audio_token = config["model"]["special_tokens"]["end_audio"]

tokenizer.add_special_tokens(
    {"additional_special_tokens": [start_audio_token, end_audio_token]}
)

# Augmentation stages disabled
# if config["augmentation"]["stages"] == "default":
#     AUGMENTATION_STAGES = default_augmentation_stages(
#         sample_rate=config["augmentation"]["sample_rate"]
#     )
# else:
#     AUGMENTATION_STAGES = config["augmentation"]["stages"]

default_system = config.get("instruct", {}).get(
    "default_system_prompt",
    "You are a helpful voice assistant. Listen to the audio and respond appropriately."
)


class CombinedInstructDataset(torch.utils.data.Dataset):
    """Combines multiple instruct datasets with their configs."""

    def __init__(
        self,
        datasets_list,
        configs_list,
        tokenizer,
        feature_extractor,
        max_text_len,
        default_system_prompt,
        augmentations=None,
        text_datasets_list=None,
        text_configs_list=None,
    ):
        self.datasets = []
        self.lengths = []
        self.cumulative_lengths = [0]

        # Add audio datasets
        for ds, cfg in zip(datasets_list, configs_list):
            wrapped = BorealisInstructDataset(
                hf_dataset=ds,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                max_text_len=max_text_len,
                default_system_prompt=default_system_prompt,
                question_column=cfg.get("question_column", "question"),
                answer_column=cfg.get("answer_column", "answer"),
                audio_column=cfg.get("audio_column", "audio"),
                system_column=cfg.get("system_column"),
                static_question=cfg.get("static_question"),
                augmentations=augmentations,
            )
            self.datasets.append(wrapped)
            self.lengths.append(len(ds))
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))

        # Add text-only datasets
        if text_datasets_list and text_configs_list:
            for ds, cfg in zip(text_datasets_list, text_configs_list):
                wrapped = BorealisTextOnlyDataset(
                    hf_dataset=ds,
                    tokenizer=tokenizer,
                    feature_extractor=feature_extractor,
                    max_text_len=max_text_len,
                    default_system_prompt="You are a helpful assistant.",
                    conversation_column=cfg.get("conversation_column", "conversation"),
                )
                self.datasets.append(wrapped)
                self.lengths.append(len(ds))
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(
            zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])
        ):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        raise IndexError(f"Index {idx} out of range")

    def set_augmentations(self, aug):
        for ds in self.datasets:
            if hasattr(ds, 'augmentations'):
                ds.augmentations = aug


train_dataset = CombinedInstructDataset(
    datasets_list=train_ds_list,
    configs_list=train_ds_configs,
    tokenizer=tokenizer,
    feature_extractor=whisper_encoder,
    max_text_len=config["model"]["max_text_len"],
    default_system_prompt=default_system,
    augmentations=None,
    text_datasets_list=text_ds_list if text_ds_list else None,
    text_configs_list=text_ds_configs if text_ds_configs else None,
)

eval_dataset = CombinedInstructDataset(
    datasets_list=val_ds_list,
    configs_list=val_ds_configs,
    tokenizer=tokenizer,
    feature_extractor=whisper_encoder,
    max_text_len=config["model"]["max_text_len"],
    default_system_prompt=default_system,
    augmentations=None,
)

print(f"Total training examples: {len(train_dataset)}")
print(f"Total validation examples: {len(eval_dataset)}")

collator = AudioCollator()

audio_encoder = WhisperModel.from_pretrained(
    config["model"]["whisper"]["pretrained"],
    dtype=getattr(torch, config["model"]["whisper"]["dtype"]),
).encoder

model = BorealisForConditionalGeneration(
    audio_encoder=audio_encoder, language_model=language_model, tokenizer=tokenizer
)

# Freeze LLM if full_finetuning is False (train only adapter/projector)
full_finetuning = config["model"].get("full_finetuning", True)
if not full_finetuning:
    print("Freezing LLM, training only adapter/projector...")
    for p in model.llm.parameters():
        p.requires_grad = False
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
else:
    print("Full finetuning: training LLM + adapter")

# Load checkpoint weights if provided
checkpoint_path = config["model"].get("checkpoint_path")
if checkpoint_path:
    print(f"Loading weights from checkpoint: {checkpoint_path}")
    # Check if checkpoint_path is a HuggingFace repo (contains /) or local file
    if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
        # It's a HuggingFace repo, download the checkpoint
        print(f"Downloading checkpoint from HuggingFace: {checkpoint_path}")
        local_ckpt_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="pytorch_model.bin",
            cache_dir=".cache/checkpoints"
        )
        checkpoint = torch.load(local_ckpt_path, map_location="cpu", weights_only=False)
    else:
        # Local file path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"Missing keys sample: {missing[:5]}")
    del checkpoint

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
    matches = re.findall(
        r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", text, re.DOTALL
    )
    return matches[-1].strip() if matches else ""


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    print(f"Min/Max predictions: {predictions.min()}, {predictions.max()}")

    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
    predictions = np.clip(predictions, 0, len(tokenizer) - 1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    decoded_labels = [extract_assistant_content(label) for label in decoded_labels]

    if len(decoded_preds) > 1:
        indices = random.sample(
            range(len(decoded_preds)),
            min(config["metrics"]["random_samples"], len(decoded_preds)),
        )
        for i in indices:
            print(f"Reference: {decoded_labels[i][:200]}...")
            print(f"Generated: {decoded_preds[i][:200]}...\n")

    exact_match = sum(
        1 for p, l in zip(decoded_preds, decoded_labels)
        if p.strip().lower() == l.strip().lower()
    ) / len(decoded_preds)

    return {"exact_match": exact_match}


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# Disable augmentations due to torchcodec/FFmpeg dependency issues
# trainer.add_callback(
#     AugmentationScheduler(
#         dataset=train_dataset,
#         noise_hf_set=noise_dataset,
#         ir_hf_set=ir_dataset,
#         stages=AUGMENTATION_STAGES,
#         sample_rate=config["augmentation"]["sample_rate"],
#     )
# )

print("Starting training...")
resume_checkpoint = config.get("training", {}).get("resume_from_checkpoint")
if resume_checkpoint:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    trainer.train()
print("Training complete!")
