from unsloth import FastModel
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperFeatureExtractor, Qwen2ForCausalLM
from borealis.dataset import BorealisBaseDataset
from borealis.utils import AudioCollator
from borealis.modeling import BorealisForConditionalGeneration
from transformers import TrainingArguments, Trainer


ds_one = load_dataset("Vikhrmodels/ToneBooksPlus", num_proc=8)
ds_two = load_dataset("Vikhrmodels/ToneSpeak", num_proc=8)
ds_three = load_dataset("Vikhrmodels/ToneWebinars", num_proc=8)
ds_four = load_dataset("Vikhrmodels/ToneRuLS", num_proc=8)

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


train_ds_list = [
    ds_one["train"].select_columns(["audio", "text"]),
    ds_two["train"].select_columns(["audio", "text"]),
    ds_three["train"].select_columns(["audio", "text"]),
    ds_four["train"].select_columns(["audio", "text"]),
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

val_ds_one = ds_one["validation"].select_columns(["audio", "text"]).select(range(79))
val_ds_two = ds_two["validation"].select_columns(["audio", "text"]).select(range(79))
val_ds_three = (
    ds_three["validation"].select_columns(["audio", "text"]).select(range(79))
)
val_ds_four = ds_four["validation"].select_columns(["audio", "text"]).select(range(79))

combined_val = concatenate_datasets(
    [
        val_ds_one,
        val_ds_two,
        val_ds_three,
        val_ds_four,
    ]
)
combined_val = combined_val.cast_column(
    "audio", Audio(decode=True, sampling_rate=16_000)
)


whisper_encoder = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

language_model, tokenizer = FastModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B",
    dtype=None,
    auto_model=Qwen2ForCausalLM,
    full_finetuning=True,
)


start_audio_token = "<|start_of_audio|>"
end_audio_token = "<|end_of_audio|>"

tokenizer.add_special_tokens(
    {"additional_special_tokens": [start_audio_token, end_audio_token]}
)


train_dataset = BorealisBaseDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    hf_ds=combined_train,
    max_text_len=320,
)

eval_dataset = BorealisBaseDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    hf_ds=combined_val,
    max_text_len=320,
)


collator = AudioCollator()

model = BorealisForConditionalGeneration(
    language_model=language_model, tokenizer=tokenizer
)

training_args = TrainingArguments(
    output_dir="./asr_qwen_ckpts",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    # gradient_accumulation_steps=1,
    # eval_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=3e-4,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    save_steps=10000,
    logging_steps=50,
    report_to="wandb",
    save_safetensors=False,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
)


trainer.train()
