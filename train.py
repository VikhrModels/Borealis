from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, AutoTokenizer
from borealis.dataset import BorealisBaseDataset
from borealis.utils import AudioCollator
from borealis.modeling import BorealisForConditionalGeneration
from transformers import TrainingArguments, Trainer

ds_one = load_dataset(
    "Vikhrmodels/ToneBooksPlus", num_proc=8
)  # cache_dir="../cache_home/",
ds_one = ds_one.cast_column("audio", Audio(decode=True, sampling_rate=16_000))

# ds_two = load_dataset(
#     "Vikhrmodels/ToneSpeak", num_proc=8
# )  # cache_dir="../cache_home/",
# ds_two = ds_one.cast_column("audio", Audio(decode=True, sampling_rate=16_000))

# ds_three = load_dataset(
#     "Vikhrmodels/ToneSlavic", num_proc=8
# )  # cache_dir="../cache_home/",
# ds_three = ds_one.cast_column("audio", Audio(decode=True, sampling_rate=16_000))

# ds_four = load_dataset(
#     "Vikhrmodels/ToneWebinars", num_proc=8
# )  # cache_dir="../cache_home/",
# ds_four = ds_one.cast_column("audio", Audio(decode=True, sampling_rate=16_000))

# ds_five = load_dataset(
#     "Vikhrmodels/ToneRuLS", num_proc=8
# )  # cache_dir="../cache_home/",
# ds_five = ds_one.cast_column("audio", Audio(decode=True, sampling_rate=16_000))


whisper_encoder = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

start_audio_token = "<|start_of_audio|>"
end_audio_token = "<|end_of_audio|>"

tokenizer.add_special_tokens(
    {"additional_special_tokens": [start_audio_token, end_audio_token]}
)


train_dataset = BorealisBaseDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    audios=list(ds_one["train"]["audio"]),
    # + list(ds_two["train"]["audio"])
    # + list(ds_three["train"]["audio"])
    # + list(ds_four["train"]["audio"])
    # + list(ds_five["train"]["audio"]),
    texts=list(ds_one["train"]["text"]),
    # + list(ds_two["train"]["text"])
    # + list(ds_three["train"]["text"])
    # + list(ds_four["train"]["text"])
    # + list(ds_five["train"]["text"]),
    max_text_len=320,
)

eval_dataset = BorealisBaseDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    audios=list(ds_one["validation"]["audio"])[:79],
    # + list(ds_two["validation"]["audio"])[:79]
    # + list(ds_three["validation"]["audio"])[:79]
    # + list(ds_four["validation"]["audio"])[:79]
    # + list(ds_five["validation"]["audio"])[:79],
    texts=list(ds_one["validation"]["text"])[:79],
    # + list(ds_two["validation"]["text"])[:79]
    # + list(ds_three["validation"]["text"])[:79]
    # + list(ds_four["validation"]["text"])[:79]
    # + list(ds_five["validation"]["text"])[:79],
    max_text_len=320,
)


collator = AudioCollator()

model = BorealisForConditionalGeneration(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./asr_qwen_ckpts",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    # gradient_accumulation_steps=1,
    # eval_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    report_to="wandb",
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
)


trainer.train()
