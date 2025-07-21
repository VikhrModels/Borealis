from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, AutoTokenizer
from borealis.dataset import BorealisBaseDataset
from borealis.utils import AudioCollator
from borealis.modeling import BorealisForConditionalGeneration
from transformers import TrainingArguments, Trainer

ds = load_dataset("Vikhrmodels/ToneBooks", cache_dir="../cache_home/")
ds = ds.cast_column("audio", Audio(decode=True, sampling_rate=16_000))

whisper_encoder = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-large-v3-turbo"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

start_audio_token = "<|start_of_audio|>"
end_audio_token = "<|end_of_audio|>"

tokenizer.add_special_tokens(
    {"additional_special_tokens": [start_audio_token, end_audio_token]}
)


train_dataset = BorealisBaseDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    audios=ds["train"]["audio"],
    texts=ds["train"]["text"],
    max_text_len=320,
)

eval_dataset = BorealisBaseDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    audios=ds["validation"]["audio"][:56],
    texts=ds["validation"]["text"][:56],
    max_text_len=320,
)


collator = AudioCollator()

model = BorealisForConditionalGeneration(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./asr_qwen_ckpts",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # gradient_accumulation_steps=1,
    # eval_accumulation_steps=1,
    num_train_epochs=5,
    learning_rate=3e-4,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
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


