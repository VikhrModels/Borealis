import os

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_ENTITY"] = "vikhr-audio"
os.environ["WANDB_PROJECT"] = "Borealis"

from unsloth import FastModel
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperFeatureExtractor, Qwen3ForCausalLM, WhisperModel
from borealis.dataset import BorealisPretrainDataset
from borealis.utils import AudioCollator, _filter_and_report
from borealis.modeling import BorealisForConditionalGeneration
from transformers import TrainingArguments, Trainer, TrainerCallback
import jiwer
import numpy as np
import torch
import random
import torchaudio.functional as F
import torchaudio.transforms as T
import torchaudio

from io import BytesIO
from pathlib import Path

torch.backends.cudnn.benchmark = True

ds_one = load_dataset("Vikhrmodels/ToneBooksPlus", num_proc=8)
ds_two = load_dataset("Vikhrmodels/ToneSpeak", num_proc=8)
ds_three = load_dataset("Vikhrmodels/ReadyFormatDF2", num_proc=8)
ds_four = load_dataset("Vikhrmodels/ToneRuLS", num_proc=8)
ds_five = load_dataset("Vikhrmodels/ToneSlavic", num_proc=8)
ds_six = load_dataset("Vikhrmodels/ToneRuDevices", num_proc=8)
ds_seven = load_dataset("Vikhrmodels/ReadyFormatDF", num_proc=8)
ds_eight = load_dataset("Vikhrmodels/ToneRuDevicesAudiobooks", num_proc=8)
ds_nine = load_dataset("bond005/podlodka_speech", num_proc=8)
ds_ten = load_dataset("Vikhrmodels/ToneGolosOpus", "Crowd", num_proc=8)
ds_eleven = load_dataset("Vikhrmodels/ToneGolosOpus", "Farfield", num_proc=8)

ds_five = ds_five.filter(
    lambda ex: ex.get("locale") is not None and "ru" in str(ex["locale"]).lower(),
    num_proc=20,
)
ds_five = ds_five.rename_column("sentence", "text")

ds_nine = ds_nine.rename_column("transcription", "text")
ds_nine = ds_nine.remove_columns(["episode", "title"])

ds_ten = ds_ten.remove_columns(["original_text"])
ds_eleven = ds_eleven.remove_columns(["original_text"])

ds_ten = ds_ten.filter(
    lambda example: example["text"] is not None and example["text"].strip() != "",
    num_proc=20,
)
ds_eleven = ds_eleven.filter(
    lambda example: example["text"] is not None and example["text"].strip() != "",
    num_proc=20,
)

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
ds_six["train"] = ds_six["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_seven["train"] = ds_seven["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_eight["train"] = ds_eight["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_nine = ds_nine.cast_column("audio", Audio(sampling_rate=None, decode=True))
ds_ten["train"] = ds_ten["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_eleven["train"] = ds_eleven["train"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)

train_ds_list = [
    ds_one["train"].select_columns(["audio", "text"]),
    ds_two["train"].select_columns(["audio", "text"]),
    ds_three["train"].select_columns(["audio", "text"]),
    ds_four["train"].select_columns(["audio", "text"]),
    ds_five["train"].select_columns(["audio", "text"]),
    ds_six["train"].select_columns(["audio", "text"]),
    ds_seven["train"].select_columns(["audio", "text"]),
    ds_eight["train"].select_columns(["audio", "text"]),
    ds_nine["train"].select_columns(["audio", "text"]),
    ds_ten["train"].select_columns(["audio", "text"]),
    ds_eleven["train"].select_columns(["audio", "text"]),
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
ds_six["validation"] = ds_six["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_seven["validation"] = ds_seven["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_eight["validation"] = ds_eight["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_nine["validation"] = ds_nine["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_ten["validation"] = ds_ten["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)
ds_eleven["validation"] = ds_eleven["validation"].cast_column(
    "audio", Audio(sampling_rate=None, decode=True)
)

val_ds_one = ds_one["validation"].select_columns(["audio", "text"]).select(range(279))
val_ds_two = ds_two["validation"].select_columns(["audio", "text"]).select(range(279))
val_ds_three = (
    ds_three["validation"].select_columns(["audio", "text"]).select(range(279))
)
val_ds_four = ds_four["validation"].select_columns(["audio", "text"]).select(range(279))
val_ds_five = ds_five["validation"].select_columns(["audio", "text"]).select(range(279))
val_ds_six = ds_six["validation"].select_columns(["audio", "text"]).select(range(279))
val_ds_seven = (
    ds_seven["validation"].select_columns(["audio", "text"]).select(range(279))
)
val_ds_eight = (
    ds_eight["validation"].select_columns(["audio", "text"]).select(range(279))
)
val_ds_nine = ds_nine["validation"].select_columns(["audio", "text"]).select(range(20))
val_ds_ten = ds_ten["validation"].select_columns(["audio", "text"]).select(range(279))
val_ds_eleven = (
    ds_eleven["validation"].select_columns(["audio", "text"]).select(range(279))
)

combined_val = concatenate_datasets(
    [
        val_ds_one,
        val_ds_two,
        val_ds_three,
        val_ds_four,
        val_ds_five,
        val_ds_six,
        val_ds_seven,
        val_ds_eight,
        val_ds_nine,
        val_ds_ten,
        val_ds_eleven,
    ]
)
combined_val = combined_val.cast_column(
    "audio", Audio(decode=True, sampling_rate=16_000)
)

combined_train = _filter_and_report(combined_train, "train")
combined_val = _filter_and_report(combined_val, "validation")

whisper_encoder = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

language_model, tokenizer = FastModel.from_pretrained(
    model_name="Vikhrmodels/ReadyFormatDF2",
    dtype=None,
    auto_model=Qwen3ForCausalLM,
    full_finetuning=True,
)

start_audio_token = "<|start_of_audio|>"
end_audio_token = "<|end_of_audio|>"

tokenizer.add_special_tokens(
    {"additional_special_tokens": [start_audio_token, end_audio_token]}
)

NOISE_PATH = "/workspace/Borealis/data_for_augs/musan/flattened_16khz/"
IR_PATH = (
    "/workspace/Borealis/data_for_augs/EchoThiefImpulseResponseLibrary/flattened_16khz/"
)


class AugmentCompose:
    def __init__(
        self,
        noise_path: str,
        ir_path: str,
        snr_min: float,
        snr_max: float,
        p_noise: float,
        p_ir: float,
        p_eq: float,
        eq_min_gain_db: float,
        eq_max_gain_db: float,
        p_heavy: float,
        heavy_hp_prob: float,
        heavy_lp_prob: float,
        resample_min_sr: int,
        resample_max_sr: int,
        alias_min_sr: int,
        alias_max_sr: int,
        enable_mp3: bool,
        mp3_min_bitrate: int,
        mp3_max_bitrate: int,
        overall_p: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.noise_path = Path(noise_path)
        self.ir_path = Path(ir_path)
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.p_noise = p_noise
        self.p_ir = p_ir
        self.p_eq = p_eq
        self.eq_min_gain_db = eq_min_gain_db
        self.eq_max_gain_db = eq_max_gain_db
        self.p_heavy = p_heavy
        self.heavy_hp_prob = heavy_hp_prob
        self.heavy_lp_prob = heavy_lp_prob
        self.resample_min_sr = resample_min_sr
        self.resample_max_sr = resample_max_sr
        self.alias_min_sr = alias_min_sr
        self.alias_max_sr = alias_max_sr
        self.enable_mp3 = enable_mp3
        self.mp3_min_bitrate = mp3_min_bitrate
        self.mp3_max_bitrate = mp3_max_bitrate
        self.overall_p = overall_p
        self.sample_rate = sample_rate

        # Предзагрузка шумов и IR как tensors
        self.noises = [
            torchaudio.load(str(p))[0] for p in self.noise_path.glob("*") if p.is_file()
        ]
        self.irs = [
            torchaudio.load(str(p))[0] for p in self.ir_path.glob("*") if p.is_file()
        ]

        # Фиксированные частоты для 7-band EQ (примерные октавы)
        self.eq_freqs = [60, 150, 400, 1000, 2400, 6000, 15000]

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self.sample_rate:
            waveform = T.Resample(sample_rate, self.sample_rate)(waveform)
            sample_rate = self.sample_rate

        if torch.rand(1).item() >= self.overall_p:
            return waveform  # Нет аугментаций

        original_length = waveform.shape[-1]

        # OneOf для шума (только один тип, без дублирования)
        if torch.rand(1).item() < self.p_noise:
            if torch.rand(1).item() < 0.5:  # AddBackgroundNoise
                if self.noises:
                    noise = random.choice(self.noises)
                    noise = (
                        noise[:, :original_length]
                        if noise.shape[1] > original_length
                        else F.pad(noise, (0, original_length - noise.shape[1]))
                    )
                    snr_db = torch.tensor([torch.uniform(self.snr_min, self.snr_max)])
                    waveform = F.add_noise(waveform, noise, snr_db)
            else:  # AddGaussianNoise
                amplitude = torch.uniform(0.002, 0.01)
                noise = torch.randn_like(waveform) * amplitude
                waveform += noise  # Прямое добавление, как гауссов шум

        # OneOf для IR/Gain (только один, без дублирования)
        if torch.rand(1).item() < self.p_ir:
            if torch.rand(1).item() < 0.5:  # ApplyImpulseResponse
                if self.irs:
                    ir = random.choice(self.irs)
                    ir = ir / torch.linalg.vector_norm(ir, ord=2)  # Нормализация
                    waveform = F.fftconvolve(waveform, ir)[
                        :, :original_length
                    ]  # Обрезать до оригинала
            else:  # Gain
                gain_db = torch.uniform(-6.0, 6.0)  # Общий диапазон
                waveform = F.gain(waveform, gain_db)

        # SevenBandParametricEQ (независимый блок, без дублирования с фильтрами)
        if torch.rand(1).item() < self.p_eq:
            for freq in self.eq_freqs:
                gain_db = torch.uniform(self.eq_min_gain_db, self.eq_max_gain_db)
                waveform = F.peaking_eq(waveform, sample_rate, freq, gain_db, q=1.0)

        # OneOf для heavy (только один вариант, с опциональными HP/LP, без дублирования)
        if torch.rand(1).item() < self.p_heavy:
            heavy_choice = random.randint(0, 2 if self.enable_mp3 else 1)
            if heavy_choice == 0:  # Resample heavy
                temp_sr = int(
                    torch.uniform(self.resample_min_sr, self.resample_max_sr).item()
                )
                resampler_down = T.Resample(sample_rate, temp_sr)
                resampler_up = T.Resample(temp_sr, sample_rate)
                waveform = resampler_up(resampler_down(waveform))
            elif heavy_choice == 1:  # Aliasing heavy (приближение через decimation)
                temp_sr = int(
                    torch.uniform(self.alias_min_sr, self.alias_max_sr).item()
                )
                factor = sample_rate // temp_sr
                waveform_down = waveform[:, ::factor][
                    :, : temp_sr * (original_length // sample_rate)
                ]
                waveform = F.interpolate(
                    waveform_down.unsqueeze(0), size=original_length, mode="linear"
                ).squeeze(0)
            else:  # MP3 heavy
                bitrate = int(
                    torch.uniform(self.mp3_min_bitrate, self.mp3_max_bitrate).item()
                )
                with BytesIO() as buf:
                    torchaudio.save(
                        buf,
                        waveform,
                        sample_rate,
                        format="mp3",
                        bits_per_sample=bitrate,
                    )
                    buf.seek(0)
                    waveform, _ = torchaudio.load(buf)

            # Опциональные HP/LP для heavy (независимо от выбора)
            if torch.rand(1).item() < self.heavy_hp_prob:
                cutoff = torch.uniform(250.0, 450.0).item()
                waveform = F.highpass_biquad(waveform, sample_rate, cutoff)
            if torch.rand(1).item() < self.heavy_lp_prob:
                cutoff = torch.uniform(3000.0, 3800.0).item()
                waveform = F.lowpass_biquad(waveform, sample_rate, cutoff)

        # Normalize всегда в конце цепочки
        waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(
            min=1e-8
        )

        return waveform


def build_augment(
    noise_path: str,
    ir_path: str,
    snr_min: float,
    snr_max: float,
    p_noise: float,
    p_ir: float,
    p_eq: float,
    eq_min_gain_db: float,
    eq_max_gain_db: float,
    p_heavy: float,
    heavy_hp_prob: float,
    heavy_lp_prob: float,
    resample_min_sr: int,
    resample_max_sr: int,
    alias_min_sr: int,
    alias_max_sr: int,
    enable_mp3: bool,
    mp3_min_bitrate: int,
    mp3_max_bitrate: int,
    overall_p: float = 0.5,
) -> AugmentCompose:
    return AugmentCompose(
        noise_path=noise_path,
        ir_path=ir_path,
        snr_min=snr_min,
        snr_max=snr_max,
        p_noise=p_noise,
        p_ir=p_ir,
        p_eq=p_eq,
        eq_min_gain_db=eq_min_gain_db,
        eq_max_gain_db=eq_max_gain_db,
        p_heavy=p_heavy,
        heavy_hp_prob=heavy_hp_prob,
        heavy_lp_prob=heavy_lp_prob,
        resample_min_sr=resample_min_sr,
        resample_max_sr=resample_max_sr,
        alias_min_sr=alias_min_sr,
        alias_max_sr=alias_max_sr,
        enable_mp3=enable_mp3,
        mp3_min_bitrate=mp3_min_bitrate,
        mp3_max_bitrate=mp3_max_bitrate,
        overall_p=overall_p,
    )


augment = build_augment(
    NOISE_PATH,
    IR_PATH,
    snr_min=18.0,
    snr_max=28.0,
    p_noise=0.40,
    p_ir=0.0,
    p_eq=0.25,
    eq_min_gain_db=-6.0,
    eq_max_gain_db=6.0,
    p_heavy=0.0,
    heavy_hp_prob=0.0,
    heavy_lp_prob=0.0,
    resample_min_sr=9000,
    resample_max_sr=12000,
    alias_min_sr=7000,
    alias_max_sr=11000,
    enable_mp3=False,
    mp3_min_bitrate=128,
    mp3_max_bitrate=192,
    overall_p=0.45,
)

train_dataset = BorealisPretrainDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    hf_ds=combined_train,
    max_text_len=320,
    augmentations=augment,
)

eval_dataset = BorealisPretrainDataset(
    audio_processor=whisper_encoder,
    text_tokenizer=tokenizer,
    hf_ds=combined_val,
    max_text_len=320,
    augmentations=None,
)

collator = AudioCollator()

audio_encoder = WhisperModel.from_pretrained(
    "openai/whisper-large-v3", dtype=torch.bfloat16
).encoder

model = BorealisForConditionalGeneration(
    audio_encoder=whisper_encoder, language_model=language_model, tokenizer=tokenizer
)

training_args = TrainingArguments(
    output_dir="./asr_qwen_ckpts",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
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


class AugSchedule(TrainerCallback):
    def __init__(self, dataset, noise_path, ir_path):
        self.dataset = dataset
        self.noise_path = noise_path
        self.ir_path = ir_path

    def on_epoch_begin(self, args, state, control, **kwargs):
        ep = int(state.epoch or 0)
        if ep < 1:
            cfg = dict(
                snr_min=18.0,
                snr_max=28.0,
                p_noise=0.40,
                p_ir=0.0,
                p_eq=0.25,
                eq_min_gain_db=-6.0,
                eq_max_gain_db=6.0,
                p_heavy=0.0,
                heavy_hp_prob=0.0,
                heavy_lp_prob=0.0,
                resample_min_sr=9000,
                resample_max_sr=12000,
                alias_min_sr=7000,
                alias_max_sr=11000,
                enable_mp3=False,
                mp3_min_bitrate=128,
                mp3_max_bitrate=192,
                overall_p=0.45,
            )
        elif ep < 3:
            cfg = dict(
                snr_min=12.0,
                snr_max=22.0,
                p_noise=0.50,
                p_ir=0.12,
                p_eq=0.38,
                eq_min_gain_db=-9.0,
                eq_max_gain_db=9.0,
                p_heavy=0.12,
                heavy_hp_prob=0.55,
                heavy_lp_prob=0.55,
                resample_min_sr=7000,
                resample_max_sr=11000,
                alias_min_sr=5000,
                alias_max_sr=10000,
                enable_mp3=False,
                mp3_min_bitrate=96,
                mp3_max_bitrate=160,
                overall_p=0.50,
            )
        elif ep < 4:
            cfg = dict(
                snr_min=10.0,
                snr_max=20.0,
                p_noise=0.55,
                p_ir=0.18,
                p_eq=0.42,
                eq_min_gain_db=-12.0,
                eq_max_gain_db=12.0,
                p_heavy=0.18,
                heavy_hp_prob=0.65,
                heavy_lp_prob=0.65,
                resample_min_sr=6000,
                resample_max_sr=10000,
                alias_min_sr=4000,
                alias_max_sr=9000,
                enable_mp3=True,
                mp3_min_bitrate=96,
                mp3_max_bitrate=160,
                overall_p=0.50,
            )
        else:
            cfg = dict(
                snr_min=12.0,
                snr_max=22.0,
                p_noise=0.50,
                p_ir=0.12,
                p_eq=0.35,
                eq_min_gain_db=-9.0,
                eq_max_gain_db=9.0,
                p_heavy=0.10,
                heavy_hp_prob=0.60,
                heavy_lp_prob=0.60,
                resample_min_sr=7000,
                resample_max_sr=11000,
                alias_min_sr=5000,
                alias_max_sr=10000,
                enable_mp3=True,
                mp3_min_bitrate=96,
                mp3_max_bitrate=160,
                overall_p=0.48,
            )
        self.dataset.augmentations = build_augment(self.noise_path, self.ir_path, **cfg)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.add_callback(AugSchedule(train_dataset, NOISE_PATH, IR_PATH))

trainer.train()
