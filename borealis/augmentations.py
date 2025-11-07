from __future__ import annotations
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as Fnn
from torchaudio import functional as AF
from torchaudio.transforms import Resample, FrequencyMasking, TimeMasking
from transformers import TrainerCallback
from datasets import Dataset

__all__ = [
    "AugmentConfig",
    "AugmentationStage",
    "AugmentationPipeline",
    "AugmentationScheduler",
    "build_augmentation_pipeline",
    "default_augmentation_stages",
]


@dataclass(frozen=True)
class AugmentConfig:
    overall_p: float
    snr_range: Tuple[float, float] = (18.0, 28.0)
    p_background_noise: float = 0.0
    p_multi_noise: float = 0.0
    max_extra_noises: int = 1
    p_gaussian_noise: float = 0.0
    gaussian_std_range: Tuple[float, float] = (0.001, 0.01)
    p_ir: float = 0.0
    p_multi_ir: float = 0.0
    max_extra_irs: int = 1
    p_eq: float = 0.0
    eq_gain_range: Tuple[float, float] = (-6.0, 6.0)
    p_random_gain: float = 0.0
    gain_range: Tuple[float, float] = (-3.0, 3.0)
    p_bandpass: float = 0.0
    bandpass_low_range: Tuple[float, float] = (150.0, 350.0)
    bandpass_high_range: Tuple[float, float] = (3200.0, 5200.0)
    p_resample: float = 0.0
    resample_range: Tuple[int, int] = (14000, 20000)
    p_telephony: float = 0.0
    telephony_sr_range: Tuple[int, int] = (8000, 12000)
    telephony_lowpass_range: Tuple[float, float] = (3200.0, 4200.0)
    telephony_highpass_range: Tuple[float, float] = (180.0, 320.0)
    p_codec: float = 0.0
    codec_bitrate_range: Tuple[int, int] = (96, 160)
    p_clipping: float = 0.0
    clipping_range: Tuple[float, float] = (0.82, 0.95)
    p_pitch_shift: float = 0.0
    pitch_steps_range: Tuple[float, float] = (-4.0, 4.0)
    p_speed: float = 0.0
    speed_factor_range: Tuple[float, float] = (0.8, 1.2)
    p_spec: float = 0.0
    num_freq_masks: Tuple[int, int] = (0, 2)
    freq_mask_max: int = 27
    num_time_masks: Tuple[int, int] = (0, 2)
    time_mask_max: int = 100


@dataclass(frozen=True)
class AugmentationStage:
    start_epoch: int
    config: AugmentConfig
    description: str = ""


def _load_audio_bank(hf_dataset: Dataset) -> List[Tuple[torch.Tensor, int]]:
    if len(hf_dataset) == 0:
        warnings.warn("[Augmentations] Audio dataset is empty.")
        return []
    bank: List[Tuple[torch.Tensor, int]] = []
    for item in hf_dataset:
        try:
            waveform, sr = (
                item["audio"].get_all_samples().data.squeeze(),
                item["audio"].get_all_samples().sample_rate,
            )
            bank.append((waveform.to(torch.float32), sr))
        except (RuntimeError, OSError, KeyError):
            warnings.warn(
                "[Augmentations] Failed to load audio from dataset item. Skipping."
            )
    return bank


class AugmentationPipeline:
    def __init__(
        self,
        sample_rate: int,
        config: AugmentConfig,
        noise_bank: Optional[List[Tuple[torch.Tensor, int]]] = None,
        ir_bank: Optional[List[Tuple[torch.Tensor, int]]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.config = config
        self.noise_bank = noise_bank or []
        self.ir_bank = ir_bank or []
        self.eq_freqs = [60.0, 150.0, 400.0, 1000.0, 2400.0, 6000.0, 15000.0]
        self._resampler_cache: Dict[Tuple[int, int], Resample] = {}

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.config.overall_p <= 0.0:
            return self._normalize(waveform, sample_rate)
        if sample_rate != self.sample_rate:
            waveform = self._resample_tensor(waveform, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate
        if random.random() > self.config.overall_p:
            return self._normalize(waveform, sample_rate)

        original_length = waveform.shape[-1]

        aug_methods = []
        if self._should(self.config.p_background_noise) and self.noise_bank:
            aug_methods.append(self._apply_background_noise)
        if self._should(self.config.p_gaussian_noise):
            aug_methods.append(self._apply_gaussian_noise)
        if self._should(self.config.p_ir) and self.ir_bank:
            aug_methods.append(self._apply_impulse_response)
        if self._should(self.config.p_eq):
            aug_methods.append(self._apply_eq)
        if self._should(self.config.p_random_gain):
            aug_methods.append(self._apply_random_gain)
        if self._should(self.config.p_bandpass):
            aug_methods.append(self._apply_bandpass)
        if self._should(self.config.p_resample):
            aug_methods.append(self._apply_resample_cycle)
        if self._should(self.config.p_telephony):
            aug_methods.append(self._apply_telephony)
        if self._should(self.config.p_codec):
            aug_methods.append(self._apply_codec)
        if self._should(self.config.p_clipping):
            aug_methods.append(self._apply_clipping)
        if self._should(self.config.p_pitch_shift):
            aug_methods.append(self._apply_pitch_shift)
        if self._should(self.config.p_speed):
            aug_methods.append(self._apply_speed)

        random.shuffle(aug_methods)

        for method in aug_methods:
            if method in [self._apply_background_noise, self._apply_impulse_response]:
                waveform = method(waveform, original_length)
            elif method in [
                self._apply_eq,
                self._apply_bandpass,
                self._apply_resample_cycle,
                self._apply_telephony,
                self._apply_codec,
                self._apply_pitch_shift,
                self._apply_speed,
            ]:
                waveform = method(waveform, sample_rate)
            else:
                waveform = method(waveform)

        waveform = waveform[..., :original_length]
        return self._normalize(waveform, sample_rate)

    def apply_spec(self, mel: torch.Tensor) -> torch.Tensor:
        if not self._should(self.config.p_spec):
            return mel
        num_f = random.randint(*self.config.num_freq_masks)
        for _ in range(num_f):
            mask_param = random.randint(0, self.config.freq_mask_max)
            mel = FrequencyMasking(mask_param)(mel.unsqueeze(0)).squeeze(0)
        num_t = random.randint(*self.config.num_time_masks)
        for _ in range(num_t):
            mask_param = random.randint(0, self.config.time_mask_max)
            mel = TimeMasking(mask_param)(mel.unsqueeze(0)).squeeze(0)
        return mel

    def _should(self, probability: float) -> bool:
        return probability > 0.0 and random.random() < probability

    def _uniform(self, bounds: Tuple[float, float]) -> float:
        low, high = bounds
        return random.uniform(low, high)

    def _uniform_int(self, bounds: Tuple[int, int]) -> int:
        low, high = bounds
        return random.randint(low, high)

    def _resample_tensor(
        self, waveform: torch.Tensor, src_sr: int, dst_sr: int
    ) -> torch.Tensor:
        if src_sr == dst_sr:
            return waveform
        key = (src_sr, dst_sr)
        if key not in self._resampler_cache:
            self._resampler_cache[key] = Resample(orig_freq=src_sr, new_freq=dst_sr)
        return self._resampler_cache[key](waveform)

    def _match_length(self, tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        current_length = tensor.shape[-1]
        if current_length == target_length:
            return tensor
        if current_length > target_length:
            start = random.randint(0, current_length - target_length)
            return tensor[..., start : start + target_length]
        pad_amount = target_length - current_length
        return Fnn.pad(tensor, (0, pad_amount), mode="constant", value=0.0)

    def _apply_background_noise(
        self, waveform: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        def add_noise(wave, noise, snr_db):
            signal_power = wave.pow(2).mean()
            noise_power = noise.pow(2).mean().clamp(min=1e-8)
            desired_noise_power = signal_power / (10 ** (snr_db / 10))
            scale = torch.sqrt(desired_noise_power / noise_power)
            return wave + scale * noise

        snr_db = self._uniform(self.config.snr_range)
        noise_waveform, noise_sr = random.choice(self.noise_bank)
        noise_waveform = self._resample_tensor(
            noise_waveform, noise_sr, self.sample_rate
        )
        noise_waveform = self._match_length(noise_waveform, target_length)
        waveform = add_noise(waveform, noise_waveform, snr_db)

        extra_noises = (
            random.randint(0, self.config.max_extra_noises)
            if self._should(self.config.p_multi_noise)
            else 0
        )
        for _ in range(extra_noises):
            snr_extra = self._uniform(self.config.snr_range)
            noise_extra, noise_sr_extra = random.choice(self.noise_bank)
            noise_extra = self._resample_tensor(
                noise_extra, noise_sr_extra, self.sample_rate
            )
            noise_extra = self._match_length(noise_extra, target_length)
            waveform = add_noise(waveform, noise_extra, snr_extra)

        return waveform

    def _apply_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        std = self._uniform(self.config.gaussian_std_range)
        return waveform + torch.randn_like(waveform) * std

    def _apply_impulse_response(
        self, waveform: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        def apply_ir(wave, ir):
            ir = ir / ir.norm(p=2).clamp(min=1e-6)
            convolved = AF.fftconvolve(wave, ir, mode="full")
            return convolved[..., :target_length]

        ir_waveform, ir_sr = random.choice(self.ir_bank)
        ir_waveform = self._resample_tensor(ir_waveform, ir_sr, self.sample_rate)
        waveform = apply_ir(waveform, ir_waveform)

        extra_irs = (
            random.randint(0, self.config.max_extra_irs)
            if self._should(self.config.p_multi_ir)
            else 0
        )
        for _ in range(extra_irs):
            ir_extra, ir_sr_extra = random.choice(self.ir_bank)
            ir_extra = self._resample_tensor(ir_extra, ir_sr_extra, self.sample_rate)
            waveform = apply_ir(waveform, ir_extra)

        return waveform

    def _apply_eq(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        for freq in self.eq_freqs:
            gain = self._uniform(self.config.eq_gain_range)
            waveform = AF.equalizer_biquad(
                waveform, sample_rate=sample_rate, center_freq=freq, gain=gain, Q=0.707
            )
        return waveform

    def _apply_random_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        gain_db = self._uniform(self.config.gain_range)
        return AF.gain(waveform, gain_db)

    def _apply_bandpass(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        low_cut = self._uniform(self.config.bandpass_low_range)
        high_cut = self._uniform(self.config.bandpass_high_range)
        low_cut = max(30.0, min(low_cut, high_cut - 100.0))
        high_cut = min(sample_rate / 2 - 10.0, max(high_cut, low_cut + 100.0))
        waveform = AF.highpass_biquad(waveform, sample_rate, low_cut)
        waveform = AF.lowpass_biquad(waveform, sample_rate, high_cut)
        return waveform

    def _apply_resample_cycle(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        target_sr = max(
            6000, min(self._uniform_int(self.config.resample_range), sample_rate)
        )
        if target_sr == sample_rate:
            return waveform
        down = self._resample_tensor(waveform, sample_rate, target_sr)
        return self._resample_tensor(down, target_sr, sample_rate)

    def _apply_telephony(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        target_sr = max(
            6000, min(self._uniform_int(self.config.telephony_sr_range), sample_rate)
        )
        down = self._resample_tensor(waveform, sample_rate, target_sr)
        up = self._resample_tensor(down, target_sr, sample_rate)
        hp_cut = self._uniform(self.config.telephony_highpass_range)
        lp_cut = self._uniform(self.config.telephony_lowpass_range)
        up = AF.highpass_biquad(up, sample_rate, hp_cut)
        up = AF.lowpass_biquad(up, sample_rate, lp_cut)
        return up

    def _apply_codec(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        bitrate = self._uniform_int(self.config.codec_bitrate_range)
        bit_rate_str = f"{max(32, bitrate)}k"
        return AF.apply_codec(
            waveform,
            sample_rate,
            format="mp3",
            bit_rate=bit_rate_str,
            channels_first=True,
        )

    def _apply_clipping(self, waveform: torch.Tensor) -> torch.Tensor:
        threshold = self._uniform(self.config.clipping_range)
        threshold = max(0.1, min(threshold, 0.99))
        clipped = waveform.clamp(min=-threshold, max=threshold)
        return clipped / threshold

    def _apply_pitch_shift(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        steps = self._uniform(self.config.pitch_steps_range)
        return AF.pitch_shift(waveform, sample_rate, steps)

    def _apply_speed(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        factor = self._uniform(self.config.speed_factor_range)
        return AF.speed(waveform, orig_freq=sample_rate, factor=factor)[0]

    def _normalize(self, waveform: torch.Tensor, _: int) -> torch.Tensor:
        rms = torch.sqrt(waveform.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        waveform = waveform / rms
        return waveform.clamp(-0.99, 0.99)


def build_augmentation_pipeline(
    noise_path: str,
    ir_path: str,
    config: AugmentConfig,
    sample_rate: int = 16_000,
    noise_bank: Optional[List[Tuple[torch.Tensor, int]]] = None,
    ir_bank: Optional[List[Tuple[torch.Tensor, int]]] = None,
) -> AugmentationPipeline:
    noise_bank = noise_bank or _load_audio_bank(Path(noise_path))
    ir_bank = ir_bank or _load_audio_bank(Path(ir_path))
    return AugmentationPipeline(
        sample_rate=sample_rate,
        config=config,
        noise_bank=noise_bank,
        ir_bank=ir_bank,
    )


def default_augmentation_stages(sample_rate: int = 16_000) -> List[AugmentationStage]:
    return [
        AugmentationStage(
            start_epoch=0,
            description="Тёплый старт без аугментаций",
            config=AugmentConfig(overall_p=0.0),
        ),
        AugmentationStage(
            start_epoch=1,
            description="Лёгкие шумы и эквализация",
            config=AugmentConfig(
                overall_p=0.4,
                snr_range=(20.0, 32.0),
                p_background_noise=0.55,
                p_multi_noise=0.3,
                max_extra_noises=1,
                p_gaussian_noise=0.25,
                gaussian_std_range=(0.001, 0.006),
                p_eq=0.3,
                eq_gain_range=(-5.0, 5.0),
                p_random_gain=0.4,
                gain_range=(-2.5, 2.5),
                p_bandpass=0.2,
                bandpass_low_range=(180.0, 320.0),
                bandpass_high_range=(3200.0, 5200.0),
                p_resample=0.15,
                resample_range=(int(sample_rate * 0.85), sample_rate),
                p_telephony=0.1,
                telephony_sr_range=(9000, 12000),
                telephony_lowpass_range=(3300.0, 4300.0),
                telephony_highpass_range=(160.0, 260.0),
                p_ir=0.2,
                p_multi_ir=0.1,
                max_extra_irs=1,
                p_codec=0.05,
                codec_bitrate_range=(112, 160),
                p_clipping=0.1,
                p_pitch_shift=0.2,
                pitch_steps_range=(-2.0, 2.0),
                p_speed=0.2,
                speed_factor_range=(0.9, 1.1),
                p_spec=0.2,
                num_freq_masks=(0, 1),
                freq_mask_max=15,
                num_time_masks=(0, 1),
                time_mask_max=50,
            ),
        ),
        AugmentationStage(
            start_epoch=3,
            description="Интенсивные шумы и телефония",
            config=AugmentConfig(
                overall_p=0.65,
                snr_range=(10.0, 25.0),
                p_background_noise=0.7,
                p_multi_noise=0.5,
                max_extra_noises=2,
                p_gaussian_noise=0.35,
                gaussian_std_range=(0.002, 0.015),
                p_eq=0.45,
                eq_gain_range=(-8.0, 8.0),
                p_random_gain=0.5,
                gain_range=(-4.0, 4.0),
                p_bandpass=0.35,
                bandpass_low_range=(150.0, 380.0),
                bandpass_high_range=(2800.0, 4600.0),
                p_resample=0.28,
                resample_range=(int(sample_rate * 0.65), int(sample_rate * 0.95)),
                p_telephony=0.35,
                telephony_sr_range=(7000, 11000),
                telephony_lowpass_range=(2900.0, 3800.0),
                telephony_highpass_range=(200.0, 320.0),
                p_ir=0.3,
                p_multi_ir=0.2,
                max_extra_irs=2,
                p_codec=0.2,
                codec_bitrate_range=(72, 144),
                p_clipping=0.2,
                p_pitch_shift=0.3,
                pitch_steps_range=(-3.0, 3.0),
                p_speed=0.3,
                speed_factor_range=(0.85, 1.15),
                p_spec=0.4,
                num_freq_masks=(0, 2),
                freq_mask_max=27,
                num_time_masks=(0, 2),
                time_mask_max=100,
            ),
        ),
        AugmentationStage(
            start_epoch=4,
            description="Фокус на телефонию и сильные искажения",
            config=AugmentConfig(
                overall_p=0.72,
                snr_range=(10.0, 25.0),
                p_background_noise=0.75,
                p_multi_noise=0.6,
                max_extra_noises=2,
                p_gaussian_noise=0.45,
                gaussian_std_range=(0.003, 0.015),
                p_eq=0.5,
                eq_gain_range=(-10.0, 10.0),
                p_random_gain=0.55,
                gain_range=(-5.0, 5.0),
                p_bandpass=0.4,
                bandpass_low_range=(120.0, 360.0),
                bandpass_high_range=(2400.0, 4200.0),
                p_resample=0.35,
                resample_range=(int(sample_rate * 0.55), int(sample_rate * 0.9)),
                p_telephony=0.5,
                telephony_sr_range=(6000, 9600),
                telephony_lowpass_range=(2500.0, 3600.0),
                telephony_highpass_range=(220.0, 360.0),
                p_ir=0.4,
                p_multi_ir=0.3,
                max_extra_irs=2,
                p_codec=0.35,
                codec_bitrate_range=(56, 128),
                p_clipping=0.3,
                p_pitch_shift=0.4,
                pitch_steps_range=(-4.0, 4.0),
                p_speed=0.4,
                speed_factor_range=(0.8, 1.2),
                p_spec=0.5,
                num_freq_masks=(1, 2),
                freq_mask_max=27,
                num_time_masks=(1, 2),
                time_mask_max=100,
            ),
        ),
    ]


class AugmentationScheduler(TrainerCallback):
    def __init__(
        self,
        dataset,
        noise_hf_set: Dataset,
        ir_hf_set: Dataset,
        stages: Optional[List[AugmentationStage]] = None,
        sample_rate: int = 16_000,
    ) -> None:
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.stages = sorted(
            stages or default_augmentation_stages(sample_rate),
            key=lambda s: s.start_epoch,
        )
        self.noise_bank = _load_audio_bank(noise_hf_set)
        self.ir_bank = _load_audio_bank(ir_hf_set)
        self._pipelines_cache: Dict[int, Optional[AugmentationPipeline]] = {}
        self._current_stage_idx: Optional[int] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._update_stage(0)

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        self._update_stage(epoch)

    def _update_stage(self, epoch: int) -> None:
        stage_idx = self._resolve_stage_index(epoch)
        if stage_idx == self._current_stage_idx:
            return
        stage = self.stages[stage_idx]
        pipeline = self._get_or_create_pipeline(stage)
        if stage.config.overall_p <= 0.0:
            self.dataset.augmentations = None
        else:
            self.dataset.augmentations = pipeline
        self._current_stage_idx = stage_idx
        description = f"[AugScheduler] epoch={epoch} → stage={stage_idx} : {stage.description or 'без описания'}"
        print(description)

    def _resolve_stage_index(self, epoch: int) -> int:
        idx = 0
        for i, stage in enumerate(self.stages):
            if epoch >= stage.start_epoch:
                idx = i
        return idx

    def _get_or_create_pipeline(
        self, stage: AugmentationStage
    ) -> Optional[AugmentationPipeline]:
        if stage.config.overall_p <= 0.0:
            return None
        key = stage.start_epoch
        if key not in self._pipelines_cache:
            self._pipelines_cache[key] = build_augmentation_pipeline(
                config=stage.config,
                sample_rate=self.sample_rate,
                noise_bank=self.noise_bank,
                ir_bank=self.ir_bank,
            )
        return self._pipelines_cache[key]
