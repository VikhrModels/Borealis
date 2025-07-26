from torch.utils.data import Dataset
import torch
from transformers import WhisperFeatureExtractor, PreTrainedTokenizer
from typing import List, Dict, Any, Optional


class BorealisBaseDataset(Dataset):
    def __init__(
        self,
        audio_processor: WhisperFeatureExtractor,
        text_tokenizer: PreTrainedTokenizer,
        hf_ds,  
        max_seconds_len: float = 30.0,
        sampling_rate: int = 16_000,
        max_text_len: int = 512,
    ):
        super().__init__()
        self.real_max_len = int(max_seconds_len * sampling_rate)
        self.sr = sampling_rate
        self.text_max_len = max_text_len

        self.audio_processor = audio_processor
        self.tokenizer = text_tokenizer

        self.hf_ds = hf_ds

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, index):
        example = self.hf_ds[index]
        audio_sample = example["audio"]["array"]
        text_sample = example["text"]

        proc = self.audio_processor(
            audio_sample,
            sampling_rate=self.sr,
            padding="max_length",
            max_length=self.real_max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        tokenized = self.tokenizer(
            text_sample + self.tokenizer.eos_token,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_len,
            return_tensors="pt",
        )

        return {
            "mel": proc.input_features.squeeze(0),
            "audio_att_mask": proc.attention_mask.squeeze(0),
            "labels": tokenized.input_ids.squeeze(0),
            "text_att_mask": tokenized.attention_mask.squeeze(0),
        }


class BorealisInstrustDataset(Dataset):
    pass
