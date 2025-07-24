from torch.utils.data import Dataset
import torch
from transformers import WhisperFeatureExtractor, PreTrainedTokenizer
from typing import List, Dict, Any, Optional


class BorealisBaseDataset(Dataset):
    def __init__(
        self,
        audio_processor: WhisperFeatureExtractor,
        text_tokenizer: PreTrainedTokenizer,
        audios: List[Dict[str, Any]],
        texts: List[str],
        max_seconds_len: float = 30.0,
        sampling_rate: int = 16_000,
        max_text_len: int = 512,
    ):
        super().__init__()
        assert len(audios) == len(texts), "Число аудио и текстов должно совпадать"

        self.real_max_len = int(max_seconds_len * sampling_rate)
        self.sr = sampling_rate
        self.text_max_len = max_text_len

        self.audio_processor = audio_processor
        self.tokenizer = text_tokenizer

        self.audios = audios
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index):
        audio_sample = self.audios[index]["array"]
        text_sample = self.texts[index]

        proc = self.audio_processor(
            audio_sample,
            sampling_rate=self.sr,
            padding="max_length",
            max_length=self.real_max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )

        tokenized = self.tokenizer(
            text_sample + "<|im_end|>",
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
