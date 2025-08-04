from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor, PreTrainedTokenizer


class BorealisBaseDataset(Dataset):
    def __init__(
        self,
        audio_processor: WhisperFeatureExtractor,
        text_tokenizer: PreTrainedTokenizer,
        hf_ds,
        max_seconds_len: float = 30.0,
        sampling_rate: int = 16_000,
        max_text_len: int = 512,
        augmentations=None,
    ):
        super().__init__()
        self.real_max_len = int(max_seconds_len * sampling_rate)
        self.sr = sampling_rate
        self.text_max_len = max_text_len

        self.audio_processor = audio_processor
        self.tokenizer = text_tokenizer

        self.hf_ds = hf_ds
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, index):
        example = self.hf_ds[index]
        audio_sample = example["audio"]["array"]
        text_sample = example["text"]

        if self.augmentations:
            audio_sample = self.augmentations(samples=audio_sample, sample_rate=self.sr)

        proc = self.audio_processor(
            audio_sample,
            sampling_rate=self.sr,
            padding="max_length",
            max_length=self.real_max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        messages = [
            {
                "role": "system",
                "content": "Вы полезный помощник по автоматическому распознаванию речи. Точно транскрибируйте аудио в текст.",
            },
            {
                "role": "user",
                "content": "Транскрибируйте это аудио: <|start_of_audio|><|end_of_audio|>",
            },
            {"role": "assistant", "content": text_sample},
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        tokenized = self.tokenizer(
            chat_text,
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