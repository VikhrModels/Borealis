from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor, PreTrainedTokenizer
import datasets


class BorealisPretrainDataset(Dataset):
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        feature_extractor: WhisperFeatureExtractor,
        max_audio_len: int = 30,
        max_text_len: int = 512,
        sampling_rate: int = 16_000,
        augmentations=None,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sr = sampling_rate
        self.real_max_len = int(max_audio_len * sampling_rate)
        self.text_max_len = max_text_len
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        audio_sample = example["audio"].get_all_samples().data.squeeze()

        while audio_sample.dim() > 1:
            audio_sample = audio_sample.mean(dim=0)

        text_sample = example["text"]

        if self.augmentations:
            audio_sample = self.augmentations(
                waveform=audio_sample, sample_rate=self.sr
            )

        conversation = [
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
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        tokenized = self.tokenizer(
            chat_text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_len,
            return_tensors="pt",
            padding_side="right",
        )

        chunks = []
        for i in range(0, len(audio_sample), self.real_max_len):
            chunk = audio_sample[i : i + self.real_max_len]
            proc = self.feature_extractor(
                chunk,
                sampling_rate=self.sr,
                padding="max_length",
                max_length=self.real_max_len,
                truncation=True,
                return_attention_mask=False,
                return_tensors="pt",
            )
            mel = proc.input_features.squeeze(0)
            if self.augmentations:
                mel = self.augmentations.apply_spec(mel)
            chunks.append(mel)

        return {
            "mel": chunks,
            "labels": tokenized.input_ids.squeeze(0),
            "text_att_mask": tokenized.attention_mask.squeeze(0),
        }
