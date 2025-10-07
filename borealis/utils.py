from typing import List, Dict
from torch import default_collate
import torch
from datasets import Dataset


class AudioCollator:
    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return default_collate(features)


def clean_dataset(
    ds: Dataset,
    audio_column: str = "audio",
    text_column: str = "text",
    num_proc: int = 20,
    min_sec: float = 0.079,
    sr: int = 16_000,
) -> Dataset:
    MIN_SAMPLES = int(min_sec * sr)

    len_before = len(ds)

    ds = ds.filter(
        lambda example: (
            (a := example.get(audio_column, None)) is not None
            and (arr := a.get("array", None)) is not None
            and arr.size != 0
            and arr.shape[0] >= MIN_SAMPLES
            and (text := example.get(text_column, None)) is not None
            and len(text.strip()) != 0
        ),
        num_proc=num_proc,
    )

    len_after = len(ds)
    print(f"Filtered {len_before - len_after} / {len_before} examples")

    return ds
