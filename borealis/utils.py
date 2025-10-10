from typing import List, Dict
import torch
from datasets import Dataset


class AudioCollator:
    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        mels = [item["mel"] for item in features]

        labels = torch.stack([item["labels"] for item in features])
        text_att_masks = torch.stack([item["text_att_mask"] for item in features])

        return {
            "mel": mels,
            "labels": labels,
            "text_att_mask": text_att_masks,
        }


def is_valid_audio(a):
    if a is None:
        return False
    try:
        return a.get_all_samples().data.shape[0] > 0
    except Exception:
        return False


def clean_dataset(
    ds: Dataset,
    audio_column: str = "audio",
    text_column: str = "text",
    num_proc: int = 42,
) -> Dataset:
    len_before = len(ds)

    ds = ds.filter(
        lambda batch: [
            (t is not None and len(t.strip()) > 0) and is_valid_audio(a)
            for t, a in zip(batch[text_column], batch[audio_column])
        ],
        batched=True,
        batch_size=7000,
        num_proc=num_proc,
    )

    len_after = len(ds)
    print(f"Удалено {len_before - len_after} примеров из {len_before}")

    return ds
