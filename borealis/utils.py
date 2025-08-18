from typing import List, Dict
import torch
import numpy as np


class AudioCollator:
    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return {
            "mel": torch.stack([f["mel"] for f in features]),  # [B, 128, T]
            "audio_att_mask": torch.stack(
                [f["audio_att_mask"] for f in features]
            ),  # [B, T]
            "labels": torch.stack([f["labels"] for f in features]),  # [B, L]
            "text_att_mask": torch.stack(
                [f["text_att_mask"] for f in features]
            ),  # [B, L]
        }


MIN_SEC = 0.10
SR = 16_000


def _has_valid_audio(example):
    a = example.get("audio", None)
    if a is None:
        return False
    arr = a.get("array", None)
    if arr is None:
        return False
    if not isinstance(arr, np.ndarray):
        return False
    if arr.size == 0:
        return False
    if np.isnan(arr).any() or np.isinf(arr).any():
        return False
    return arr.shape[0] >= int(MIN_SEC * SR)


def _filter_and_report(ds, name: str, num_proc: int = 20):
    before = len(ds)
    ds = ds.filter(_has_valid_audio, num_proc=num_proc)
    after = len(ds)
    print(f"[filter] {name}: {before} -> {after} (removed {before - after})")
    return ds
