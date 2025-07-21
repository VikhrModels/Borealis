from typing import List, Dict
import torch


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
