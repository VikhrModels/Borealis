from typing import List, Dict
import torch
from datasets import Dataset, load_dataset, Audio
import re


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


def load_and_process_dataset(
    name,
    config_name,
    target_split,
    columns,
    num_proc,
    sampling_rate,
    select_range=None,
    rename_text=None,
    rename_audio=None,
    filter_locale_ru=False,
):
    if config_name:
        ds_dict = load_dataset(name, config_name, columns=columns, num_proc=num_proc)
    else:
        ds_dict = load_dataset(name, columns=columns, num_proc=num_proc)
    ds = ds_dict[target_split]

    if rename_text:
        if rename_text in ds.column_names:
            ds = ds.rename_column(rename_text, "text")
        else:
            print(
                f"Warning: Text column '{rename_text}' not found in dataset '{name}'. Skipping rename."
            )

    if rename_audio:
        if rename_audio in ds.column_names:
            ds = ds.rename_column(rename_audio, "audio")
        else:
            print(
                f"Warning: Audio column '{rename_audio}' not found in dataset '{name}'. Skipping rename."
            )

    if filter_locale_ru:
        ds = ds.filter(
            lambda ex: ex.get("locale") is not None
            and "ru" in str(ex["locale"]).lower(),
            num_proc=20,
        )

    if select_range is not None:
        ds = ds.select(range(select_range))

    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    return ds


def convert_numeric_strings(obj):
    """Рекурсивно конвертирует строки в int/float, если они выглядят как числа."""
    if isinstance(obj, dict):
        return {k: convert_numeric_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numeric_strings(item) for item in obj]
    elif isinstance(obj, str):
        # Проверяем на int (целые числа, включая отрицательные)
        if re.match(r"^-?\d+$", obj):
            return int(obj)
        # Проверяем на float (включая научную нотацию вроде 3e-4, 1.23, -1.0e+2)
        elif re.match(r"^-?\d*\.?\d+(?:[eE][+-]?\d+)?$", obj):
            return float(obj)
    return obj
