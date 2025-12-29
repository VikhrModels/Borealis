from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor, PreTrainedTokenizer
import datasets
import torch
import numpy as np
import random


def is_valid_audio(audio_data):
    """Check if audio data is valid and non-empty."""
    if audio_data is None:
        return False
    try:
        if hasattr(audio_data, 'get_all_samples'):
            data = audio_data.get_all_samples().data
            return data is not None and data.numel() > 0
        elif isinstance(audio_data, dict) and 'array' in audio_data:
            arr = audio_data['array']
            return arr is not None and len(arr) > 0
        elif isinstance(audio_data, (np.ndarray, torch.Tensor)):
            return len(audio_data) > 0
        return False
    except Exception:
        return False


def is_valid_text(text):
    """Check if text is valid and non-empty."""
    if text is None:
        return False
    if isinstance(text, str):
        return len(text.strip()) > 0
    return False


class BorealisInstructDataset(Dataset):
    """Dataset for instruction-tuning with audio input."""

    MIN_AUDIO_SAMPLES = 1600  # 0.1 sec at 16kHz
    MAX_AUDIO_SAMPLES = 30 * 16000  # 30 sec at 16kHz

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        feature_extractor: WhisperFeatureExtractor,
        max_audio_len: int = 30,
        max_text_len: int = 1024,
        sampling_rate: int = 16_000,
        augmentations=None,
        default_system_prompt: str = "You are a helpful voice assistant. Listen to the audio and respond appropriately.",
        question_column: str = "question",
        answer_column: str = "answer",
        audio_column: str = "audio",
        system_column: str = None,
        static_question: str = None,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sr = sampling_rate
        self.real_max_len = int(max_audio_len * sampling_rate)
        self.text_max_len = max_text_len
        self.augmentations = augmentations
        self.default_system_prompt = default_system_prompt
        self.question_column = question_column
        self.answer_column = answer_column
        self.audio_column = audio_column
        self.system_column = system_column
        self.static_question = static_question

        # Stats tracking
        self._skip_count = 0
        self._success_count = 0
        self._skip_reasons = {}
        self._log_interval = 1000

    def __len__(self):
        return len(self.dataset)

    def _log_skip(self, reason):
        """Track skip reason and log periodically."""
        self._skip_count += 1
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1

        # Log every N skips
        if self._skip_count % self._log_interval == 0:
            total = self._skip_count + self._success_count
            skip_rate = (self._skip_count / total * 100) if total > 0 else 0
            print(f"[Dataset Stats] Processed: {total}, Skipped: {self._skip_count} ({skip_rate:.2f}%), Success: {self._success_count}")
            print(f"[Dataset Stats] Skip reasons: {dict(self._skip_reasons)}")

    def _log_success(self):
        """Track successful sample."""
        self._success_count += 1

        # Log every N successes
        if self._success_count % (self._log_interval * 10) == 0:
            total = self._skip_count + self._success_count
            skip_rate = (self._skip_count / total * 100) if total > 0 else 0
            print(f"[Dataset Stats] Processed: {total}, Skipped: {self._skip_count} ({skip_rate:.2f}%), Success: {self._success_count}")

    def _extract_audio(self, audio_data):
        """Extract audio tensor from various formats."""
        if hasattr(audio_data, 'get_all_samples'):
            audio_sample = audio_data.get_all_samples().data.squeeze()
        elif isinstance(audio_data, dict) and 'array' in audio_data:
            arr = audio_data['array']
            if isinstance(arr, np.ndarray):
                audio_sample = torch.from_numpy(arr).float()
            else:
                audio_sample = torch.tensor(arr).float()
        elif isinstance(audio_data, np.ndarray):
            audio_sample = torch.from_numpy(audio_data).float()
        else:
            audio_sample = torch.tensor(audio_data).float()

        # Ensure 1D
        while audio_sample.dim() > 1:
            audio_sample = audio_sample.mean(dim=0)

        return audio_sample

    def _validate_sample(self, audio_sample, question, answer):
        """Validate audio and text data."""
        # Check audio
        if audio_sample is None or audio_sample.numel() == 0:
            return False, "empty audio"

        if len(audio_sample) < self.MIN_AUDIO_SAMPLES:
            return False, f"audio too short ({len(audio_sample)} samples)"

        # Check for NaN/Inf in audio
        if torch.isnan(audio_sample).any() or torch.isinf(audio_sample).any():
            return False, "audio contains NaN or Inf"

        # Check audio is not silent (all zeros or near-zero)
        if audio_sample.abs().max() < 1e-6:
            return False, "audio is silent"

        # Check text
        if not is_valid_text(question):
            return False, "invalid question"

        if not is_valid_text(answer):
            return False, "invalid answer"

        return True, None

    def __getitem__(self, index):
        max_attempts = 20

        for attempt in range(max_attempts):
            try:
                actual_index = (index + attempt) % len(self.dataset)
                example = self.dataset[actual_index]

                # Check audio exists
                audio_data = example.get(self.audio_column)
                if not is_valid_audio(audio_data):
                    self._log_skip("invalid_audio")
                    continue

                # Extract audio
                audio_sample = self._extract_audio(audio_data)

                # Get text
                if self.static_question:
                    question = self.static_question
                else:
                    question = example.get(self.question_column, "")
                answer = example.get(self.answer_column, "")

                # Validate
                is_valid, reason = self._validate_sample(audio_sample, question, answer)
                if not is_valid:
                    self._log_skip(reason)
                    continue

                # Get system prompt
                if self.system_column and self.system_column in example:
                    system_prompt = example[self.system_column]
                    if not is_valid_text(system_prompt):
                        system_prompt = self.default_system_prompt
                else:
                    system_prompt = self.default_system_prompt


                # Build conversation - only add audio tags if not already present
                if "<|start_of_audio|>" in question:
                    user_content = question
                else:
                    user_content = f"{question}\n<|start_of_audio|><|end_of_audio|>"

                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                ]

                chat_text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                tokenized = self.tokenizer(
                    chat_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.text_max_len,
                    return_tensors="pt",
                    padding_side="right",
                )

                labels = tokenized.input_ids.squeeze(0)
                text_att_mask = tokenized.attention_mask.squeeze(0)

                # Validate tokenized output
                if labels.numel() == 0:
                    self._log_skip("empty_labels")
                    continue

                # Check not all padding
                non_pad_count = (labels != self.tokenizer.pad_token_id).sum()
                if non_pad_count < 5:  # At least some real tokens
                    self._log_skip("all_padding")
                    continue

                # Process audio chunks
                chunks = []
                for i in range(0, len(audio_sample), self.real_max_len):
                    chunk = audio_sample[i : i + self.real_max_len]

                    if len(chunk) < 100:  # Skip tiny trailing chunks
                        continue

                    proc = self.feature_extractor(
                        chunk.numpy() if isinstance(chunk, torch.Tensor) else chunk,
                        sampling_rate=self.sr,
                        padding="max_length",
                        max_length=self.real_max_len,
                        truncation=True,
                        return_attention_mask=False,
                        return_tensors="pt",
                    )
                    mel = proc.input_features.squeeze(0)

                    # Validate mel
                    if mel.numel() == 0:
                        continue
                    if torch.isnan(mel).any() or torch.isinf(mel).any():
                        continue

                    chunks.append(mel)

                # Must have at least one valid chunk
                if len(chunks) == 0:
                    self._log_skip("no_valid_chunks")
                    continue

                self._log_success()
                return {
                    "mel": chunks,
                    "labels": labels,
                    "text_att_mask": text_att_mask,
                }

            except Exception as e:
                if self._skip_count < 10:  # Print first 10 errors
                    print(f"[Dataset Error] {type(e).__name__}: {e}")
                self._log_skip(f"exception:{type(e).__name__}")
                continue

        # Fallback: return a random valid sample
        self._log_skip("max_attempts_exceeded")
        return self.__getitem__(random.randint(0, len(self.dataset) - 1))


class BorealisTextOnlyDataset(Dataset):
    """Dataset for text-only instruction-tuning (no audio).

    Used to add text instruction data like GrandMaster-PRO-MAX to training.
    Returns dummy mel features to maintain batch compatibility with audio data.
    """

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        feature_extractor: WhisperFeatureExtractor,
        max_text_len: int = 1024,
        sampling_rate: int = 16_000,
        default_system_prompt: str = "You are a helpful assistant.",
        conversation_column: str = "conversation",
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sr = sampling_rate
        self.text_max_len = max_text_len
        self.default_system_prompt = default_system_prompt
        self.conversation_column = conversation_column

        # Create dummy mel for text-only samples (silent audio)
        # Whisper expects 30s * 16kHz = 480000 samples -> 3000 frames
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        proc = self.feature_extractor(
            dummy_audio,
            sampling_rate=self.sr,
            padding="max_length",
            max_length=30 * self.sr,
            truncation=True,
            return_attention_mask=False,
            return_tensors="pt",
        )
        self.dummy_mel = proc.input_features.squeeze(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        max_attempts = 10

        for attempt in range(max_attempts):
            try:
                actual_index = (index + attempt) % len(self.dataset)
                example = self.dataset[actual_index]

                # Get conversation data
                conversation_data = example.get(self.conversation_column, [])

                if not conversation_data or len(conversation_data) < 2:
                    continue

                # Build conversation from the data
                # GrandMaster format: list of {"role": "...", "content": "..."}
                conversation = []

                # Add system prompt
                conversation.append({
                    "role": "system",
                    "content": self.default_system_prompt
                })

                # Add conversation turns
                for turn in conversation_data:
                    role = turn.get("role", "")
                    content = turn.get("content", "")

                    if role == "user" and is_valid_text(content):
                        conversation.append({"role": "user", "content": content})
                    elif role == "assistant" and is_valid_text(content):
                        conversation.append({"role": "assistant", "content": content})

                # Must have at least user + assistant
                if len(conversation) < 3:
                    continue

                chat_text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                tokenized = self.tokenizer(
                    chat_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.text_max_len,
                    return_tensors="pt",
                    padding_side="right",
                )

                labels = tokenized.input_ids.squeeze(0)
                text_att_mask = tokenized.attention_mask.squeeze(0)

                # Check not all padding
                non_pad_count = (labels != self.tokenizer.pad_token_id).sum()
                if non_pad_count < 10:
                    continue

                return {
                    "mel": [self.dummy_mel],  # Single dummy mel chunk
                    "labels": labels,
                    "text_att_mask": text_att_mask,
                    "is_text_only": True,  # Flag for potential special handling
                }

            except Exception as e:
                continue

        # Fallback
        return self.__getitem__(random.randint(0, len(self.dataset) - 1))


class BorealisClassificationDataset(Dataset):
    """Dataset for classification tasks with diverse instruction formats.

    Supports multiple output formats:
    - JSON format
    - Natural language
    - Short answer
    - Confidence-based answers
    """

    MIN_AUDIO_SAMPLES = 1600  # 0.1 sec at 16kHz

    # Emotion label mappings
    EMOTION_LABELS = {
        0: "angry",
        1: "disgusted",
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprised"
    }

    EMOTION_LABELS_RU = {
        0: "злость",
        1: "отвращение",
        2: "страх",
        3: "радость",
        4: "нейтральный",
        5: "грусть",
        6: "удивление"
    }

    # Diverse instruction templates
    INSTRUCTIONS_EN = [
        # JSON format
        ("Classify the emotion in this audio and respond in JSON format with keys 'emotion' and 'confidence'.",
         lambda label: f'{{"emotion": "{label}", "confidence": "high"}}'),
        ("Analyze the speaker's emotion. Output: {\"emotion\": \"<emotion>\", \"intensity\": \"<low/medium/high>\"}",
         lambda label: f'{{"emotion": "{label}", "intensity": "medium"}}'),
        ("Detect the emotional state in the audio. Return JSON: {\"detected_emotion\": \"...\", \"alternative\": \"...\"}",
         lambda label: f'{{"detected_emotion": "{label}", "alternative": "neutral"}}'),

        # Natural language - detailed
        ("Listen to the audio and describe what emotion the speaker is expressing. Explain your reasoning.",
         lambda label: f"The speaker is expressing {label}. This can be detected from the tone of voice, speech patterns, and vocal characteristics that are typical of someone feeling {label}."),
        ("What emotional state is conveyed in this audio recording? Provide a brief analysis.",
         lambda label: f"The audio conveys a {label} emotional state. The vocal cues such as pitch, tempo, and intensity indicate that the speaker is {label}."),
        ("Analyze the emotional content of this speech. What is the speaker feeling?",
         lambda label: f"Based on the audio analysis, the speaker appears to be feeling {label}. The prosodic features and voice quality are consistent with this emotion."),

        # Short answers
        ("What emotion is in the audio?",
         lambda label: label.capitalize()),
        ("Classify the emotion:",
         lambda label: label),
        ("Detected emotion?",
         lambda label: label.capitalize()),
        ("Speaker's emotion:",
         lambda label: label.capitalize()),

        # Question format
        ("Is the speaker happy, sad, angry, or neutral? What emotion do you detect?",
         lambda label: f"The speaker is {label}."),
        ("Can you identify the emotion in this voice recording?",
         lambda label: f"Yes, the emotion is {label}."),
        ("What feeling is the person expressing in this audio clip?",
         lambda label: f"The person is expressing {label}."),

        # Formal/technical
        ("Perform emotion recognition on the audio input. Classification result:",
         lambda label: f"Emotion classification: {label.upper()}"),
        ("Execute sentiment analysis on the speech signal. Output the detected emotion category.",
         lambda label: f"Detected category: {label}"),
    ]

    INSTRUCTIONS_RU = [
        # JSON формат
        ("Классифицируй эмоцию в этом аудио и ответь в формате JSON с ключами 'emotion' и 'confidence'.",
         lambda label: f'{{"emotion": "{label}", "confidence": "high"}}'),
        ("Определи эмоциональное состояние говорящего. Формат: {{\"эмоция\": \"...\", \"уверенность\": \"...\"}}",
         lambda label: f'{{"эмоция": "{label}", "уверенность": "высокая"}}'),

        # Natural language
        ("Прослушай аудио и опиши, какую эмоцию выражает говорящий.",
         lambda label: f"Говорящий выражает {label}. Это определяется по тону голоса и речевым паттернам."),
        ("Какое эмоциональное состояние передаётся в этой аудиозаписи?",
         lambda label: f"В аудио передаётся эмоция: {label}."),
        ("Проанализируй эмоциональное содержание речи. Что чувствует говорящий?",
         lambda label: f"Судя по анализу аудио, говорящий испытывает {label}."),

        # Короткие ответы
        ("Какая эмоция в аудио?",
         lambda label: label.capitalize()),
        ("Определи эмоцию:",
         lambda label: label),
        ("Эмоция говорящего:",
         lambda label: label.capitalize()),

        # Вопросы
        ("Говорящий счастлив, грустен, зол или нейтрален?",
         lambda label: f"Говорящий испытывает {label}."),
        ("Можешь определить эмоцию в этой голосовой записи?",
         lambda label: f"Да, эмоция: {label}."),
    ]

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        feature_extractor: WhisperFeatureExtractor,
        max_audio_len: int = 30,
        max_text_len: int = 1024,
        sampling_rate: int = 16_000,
        augmentations=None,
        default_system_prompt: str = "You are an emotion recognition assistant. Analyze the audio and identify the emotional state.",
        audio_column: str = "audio",
        label_column: str = "label",
        label_mapping: dict = None,
        use_russian: bool = True,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sr = sampling_rate
        self.real_max_len = int(max_audio_len * sampling_rate)
        self.text_max_len = max_text_len
        self.augmentations = augmentations
        self.default_system_prompt = default_system_prompt
        self.audio_column = audio_column
        self.label_column = label_column
        self.label_mapping = label_mapping or self.EMOTION_LABELS
        self.label_mapping_ru = self.EMOTION_LABELS_RU
        self.use_russian = use_russian

        # Combine instructions
        self.instructions = self.INSTRUCTIONS_EN.copy()
        if use_russian:
            self.instructions.extend(self.INSTRUCTIONS_RU)

        # Stats
        self._skip_count = 0
        self._success_count = 0

    def __len__(self):
        return len(self.dataset)

    def _extract_audio(self, audio_data):
        """Extract audio tensor from various formats."""
        if hasattr(audio_data, 'get_all_samples'):
            audio_sample = audio_data.get_all_samples().data.squeeze()
        elif isinstance(audio_data, dict) and 'array' in audio_data:
            arr = audio_data['array']
            if isinstance(arr, np.ndarray):
                audio_sample = torch.from_numpy(arr).float()
            else:
                audio_sample = torch.tensor(arr).float()
        elif isinstance(audio_data, np.ndarray):
            audio_sample = torch.from_numpy(audio_data).float()
        else:
            audio_sample = torch.tensor(audio_data).float()

        while audio_sample.dim() > 1:
            audio_sample = audio_sample.mean(dim=0)

        return audio_sample

    def __getitem__(self, index):
        max_attempts = 20

        for attempt in range(max_attempts):
            try:
                actual_index = (index + attempt) % len(self.dataset)
                example = self.dataset[actual_index]

                # Get audio
                audio_data = example.get(self.audio_column)
                if not is_valid_audio(audio_data):
                    continue

                audio_sample = self._extract_audio(audio_data)

                if audio_sample.numel() == 0 or len(audio_sample) < self.MIN_AUDIO_SAMPLES:
                    continue

                # Get label
                label_idx = example.get(self.label_column)
                if label_idx is None:
                    continue

                # Select random instruction template
                instruction_template, answer_func = random.choice(self.instructions)

                # Determine if Russian instruction
                is_russian = instruction_template in [t[0] for t in self.INSTRUCTIONS_RU]

                # Get label text
                if is_russian:
                    label_text = self.label_mapping_ru.get(label_idx, str(label_idx))
                else:
                    label_text = self.label_mapping.get(label_idx, str(label_idx))

                # Generate answer
                answer = answer_func(label_text)

                # Build conversation
                user_content = f"{instruction_template}\n<|start_of_audio|><|end_of_audio|>"

                conversation = [
                    {"role": "system", "content": self.default_system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                ]

                chat_text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                tokenized = self.tokenizer(
                    chat_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.text_max_len,
                    return_tensors="pt",
                    padding_side="right",
                )

                labels = tokenized.input_ids.squeeze(0)
                text_att_mask = tokenized.attention_mask.squeeze(0)

                if labels.numel() == 0:
                    continue

                # Process audio chunks
                chunks = []
                for i in range(0, len(audio_sample), self.real_max_len):
                    chunk = audio_sample[i : i + self.real_max_len]

                    if len(chunk) < 100:
                        continue

                    proc = self.feature_extractor(
                        chunk.numpy() if isinstance(chunk, torch.Tensor) else chunk,
                        sampling_rate=self.sr,
                        padding="max_length",
                        max_length=self.real_max_len,
                        truncation=True,
                        return_attention_mask=False,
                        return_tensors="pt",
                    )
                    mel = proc.input_features.squeeze(0)

                    if mel.numel() == 0 or torch.isnan(mel).any():
                        continue

                    chunks.append(mel)

                if len(chunks) == 0:
                    continue

                self._success_count += 1
                return {
                    "mel": chunks,
                    "labels": labels,
                    "text_att_mask": text_att_mask,
                }

            except Exception as e:
                self._skip_count += 1
                continue

        return self.__getitem__(random.randint(0, len(self.dataset) - 1))


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
