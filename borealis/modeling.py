import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, AutoModelForCausalLM


class AudioLanguageAdapter(nn.Module):
    """Простой MLP‑адаптер для повышения размерности аудио‑эмбеддингов под hidden‑size LLM."""

    def __init__(self, hidden_size: int, dim: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(hidden_size, dim, bias=False)
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (⋯, hidden_size) → (⋯, dim)
        return self.w_out(self.gelu(self.w_in(x)))


class BorealisForConditionalGeneration(nn.Module):
    """
    Минимальный рабочий класс модели «аудио‑энкодер → адаптер → LLM».
    Логика сохранена, исправлены ошибки типов, масок и shape‑операций.
    """

    def __init__(
        self,
        whisper_encoder_name: str = "openai/whisper-large-v3",
        llm_name: str = "Qwen/Qwen2.5-0.5B",
        tokenizer=None,
        downsample_factor: int = 4,
    ):
        super().__init__()
        assert tokenizer is not None, "Tokenizer надо передать в модельку"

        # ─── Whisper‑энкодер (заморожен) ─────────────────────────────────────────
        self.encoder: WhisperModel = WhisperModel.from_pretrained(
            whisper_encoder_name
        ).encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ─── LLM ────────────────────────────────────────────────────────────────
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.tokenizer = tokenizer
        self.llm.resize_token_embeddings(len(tokenizer))

        # ─── Адаптер ────────────────────────────────────────────────────────────
        self.downsample_factor = downsample_factor
        self.adapter = AudioLanguageAdapter(
            hidden_size=self.encoder.config.d_model * downsample_factor,
            dim=self.llm.config.hidden_size,
        )

        # ─── ID спец‑токенов ────────────────────────────────────────────────────
        self.bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")

    # ════════════════════════════════════════════════════════════════════════════
    #                               UTILITIES
    # ════════════════════════════════════════════════════════════════════════════

    def _downsample(self, seq: torch.Tensor) -> torch.Tensor:
        """(T, d) → (ceil(T/k), d*k)"""
        k, (T, d) = self.downsample_factor, seq.shape
        target = k * math.ceil(T / k)
        if target != T:
            seq = F.pad(seq, (0, 0, 0, target - T))
        return seq.contiguous().view(target // k, d * k)

    def _tok_embed(self, tok_id: int, batch: int, device) -> torch.Tensor:
        idx = torch.full((batch, 1), tok_id, dtype=torch.long, device=device)
        return self.llm.get_input_embeddings()(idx)

    # ════════════════════════════════════════════════════════════════════════════
    #                                 FORWARD
    # ════════════════════════════════════════════════════════════════════════════

    def forward(
        self,
        mel: torch.Tensor,  # (B, 128, T)
        audio_att_mask: torch.Tensor,  # (B, T)
        labels: torch.Tensor,  # (B, L)
        text_att_mask: torch.Tensor,  # (B, L)
    ):
        B, device = mel.size(0), mel.device

        # 1. Whisper‑encoder
        enc_out = self.encoder(
            input_features=mel, attention_mask=None, return_dict=True
        ).last_hidden_state  # (B, T_enc, d)

        # 2. Down‑sample + adapter
        audio_embs, audio_mask, max_T = [], [], 0
        for seq in enc_out:  # (T_i, d)
            ds = self._downsample(seq)  # (T_ds_i, d*k)
            audio_embs.append(ds)
            max_T = max(max_T, ds.size(0))

        for ds in audio_embs:
            pad = max_T - ds.size(0)
            audio_mask.append(
                torch.cat(
                    [
                        torch.ones(ds.size(0), dtype=torch.long, device=device),
                        torch.zeros(pad, dtype=torch.long, device=device),
                    ]
                )
            )
            if pad:
                ds = F.pad(ds, (0, 0, 0, pad))
        audio_embeddings = torch.stack(audio_embs, 0)  # (B, max_T, d*k)
        audio_mask = torch.stack(audio_mask, 0)  # (B, max_T)
        audio_embeddings = self.adapter(audio_embeddings)  # (B, max_T, h)

        # 3. Текст
        text_embeddings = self.llm.get_input_embeddings()(labels)  # (B, L, h)

        # 4. Спец‑токены
        emb_bos, emb_sa, emb_ea = (
            self._tok_embed(t, B, device)
            for t in (self.bos_id, self.audio_start_id, self.audio_end_id)
        )

        # 5. Инпуты и маска
        inputs_embeds = torch.cat(
            [emb_bos, emb_sa, audio_embeddings, emb_ea, text_embeddings], dim=1
        )
        att_mask = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.long, device=device),  # BOS
                torch.ones(B, 1, dtype=torch.long, device=device),  # <audio>
                audio_mask,
                torch.ones(B, 1, dtype=torch.long, device=device),  # </audio>
                text_att_mask.long(),
            ],
            dim=1,
        )

        # 6. Loss labels
        prefix = 1 + 1 + max_T + 1
        ignore = labels.new_full((B, prefix), -100)
        loss_labels = torch.cat([ignore, labels], dim=1)
        if self.tokenizer.pad_token_id is not None:
            loss_labels[loss_labels == self.tokenizer.pad_token_id] = -100

        # 7. LLM
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=att_mask,
            labels=loss_labels,
            return_dict=True,
        )
        return out.loss, out.logits

    # ════════════════════════════════════════════════════════════════════════════
    #                                GENERATE
    # ════════════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,  # (B, 128, T) or (128, T)
        att_mask: torch.Tensor,  # (B, T)     or (T,)
        max_new_tokens: int = 512,
        **kwargs,
    ):
        single = mel.dim() == 2
        if single:
            mel, att_mask = mel.unsqueeze(0), att_mask.unsqueeze(0)

        B, device = mel.size(0), mel.device

        # 1. Encoder
        enc_out = self.encoder(
            input_features=mel, attention_mask=None, return_dict=True
        ).last_hidden_state

        # 2. Down‑sample + adapter
        audio_embs, audio_mask, max_T = [], [], 0
        for seq in enc_out:
            ds = self._downsample(seq)
            audio_embs.append(ds)
            max_T = max(max_T, ds.size(0))

        for i, ds in enumerate(audio_embs):
            pad = max_T - ds.size(0)
            audio_mask.append(
                torch.cat(
                    [
                        torch.ones(ds.size(0), dtype=torch.long, device=device),
                        torch.zeros(pad, dtype=torch.long, device=device),
                    ]
                )
            )
            if pad:
                audio_embs[i] = F.pad(ds, (0, 0, 0, pad))
        audio_embeddings = torch.stack(audio_embs, 0)
        audio_mask = torch.stack(audio_mask, 0)
        audio_embeddings = self.adapter(audio_embeddings)

        # 3. Токены
        emb_bos, emb_sa, emb_ea = (
            self._tok_embed(t, B, device)
            for t in (self.bos_id, self.audio_start_id, self.audio_end_id)
        )

        inputs_embeds = torch.cat([emb_bos, emb_sa, audio_embeddings, emb_ea], dim=1)
        att_mask = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.long, device=device),
                torch.ones(B, 1, dtype=torch.long, device=device),
                audio_mask,
                torch.ones(B, 1, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=att_mask,
            max_new_tokens=max_new_tokens,
            # eos_token_id=self.tokenizer.eos_token_id,
            eos_token_id=151645,
            pad_token_id=pad_id,
            **kwargs,
        )
        txt = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return txt[0] if single else txt
