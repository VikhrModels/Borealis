import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, WhisperModel
import math


class AudioLanguageAdapter(nn.Module):
    def __init__(self, hidden_size: int, dim: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(hidden_size, dim, bias=False)
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))


class BorealisForConditionalGeneration(nn.Module):
    def __init__(
        self,
        whisper_encoder_name="openai/whisper-large-v3-turbo",
        llm_name="Qwen/Qwen2.5-0.5B",
        tokenizer=None,
        downsample_factor=4,
    ):
        super().__init__()
        assert tokenizer is not None, "Tokenizer надо передать в модельку"

        self.encoder = WhisperModel.from_pretrained(whisper_encoder_name).encoder
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer
        self.llm.resize_token_embeddings(len(tokenizer))

        self.downsample_factor = downsample_factor
        self.adapter = AudioLanguageAdapter(
            hidden_size=self.encoder.config.d_model * self.downsample_factor,
            dim=self.llm.config.hidden_size,
        )

        self.bos_id = tokenizer("<|im_start|>", return_tensors="pt")["input_ids"][0, 0]
        self.audio_start_id = tokenizer("<|start_of_audio|>", return_tensors="pt")[
            "input_ids"
        ][0, 0]
        self.audio_end_id = tokenizer("<|end_of_audio|>", return_tensors="pt")[
            "input_ids"
        ][0, 0]

    def forward(
        self,
        mel: torch.Tensor,  # (B, 128, 3000)
        audio_att_mask: torch.Tensor,  # (B, 3000)
        labels: torch.Tensor,  # (B, max_text_len))
        text_att_mask: torch.Tensor,  # (B, max_text_len)
    ):
        B = mel.size(0)
        device = mel.device

        enc_out = self.encoder(
            input_features=mel,
            attention_mask=audio_att_mask,
            return_dict=True,
        ).last_hidden_state

        audio_enc_list = []
        for embedding in enc_out:  # по батчу: каждый (seq_len, d_model)
            seq_len, dim = embedding.shape
            target_seq_len = self.downsample_factor * math.ceil(
                seq_len / self.downsample_factor
            )
            padded_embedding = torch.nn.functional.pad(
                embedding,
                (0, 0, 0, target_seq_len - seq_len),
            )
            reshaped = padded_embedding.reshape(
                target_seq_len // self.downsample_factor, dim * self.downsample_factor
            )
            audio_enc_list.append(reshaped)

        ready_for_adapter = torch.cat(audio_enc_list, dim=0)

        adapter_out = self.adapter(ready_for_adapter)

        audio_embeddings_list = list(
            torch.split(adapter_out, [a.shape[0] for a in audio_enc_list], dim=0)
        )

        audio_embeddings = torch.stack(audio_embeddings_list, dim=0)

        transcript_embeddings = self.llm.get_input_embeddings()(labels)

        def expand_embed(token_id):
            idx = torch.full((B, 1), token_id, device=device, dtype=torch.long)
            return self.llm.get_input_embeddings()(idx)

        emb_bos = expand_embed(self.bos_id)
        emb_sa = expand_embed(self.audio_start_id)
        emb_ea = expand_embed(self.audio_end_id)

        ready_inputs = torch.cat(
            [
                emb_bos,  # (B, 1, d)
                emb_sa,  # (B, 1, d)
                audio_embeddings,  # (B, 1500, d)
                emb_ea,  # (B, 1, d)
                transcript_embeddings,  # (B, L, d)
            ],
            dim=1,
        )

        def ones(n):
            return torch.ones(B, n, device=device, dtype=torch.float32)

        ready_att_mask = torch.cat(
            [
                ones(1),  # bos
                ones(1),  # start_audio
                ones(audio_embeddings.size(1)),  # аудио (1500)
                ones(1),  # end_audio
                text_att_mask,  # текст
            ],
            dim=1,
        )

        prefix_len = 1 + 1 + audio_embeddings.size(1) + 1
        ignore_pre = labels.new_full((B, prefix_len), -100)
        loss_labels = torch.cat([ignore_pre, labels], dim=1)

        loss_labels[loss_labels == self.tokenizer.pad_token_id] = -100

        out = self.llm(
            inputs_embeds=ready_inputs,
            attention_mask=ready_att_mask,
            labels=loss_labels,
            return_dict=True,
        )
        return out.loss, out.logits

    def generate(
        self,
        mel: torch.Tensor,  # (B, 128, 3000) or (128, 3000)
        att_mask: torch.Tensor,  # (B, 3000) or (3000,)
        max_new_tokens: int = 512,
        **kwargs,
    ):
        is_single = False
        if mel.dim() == 2:
            is_single = True
            mel = mel.unsqueeze(0)
            att_mask = att_mask.unsqueeze(0)

        B = mel.size(0)
        device = mel.device

        enc_out = self.encoder(
            input_features=mel,
            attention_mask=att_mask,
            return_dict=True,
        ).last_hidden_state

        audio_enc_list = []
        for embedding in enc_out:  # по батчу: каждый (seq_len, d_model)
            seq_len, dim = embedding.shape
            target_seq_len = self.downsample_factor * math.ceil(
                seq_len / self.downsample_factor
            )
            padded_embedding = torch.nn.functional.pad(
                embedding,
                (0, 0, 0, target_seq_len - seq_len),
            )
            reshaped = padded_embedding.reshape(
                target_seq_len // self.downsample_factor, dim * self.downsample_factor
            )
            audio_enc_list.append(reshaped)

        ready_for_adapter = torch.cat(audio_enc_list, dim=0)

        adapter_out = self.adapter(ready_for_adapter)

        audio_embeddings_list = list(
            torch.split(adapter_out, [a.shape[0] for a in audio_enc_list], dim=0)
        )

        audio_embeddings = torch.stack(audio_embeddings_list, dim=0)

        def expand_embed(token_id):
            idx = torch.full((B, 1), token_id, device=device, dtype=torch.long)
            return self.llm.get_input_embeddings()(idx)

        emb_bos = expand_embed(self.bos_id)
        emb_sa = expand_embed(self.audio_start_id)
        emb_ea = expand_embed(self.audio_end_id)

        inputs_embeds = torch.cat(
            [
                emb_bos,  # (B, 1, d)
                emb_sa,  # (B, 1, d)
                audio_embeddings,  # (B, reduced_seq, d)
                emb_ea,  # (B, 1, d)
            ],
            dim=1,
        )

        ones = lambda n: torch.ones(B, n, device=device, dtype=torch.float32)
        attention_mask = torch.cat(
            [
                ones(1),  # bos
                ones(1),  # start_audio
                ones(audio_embeddings.size(1)),  # аудио
                ones(1),  # end_audio
            ],
            dim=1,
        )

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
            **kwargs,
        )

        transcripts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        if is_single:
            return transcripts[0]
        return transcripts
