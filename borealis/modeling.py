import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        audio_encoder=None,
        tokenizer=None,
        language_model=None,
        downsample_factor: int = 4,
    ):
        super().__init__()

        self.encoder = audio_encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.llm = language_model
        self.tokenizer = tokenizer
        self.llm.resize_token_embeddings(len(tokenizer))

        print("Pad token:", self.llm.config.pad_token_id)
        print("EOS token:", self.llm.config.eos_token_id)

        print("Tokenizer EOS token ID:", tokenizer.eos_token_id)
        print("Tokenizer PAD token ID:", tokenizer.pad_token_id)

        self.downsample_factor = downsample_factor
        self.adapter = AudioLanguageAdapter(
            hidden_size=self.encoder.config.d_model * downsample_factor,
            dim=self.llm.config.hidden_size,
        ).to(self.llm.dtype)

        self.bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")

        self.chunk_mel_frames = 3000

    def _downsample(self, seq: torch.Tensor) -> torch.Tensor:
        k, (T, d) = self.downsample_factor, seq.shape
        target = k * math.ceil(T / k)
        if target != T:
            seq = F.pad(seq, (0, 0, 0, target - T))
        return seq.contiguous().view(target // k, d * k)

    def _tok_embed(self, tok_id: int, batch: int, device) -> torch.Tensor:
        idx = torch.full((batch, 1), tok_id, dtype=torch.long, device=device)
        return self.llm.get_input_embeddings()(idx)

    def _process_audio(self, mel) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        B, device = len(mel), mel[0][0].device
        audio_embs = []
        audio_mask = []
        per_sample_T = []
        max_T = 0
        for b in range(B):
            chunk_stack = torch.stack(mel[b])
            enc_chunks = self.encoder(
                input_features=chunk_stack, return_dict=True
            ).last_hidden_state
            # enc_long = torch.cat(enc_chunks, dim=0)
            enc_long = enc_chunks.view(-1, enc_chunks.size(-1))
            ds_long = self._downsample(enc_long)
            audio_embs.append(ds_long)
            per_sample_T.append(ds_long.size(0))
            max_T = max(max_T, ds_long.size(0))

        for i in range(B):
            pad = max_T - per_sample_T[i]
            if pad > 0:
                audio_embs[i] = F.pad(audio_embs[i], (0, 0, 0, pad))
                audio_mask.append(
                    torch.ones(per_sample_T[i], dtype=torch.long, device=device)
                )
                audio_mask[i] = F.pad(audio_mask[i], (0, pad), value=0)
            else:
                audio_mask.append(
                    torch.ones(per_sample_T[i], dtype=torch.long, device=device)
                )

        audio_embeddings = torch.stack(audio_embs)
        audio_mask = torch.stack(audio_mask)
        audio_embeddings = self.adapter(audio_embeddings)

        return audio_embeddings, audio_mask, per_sample_T

    def forward(
        self,
        mel,
        labels: torch.Tensor,
        text_att_mask: torch.Tensor,
    ):
        B, device = labels.size(0), labels.device

        audio_embeddings, audio_mask, per_sample_T = self._process_audio(mel)

        text_embeddings = self.llm.get_input_embeddings()(labels)

        sa_positions = (labels == self.audio_start_id).nonzero(as_tuple=True)
        ea_positions = (labels == self.audio_end_id).nonzero(as_tuple=True)

        inputs_embeds = []
        att_mask = []
        for b in range(B):
            sa_idx = sa_positions[1][sa_positions[0] == b].item()
            ea_idx = ea_positions[1][ea_positions[0] == b].item()

            prefix_emb = text_embeddings[b, : sa_idx + 1]
            postfix_emb = text_embeddings[b, ea_idx:]

            emb = torch.cat([prefix_emb, audio_embeddings[b], postfix_emb], dim=0)

            prefix_mask = text_att_mask[b, : sa_idx + 1]
            postfix_mask = text_att_mask[b, ea_idx:]
            full_mask = torch.cat([prefix_mask, audio_mask[b], postfix_mask], dim=0)

            inputs_embeds.append(emb)
            att_mask.append(full_mask)

        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            inputs_embeds, batch_first=True, padding_value=0.0
        )
        att_mask = torch.nn.utils.rnn.pad_sequence(
            att_mask, batch_first=True, padding_value=0
        )

        assistant_prompt = self.tokenizer(
            "<|im_start|>assistant\n", add_special_tokens=False
        ).input_ids
        assistant_starts = []
        for b in range(B):
            seq = labels[b]
            for i in range(len(seq) - len(assistant_prompt)):
                if torch.equal(
                    seq[i : i + len(assistant_prompt)],
                    torch.tensor(assistant_prompt, device=device),
                ):
                    assistant_start = i + len(assistant_prompt)
                    break
            else:
                raise ValueError("Assistant prompt not found")
            assistant_starts.append(
                assistant_start + (ea_idx - sa_idx - 1) + per_sample_T[b]
            )

        max_len = inputs_embeds.size(1)
        loss_labels = labels.new_full((B, max_len), -100)
        for b in range(B):
            orig_assist_start = (
                assistant_starts[b] - per_sample_T[b] - (ea_idx - sa_idx - 1)
            )
            content_len = len(labels[b]) - orig_assist_start
            loss_labels[b, assistant_starts[b] : assistant_starts[b] + content_len] = (
                labels[b, orig_assist_start:]
            )

        if self.tokenizer.pad_token_id is not None:
            loss_labels[loss_labels == self.tokenizer.pad_token_id] = -100

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=att_mask,
            labels=loss_labels,
            return_dict=True,
        )
        return out.loss, out.logits

    @torch.inference_mode()
    def generate(
        self,
        mel,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        single = not isinstance(mel[0], list)
        if single:
            mel = [mel]

        mel = [[c.to(torch.bfloat16) for c in m] for m in mel]

        B, device = len(mel), mel[0][0].device

        audio_embeddings, audio_mask, per_sample_T = self._process_audio(mel)

        messages = [
            {
                "role": "system",
                "content": "Вы полезный помощник по автоматическому распознаванию речи. Точно транскрибируйте аудио в текст.",
            },
            {
                "role": "user",
                "content": "Транскрибируйте это аудио: <|start_of_audio|><|end_of_audio|>",
            },
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        model_inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)

        input_ids = model_inputs.input_ids.repeat(B, 1)
        text_att_mask = model_inputs.attention_mask.repeat(B, 1)

        text_embeddings = self.llm.get_input_embeddings()(input_ids)

        sa_idx = (input_ids[0] == self.audio_start_id).nonzero(as_tuple=True)[0].item()
        ea_idx = (input_ids[0] == self.audio_end_id).nonzero(as_tuple=True)[0].item()

        inputs_embeds = []
        full_att_mask = []
        for b in range(B):
            prefix_emb = text_embeddings[b, : sa_idx + 1]
            postfix_emb = text_embeddings[b, ea_idx:]
            emb = torch.cat([prefix_emb, audio_embeddings[b], postfix_emb], dim=0)

            prefix_mask = text_att_mask[b, : sa_idx + 1]
            postfix_mask = text_att_mask[b, ea_idx:]
            mask = torch.cat([prefix_mask, audio_mask[b], postfix_mask], dim=0)

            inputs_embeds.append(emb)
            full_att_mask.append(mask)

        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            inputs_embeds, batch_first=True, padding_value=0.0
        )

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        txt = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return txt[0] if single else txt
