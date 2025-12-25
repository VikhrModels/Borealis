"""
Gradio UI for Borealis Audio-Language Model
"""

import os
os.environ["HF_AUDIO_DECODER_BACKEND"] = "soundfile"

import torch
import gradio as gr
from transformers import AutoModel

# Global model variable
model = None

CHECKPOINT_PATH = "/home/alex/Borealis/ablations/outputs/ru_text_10pct/checkpoint-1624/pytorch_model.bin"

def load_model():
    global model
    if model is None:
        print("Loading Borealis model...")
        model = AutoModel.from_pretrained(
            "Vikhrmodels/Borealis-5b-it",
            trust_remote_code=True,
            device="cuda"
        )

        # Load ablation checkpoint weights
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Loading checkpoint: {CHECKPOINT_PATH}")
            state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded!")

        model.eval()
        print("Model ready!")
    return model

def process_audio(audio, system_prompt, user_prompt, max_tokens, temperature, top_p):
    """Process audio and generate response."""
    if audio is None:
        return "Please upload or record an audio file."

    m = load_model()

    sr, audio_array = audio

    # Convert to torch tensor and normalize
    audio_tensor = torch.tensor(audio_array).float()
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=-1)  # Convert stereo to mono

    # Normalize to [-1, 1] if needed
    if audio_tensor.abs().max() > 1.0:
        audio_tensor = audio_tensor / 32768.0

    # Resample if needed
    if sr != 16000:
        import torchaudio
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)

    # Ensure audio tags in prompt
    if "<|start_of_audio|>" not in user_prompt:
        user_prompt = f"{user_prompt} <|start_of_audio|><|end_of_audio|>"

    with torch.inference_mode():
        output = m.generate(
            audio=audio_tensor,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

    response = m.decode(output[0])
    return response

# Preset prompts
PRESET_PROMPTS = {
    "Transcription (EN)": {
        "system": "You are a speech recognition assistant. Accurately transcribe audio to text.",
        "user": "Transcribe this audio: <|start_of_audio|><|end_of_audio|>"
    },
    "Transcription (RU)": {
        "system": "Ты ассистент по распознаванию речи. Точно транскрибируй аудио в текст.",
        "user": "Транскрибируй это аудио: <|start_of_audio|><|end_of_audio|>"
    },
    "Summarization (EN)": {
        "system": "You are a helpful voice assistant.",
        "user": "Summarize what is said in this recording: <|start_of_audio|><|end_of_audio|>"
    },
    "Summarization (RU)": {
        "system": "Ты полезный голосовой ассистент.",
        "user": "Кратко перескажи содержание аудио: <|start_of_audio|><|end_of_audio|>"
    },
    "Q&A (EN)": {
        "system": "You are a helpful voice assistant. Listen to the audio and respond appropriately.",
        "user": "What is being discussed in this audio? <|start_of_audio|><|end_of_audio|>"
    },
    "Q&A (RU)": {
        "system": "Ты полезный голосовой ассистент. Слушай аудио и отвечай на вопросы.",
        "user": "О чём говорится в этой аудиозаписи? <|start_of_audio|><|end_of_audio|>"
    },
    "Description (EN)": {
        "system": "You are an attentive listener.",
        "user": "Describe in detail what you hear: <|start_of_audio|><|end_of_audio|>"
    },
    "Description (RU)": {
        "system": "Ты внимательный слушатель.",
        "user": "Опиши подробно, что ты слышишь: <|start_of_audio|><|end_of_audio|>"
    },
    "Custom": {
        "system": "You are a helpful voice assistant.",
        "user": "<|start_of_audio|><|end_of_audio|>"
    }
}

def update_prompts(preset):
    """Update prompts based on selected preset."""
    prompts = PRESET_PROMPTS.get(preset, PRESET_PROMPTS["Custom"])
    return prompts["system"], prompts["user"]

# Build Gradio interface
with gr.Blocks(title="Borealis Audio-Language Model") as demo:
    gr.Markdown("""
    # Borealis-5B-IT (ru_text_10pct/checkpoint-1624)

    Audio-Language Model for Speech Understanding | **WER: 19.17%** ⭐

    Upload or record audio, select a prompt preset or write your own, and generate a response.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Audio Input",
                type="numpy",
                sources=["upload", "microphone"]
            )

            preset_dropdown = gr.Dropdown(
                choices=list(PRESET_PROMPTS.keys()),
                value="Q&A (EN)",
                label="Prompt Preset"
            )

            system_prompt = gr.Textbox(
                label="System Prompt",
                value=PRESET_PROMPTS["Q&A (EN)"]["system"],
                lines=2
            )

            user_prompt = gr.Textbox(
                label="User Prompt",
                value=PRESET_PROMPTS["Q&A (EN)"]["user"],
                lines=2,
                info="Include <|start_of_audio|><|end_of_audio|> tags where audio should be placed"
            )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=32,
                    maximum=1024,
                    value=256,
                    step=32,
                    label="Max Tokens"
                )

            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p"
                )

            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Model Response",
                lines=15
            )

    # Event handlers
    preset_dropdown.change(
        fn=update_prompts,
        inputs=[preset_dropdown],
        outputs=[system_prompt, user_prompt]
    )

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, system_prompt, user_prompt, max_tokens, temperature, top_p],
        outputs=[output_text]
    )

    gr.Markdown("""
    ---
    **Model**: [Vikhrmodels/Borealis-5b-it](https://huggingface.co/Vikhrmodels/Borealis-5b-it)

    **Architecture**: Whisper Large V3 (encoder) + Qwen3-4B (LLM)
    """)

if __name__ == "__main__":
    # Preload model
    load_model()

    # Launch app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
