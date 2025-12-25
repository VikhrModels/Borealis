import torch
import gradio as gr
from transformers import AutoModel
import torchaudio

print('Loading Borealis model...')
model = AutoModel.from_pretrained('Vikhrmodels/Borealis-5b-it', trust_remote_code=True, device='cuda')
print('Base model loaded, now loading checkpoint...')
ckpt_path = '/home/alex/Borealis/borealis_adapter_only/checkpoint-28000/pytorch_model.bin'
state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f'Checkpoint loaded from {ckpt_path}')
print(f'Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}')
model.eval()
print('Model ready!')

def process_audio(audio_path, question, system_prompt, max_tokens, temperature, top_p, do_sample):
    print(f'Processing audio: {audio_path}')
    if audio_path is None:
        return "Error: No audio provided"

    try:
        waveform, sr = torchaudio.load(audio_path)
        print(f'Loaded audio: sr={sr}, shape={waveform.shape}')
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio = waveform.squeeze()

        # Add /no_think to disable thinking mode in Qwen3
        prompt = f'{question} /no_think <|start_of_audio|><|end_of_audio|>'

        with torch.inference_mode():
            out = model.generate(
                audio=audio,
                user_prompt=prompt,
                system_prompt=system_prompt or 'You are a helpful voice assistant.',
                max_new_tokens=int(max_tokens),
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        result = model.decode(out[0])

        # Strip thinking tokens if present
        if '<think>' in result:
            result = result.split('</think>')[-1].strip()

        print(f'Generated: {result[:100]}...')
        return result
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(type='filepath', label='Audio'),
        gr.Textbox(label='Question', value='What is being said in this audio?'),
        gr.Textbox(label='System Prompt', value='You are a helpful voice assistant.'),
        gr.Slider(64, 1024, value=256, step=64, label='Max Tokens'),
        gr.Slider(0.1, 1.5, value=0.7, step=0.1, label='Temperature'),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label='Top P'),
        gr.Checkbox(value=True, label='Do Sample'),
    ],
    outputs=gr.Textbox(label='Response', lines=10),
    title='Borealis Adapter-Only (checkpoint-28000)',
)
demo.launch(server_name='0.0.0.0', server_port=7861, share=True)
