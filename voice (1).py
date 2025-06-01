# Install required packages
!pip install --upgrade diffusers
!pip install invisible_watermark transformers accelerate safetensors
!pip install gradio
!pip install git+https://github.com/openai/whisper.git
!sudo apt install -y ffmpeg
!pip install deepmultilingualpunctuation
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
import whisper
import gradio as gr
from deepmultilingualpunctuation import PunctuationModel
import numpy as np
import random
import os

# Configuration Variables
ENABLE_GAN_MODULE = True
LATENT_DIMENSION = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL_SIZE = "small"
STABLE_DIFFUSION_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# GAN Generator 
class GANGenerator(nn.Module):
    def _init_(self, latent_dim=LATENT_DIMENSION, output_channels=3, image_size=64):
        super(GANGenerator, self)._init_()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_channels * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(-1, 3, self.image_size, self.image_size)
        return img

gan_generator = GANGenerator()

def generate_noise(batch_size=1, latent_dim=LATENT_DIMENSION):
    return torch.randn(batch_size, latent_dim)

def run_gan_generator():
    noise = generate_noise()
    with torch.no_grad():
        generated_image = gan_generator(noise)
    return generated_image
print("Loading Whisper model...")
asr_model = whisper.load_model(WHISPER_MODEL_SIZE)

print("Loading Punctuation model...")
punctuation_model = PunctuationModel()

print("Loading Stable Diffusion pipeline...")
diffusion_pipeline = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path=STABLE_DIFFUSION_MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
diffusion_pipeline.to(DEVICE)
def preprocess_text(text_input):
    text = str(text_input).strip()
    text = text.lower().capitalize()
    return text

def enhance_punctuation(text_input):
    punctuated = punctuation_model.restore_punctuation(text_input)
    cleaned = preprocess_text(punctuated)
    return cleaned

def build_prompt(prompt_text):
    base = "A high-quality image of"
    return f"{base} {prompt_text}, highly detailed, trending on ArtStation, sharp focus"

def transcribe_audio(audio_path):
    result = asr_model.transcribe(audio_path)
    return preprocess_text(result["text"])

def generate_image(prompt_text, negative_text):
    prompt = build_prompt(prompt_text)
    image = diffusion_pipeline(prompt=prompt, negative_prompt=negative_text).images[0]
    return image

def tokenize_prompt(text):
    return [char for char in text]

def preprocess_pipeline(text):
    tokens = tokenize_prompt(text)
    return ''.join(tokens)
with gr.Blocks() as interface:
    gr.Markdown("## 🎤 Voice-to-Image Generator (AI Enhanced)")
    gr.Markdown("This application transcribes your voice prompt using Whisper, applies linguistic enhancement, and generates a rich image using Stable Diffusion XL. GAN module is integrated for future training augmentation.")

    audio_input = gr.Audio(type="filepath", label="🎙️ Upload or Record Your Voice")
    transcribed_output = gr.Textbox(label="📝 Transcribed Prompt", interactive=True)
    negative_prompt_input = gr.Textbox(label="❌ Negative Prompt (optional)")
    generate_btn = gr.Button("🚀 Generate Image")
    image_output = gr.Image(type="pil", label="🖼️ AI Generated Image")

    def handle_audio_input(audio_file):
        print("Processing audio...")
        text = transcribe_audio(audio_file)
        enhanced_text = enhance_punctuation(text)
        preprocess_pipeline(enhanced_text)  # Optional pre-steps
        return enhanced_text

    def handle_generation(prompt_text, negative_text):
        print("Generating image...")
        return generate_image(prompt_text, negative_text)

    audio_input.change(handle_audio_input, inputs=audio_input, outputs=transcribed_output)
    generate_btn.click(handle_generation, inputs=[transcribed_output, negative_prompt_input], outputs=image_output)

print("Launching interface...")
interface.launch(share=True)