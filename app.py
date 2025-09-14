import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch

# Load SDXL Model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate function with resolution option
def generate(prompt, size):
    if size == "512x512":
        width, height = 512, 512
    elif size == "768x768":
        width, height = 768, 768
    else:  # Default 1024
        width, height = 1024, 1024

    image = pipe(prompt, width=width, height=height).images[0]
    return image

# Gradio UI
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Radio(["512x512", "768x768", "1024x1024"], label="Image Size", value="1024x1024")
    ],
    outputs="image",
)

demo.launch(enable_queue=True)
