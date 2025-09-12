import gradio as gr
from diffusers import StableDiffusionPipeline

# Load SDXL safely on CPU
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)

def generate_image(prompt):
    # Generate image
    image = pipe(prompt).images[0]
    return image

# Create Gradio Interface
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Image(label="Generated Image")
)

if __name__ == "__main__":
    # Launch server for Railway
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(__import__("os").environ.get("PORT", 7860)),
        show_api=True,        # Exposes /api/predict/
        enable_queue=True     # Handles multiple requests safely
    )
