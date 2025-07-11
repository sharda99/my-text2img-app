import os
import io
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image

# 1. Ensure HF token is set in environment
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    raise EnvironmentError(
        "Please set the HUGGINGFACEHUB_API_TOKEN environment variable."
    )

# 2. Model & scheduler setup
model_id  = "runwayml/stable-diffusion-v1-5"
device    = "cuda" if torch.cuda.is_available() else "cpu"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16
).to(device)
pipe.enable_attention_slicing()

# 3. Inference function
def generate_image(prompt, steps, guidance, seed):
    gen = torch.Generator(device).manual_seed(int(seed))
    image = pipe(
        prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=gen
    ).images[0]
    return image

# 4. Convert PIL image → bytes for download
def to_bytes(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# 5. Dark-theme CSS for Gradio
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;500;700&display=swap');
body { background: #000 !important; color: #fff !important; font-family: 'Montserrat', sans-serif; }
.gradio-container { max-width: 960px; margin: auto; padding: 30px; }
#header { text-align: center; margin-bottom: 20px; }
#header h1 { font-size: 3rem; margin: 0; color: #fff !important; }
#header p { font-size: 1.1rem; color: #ccc !important; }
.gradio-row > .gradio-column { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
input, textarea, .gradio-slider, .gradio-dropdown { background: rgba(255,255,255,0.1) !important; color: #fff !important; }
.gradio-button { background: #2c5364 !important; color: #fff !important; border: none; }
.gradio-button:hover { background: #1f3f51 !important; }
.gradio-gallery img { border-radius: 8px; transition: transform 0.2s ease; }
.gradio-gallery img:hover { transform: scale(1.05); box-shadow: 0 12px 32px rgba(0,0,0,0.7); }
"""

# 6. Build Gradio Blocks UI
with gr.Blocks(css=custom_css) as demo:
    # Header
    gr.HTML("""
      <div id="header">
        <h1>✨ Text-to-Image Art Studio ✨</h1>
        <p>Type your dream scene, choose style, and watch it come alive!</p>
      </div>
    """)
    # Inputs & controls
    with gr.Row():
        with gr.Column(scale=1):
            prompt    = gr.Textbox(label="Prompt", placeholder="A futuristic city at dusk", lines=2)
            style     = gr.Dropdown(
                           label="Style Presets",
                           choices=["None","Photorealistic","Oil Painting","Cyberpunk","Watercolor"],
                           value="None"
                       )
            steps     = gr.Slider(10, 100, value=50, step=5, label="Inference Steps")
            guidance  = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
            seed      = gr.Number(value=42, label="Random Seed")
            generate  = gr.Button("Generate Image", variant="primary")
        with gr.Column(scale=1):
            gallery   = gr.Gallery(label="Your Art Gallery", columns=2, height="auto")
            last_img  = gr.Image(label="Last Image", interactive=False)
            download  = gr.DownloadButton(label="Download Last Image")
    state = gr.State([])

    def make_and_update(prompt, style, steps, guidance, seed, gallery_list):
        full_prompt = f"{prompt}, {style}" if style != "None" else prompt
        img = generate_image(full_prompt, steps, guidance, seed)
        gallery_list = gallery_list + [img]
        return gallery_list, img, gallery_list

    generate.click(
        fn=make_and_update,
        inputs=[prompt, style, steps, guidance, seed, state],
        outputs=[gallery, last_img, state]
    )
    download.click(fn=to_bytes, inputs=last_img, outputs=download)

# 7. Launch (no share=True)
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
