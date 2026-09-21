import gradio as gr
from main import app as fastapi_app  # Import your existing FastAPI app

# Define a lightweight UI wrapper for Hugging Face
with gr.Blocks(title="AI Email Assistant API") as demo:
    gr.Markdown("# AI Email Assistant Backend Service")
    gr.Markdown("FastAPI server is running and ready for Vercel requests.")

# Mount FastAPI app onto Gradio (Gradio serves on 0.0.0.0:7860 by default)
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
