import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Clean FastAPI instance for Render (no Gradio)
app = FastAPI(title="Personal Email Assistant API")

# Dynamic CORS — no hardcoded Vercel/Render/HF URLs
# FRONTEND_URL injected via env on Render/Vercel; defaults to localhost for dev
frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Re-export actual API app with all routes (ensures main:app has endpoints)
# api.py already configures matching dynamic CORS via _get_cors_origins()
try:
    from api import app as api_app
    app = api_app
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
