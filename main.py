import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Personal Email Assistant API")

# Fetch FRONTEND_URL and sanitize trailing slashes
frontend_url = os.getenv("FRONTEND_URL", "https://personal-email-assistant-mvp.vercel.app").rstrip("/")

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    frontend_url,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Re-export actual API app with all routes (ensures main:app has endpoints)
# api.py already configures matching dynamic CORS via _get_cors_origins()
# Keep CORS fix effective after re-export by re-applying same middleware to api_app
try:
    from api import app as api_app
    # Re-apply corrected CORS to api_app so OPTIONS preflight from Vercel passes even after app = api_app
    try:
        api_app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception:
        pass
    app = api_app
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
