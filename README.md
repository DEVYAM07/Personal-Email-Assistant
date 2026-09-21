---
title: AI Email Assistant Backend Service
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Personal Email Assistant — RAG + Gmail OAuth

Backend: FastAPI mounted on Gradio (`app.py` → `app:app` on `0.0.0.0:7860`) → Hugging Face Spaces `DarkByteX/personal-email-assistant` (Gradio SDK, FREE CPU Basic)
Frontend: Vite + React → Vercel `https://personal-email-assistant-2.vercel.app`

## Entry Points
- `app.py:1` — HF Gradio entry (required): `gr.mount_gradio_app(fastapi_app, demo, path="/")` where `fastapi_app` = `main.py:26` `from api import app`
- `main.py:26` — re-exports `api.py:623` `app` (kept for `main:app` compat)
- `api.py:623` — core FastAPI with `/api/health`, `/api/auth/*`, `/api/sync`, `/api/query`, CORS for Vercel + HF

## Deploy (HF Gradio)
See `DEPLOYMENT.md:22` for full guide. Summary:
1. Create Space: Gradio, Blank, CPU Basic → `DarkByteX/personal-email-assistant`
2. `git remote add hf https://huggingface.co/spaces/DarkByteX/personal-email-assistant && git push hf main`
3. Space → Settings → Variables: `GEMINI_API_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `REDIRECT_URI=https://DarkByteX-personal-email-assistant.hf.space/api/auth/callback`, `FRONTEND_URL=https://personal-email-assistant-2.vercel.app`
4. Google Console → add redirect `https://DarkByteX-personal-email-assistant.hf.space/api/auth/callback` + origin `https://DarkByteX-personal-email-assistant.hf.space`
5. Vercel → `VITE_API_URL=https://DarkByteX-personal-email-assistant.hf.space` → redeploy

## Local Dev
```bash
pip install -r requirements.txt  # includes gradio==4.44.1
uvicorn app:app --host 0.0.0.0 --port 7860  # Gradio+FastAPI
# or
uvicorn api:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev  # VITE_API_URL=http://localhost:7860 or 8000
```

## Sync
- HF 16GB RAM restores `maxResults=100` in `api.py:203` and `fetch_emails.py:69` (was capped `15` for Render 512MB OOM).

