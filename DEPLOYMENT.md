# Deployment Guide — Vercel (Frontend) + Hugging Face Gradio (Backend)

This repo is split:
- **Frontend** (`/frontend`): Vite + React → **Vercel**
- **Backend** (`/`): FastAPI mounted on Gradio (`app.py` → `app:app`) → **Hugging Face Spaces (Gradio SDK, FREE, CPU Basic)**

> **Deprecated:** `render.yaml` / Render deployment removed. HF Gradio Space is now primary. Docker `Dockerfile` remains for optional local Docker, now serves `app:app` (Gradio + FastAPI) on `7860`.

---

## 1) Prerequisites

- Google Cloud OAuth 2.0 **Web** client (not Desktop)
  - Console → APIs & Services → Credentials → Create OAuth client → **Web application**
  - Add **Authorized redirect URIs**:
    - Local: `http://localhost:7860/api/auth/callback` (via `app.py` Gradio mount, port 7860)
    - HF Prod: `https://DarkByteX-personal-email-assistant.hf.space/api/auth/callback` (exact, case-sensitive)
    - Legacy local 8000: `http://localhost:8000/api/auth/callback`
  - Add **Authorized JavaScript origins**:
    - `https://personal-email-assistant-2.vercel.app`
    - `https://DarkByteX-personal-email-assistant.hf.space`
- Gemini API key
- Hugging Face Space: `DarkByteX/personal-email-assistant` (Gradio, Blank template, CPU Basic Free)

---

## 2) Backend — Hugging Face Spaces (Gradio, Free)

### Create Space
1. Hugging Face → New Space → **Gradio** (orange), Template **Blank**, Hardware **CPU Basic** → Create `DarkByteX/personal-email-assistant`.
2. Local → `git remote add hf https://huggingface.co/spaces/DarkByteX/personal-email-assistant`
3. `git push hf main` → HF builds: `pip install -r requirements.txt` (includes `gradio>=4.0.0`) and runs `app.py` (Gradio mounts `main:app` at `/`).

### Space → Settings → Variables and secrets
```
GEMINI_API_KEY=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
REDIRECT_URI=https://DarkByteX-personal-email-assistant.hf.space/api/auth/callback
FRONTEND_URL=https://personal-email-assistant-2.vercel.app
# optional (defaults to relative paths suitable for HF ephemeral FS):
# DB_PATH=./emails.db
# CHROMA_DB_PATH=./chroma_db
```

*Note:* HF ephemeral FS — `emails.db`/`chroma_db` reset on Space restart (same as Render free). For persistence later, migrate to Postgres + hosted vector DB.

### Entry Point
- `app.py` (HF required):
  ```python
  import gradio as gr
  from main import app as fastapi_app
  with gr.Blocks(title="AI Email Assistant API") as demo:
      gr.Markdown("# AI Email Assistant Backend Service")
      gr.Markdown("FastAPI server is running and ready for Vercel requests.")
  app = gr.mount_gradio_app(fastapi_app, demo, path="/")
  ```
- `main.py:26` re-exports `api.py:623` `app` for `from main import app` import chain.
- `Dockerfile:19` `CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]` serves mounted app via Docker (HF Gradio SDK ignores Dockerfile).

Verify:
```
curl https://DarkByteX-personal-email-assistant.hf.space/api/health
# -> {"status":"ok"}
curl -H "Origin: https://personal-email-assistant-2.vercel.app" -I https://DarkByteX-personal-email-assistant.hf.space/api/health
# -> access-control-allow-origin
```

---

## 3) Frontend — Vercel

### Deploy
1. Vercel → Add New → Project → Import GitHub repo.
2. **Root Directory**: `frontend` (very important — or Vercel will try to build the Python backend)
3. Framework preset: **Vite** (auto-detected)
4. Build Command: `npm run build`  Output: `dist`  (already in `frontend/vercel.json`)
5. Env Var:
   ```
   VITE_API_URL=https://DarkByteX-personal-email-assistant.hf.space
   ```
   (no trailing slash; the app appends `/api/...`; was `https://personal-email-assistant-api.onrender.com`)
6. Deploy → visit `https://personal-email-assistant-2.vercel.app`.

### Local preview with HF backend
```bash
cd frontend
echo "VITE_API_URL=https://DarkByteX-personal-email-assistant.hf.space" > .env.local
npm run dev
```

---

## 4) Wiring OAuth for Production (HF)

1. Backend env `REDIRECT_URI` must equal **exact** HF Space URI in Google Console: `https://DarkByteX-personal-email-assistant.hf.space/api/auth/callback`
2. Backend env `FRONTEND_URL` must be your Vercel URL (CORS + post-login redirect) — `api.py:625` `_get_cors_origins()` allows `https://personal-email-assistant-2.vercel.app` + `*.vercel.app`.
3. Frontend env `VITE_API_URL` must be your HF Space URL.
4. After env changes: HF Space auto-restarts (no manual redeploy); Vercel needs redeploy (Vite bakes env).

Test flow:
- Open frontend → `Connect Gmail` → Google consent → redirects back to `FRONTEND_URL?auth=success&email=...` → badge shows → `GET /api/auth/status?email=...` → `POST /api/sync?email=...` (202 + poll `GET /api/sync/status?job_id=`) → `POST /api/query`.

---

## 5) Verification

```bash
# backend health (HF)
curl https://DarkByteX-personal-email-assistant.hf.space/api/health
# -> {"status":"ok"}

# CORS check from frontend origin
curl -H "Origin: https://personal-email-assistant-2.vercel.app" -I https://DarkByteX-personal-email-assistant.hf.space/api/health

# Gradio UI
curl https://DarkByteX-personal-email-assistant.hf.space/
# -> Gradio HTML with "AI Email Assistant Backend Service"

# frontend build
cd frontend && npm run build && npm run preview
```

---

## 6) Common Pitfalls

| Issue | Fix |
|-------|-----|
| `CORS blocked` | `FRONTEND_URL` must be Vercel URL; `api.py` allows via `CORSMiddleware(allow_origins=_get_cors_origins(), allow_origin_regex=r"https://.*\.vercel\.app")`. Add HF URL to `CORS_ALLOWED_ORIGINS` if needed. |
| `redirect_uri_mismatch` | `REDIRECT_URI` env + Google Console URI must match **including** `https` and trailing path and case `DarkByteX-...`. Re-save credentials and wait 5 min. |
| `No refresh token` | Ensure OAuth URL uses `access_type=offline` & `prompt=consent` (already in `api.py:691`). Re-auth after clearing Google App permissions. |
| `SQLite read-only` on HF | Use relative `./emails.db` / `./chroma_db` (default `api.py:65`). `/tmp` was Render-specific; now deprecated. |
| `VITE_API_URL not applied` | Vercel needs redeploy after env change; Vite envs are baked at build time. Ensure var starts with `VITE_` and no trailing slash. |
| `Gradio 404 on HF` | Ensure `app.py` exists at root with `app = gr.mount_gradio_app(...)` and `sdk: gradio` in `README.md` frontmatter; HF Build must be `Gradio` not `Docker`. |
| `Space sleeps` | HF CPU Basic sleeps after inactivity → first request ~10-20s cold start. No UptimeRobot needed like Render. |

---

## 7) Future Production Hardening (optional)

- Swap SQLite for Postgres (Neon/Supabase) + update `database.py` / `api.py`.
- Move Chroma to persistent hosted (Chroma Cloud, Pinecone, Qdrant).
- Add `uvicorn --workers 2` behind gunicorn for concurrency (Docker only).
- Set `OAUTHLIB_INSECURE_TRANSPORT=0` in prod (currently `1` for HF http callback dev).
- Add persistent volume for HF (paid) or external DB if 100-email sync needs durability.
