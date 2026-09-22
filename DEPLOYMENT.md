# Deployment Guide — Vercel (Frontend) + Render (Backend)

This repo is split:
- **Frontend** (`/frontend`): Vite + React → **Vercel**
- **Backend** (`/`): FastAPI (`api.py` via `main.py`) → **Render** (free, 512MB)

---

## 1) Prerequisites

- Google Cloud OAuth 2.0 **Web** client
  - Console → APIs & Services → Credentials → Create OAuth client → **Web application**
  - Add **Authorized redirect URIs** (use your fresh deployment URLs, not previous hardcoded ones):
    - Local: `http://localhost:8000/api/auth/callback`
    - Prod: `https://your-backend.onrender.com/api/auth/callback` (your new Render URL)
  - Add **Authorized JavaScript origins**:
    - Local: `http://localhost:5173`
    - Prod: `https://your-frontend.vercel.app` (your new Vercel URL)
- Gemini API key

---

## 2) Backend — Render

### Option A: `render.yaml` (Infrastructure as Code)
1. Push repo to GitHub.
2. Render → New → Blueprint → select repo (reads `render.yaml`).
3. Set env vars in Render dashboard (sync: false → fill manually):
   ```
   GEMINI_API_KEY=...
   GOOGLE_CLIENT_ID=...
   GOOGLE_CLIENT_SECRET=...
   REDIRECT_URI=https://your-backend.onrender.com/api/auth/callback
   FRONTEND_URL=https://your-frontend.vercel.app
   SYNC_BATCH_SIZE=15
   DB_PATH=./emails.db
   CHROMA_DB_PATH=./chroma_db
   ```
4. Deploy → check `https://your-backend.onrender.com/api/health` and `/docs` (dynamic URLs).

### Option B: Manual Render Web Service
- Runtime: **Python 3.10+**
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT` (main.py reads `PORT` env, defaults 8000)
- Health check: `/api/health`
- Env: same as above; no persistent disk needed for free tier (ephemeral FS). For persistence later, add disk `/data`.

> **Sync 15 emails per pass:** `SYNC_BATCH_SIZE=15` (default, was 100) uses remote `text-embedding-004` with `embedding_function=None` to keep RAM low; background `202` on `/api/sync` avoids gateway timeouts.

---

## 3) Frontend — Vercel

### Deploy
1. Vercel → Add New → Project → Import GitHub repo.
2. **Root Directory**: `frontend` (important — else Vercel builds Python backend)
3. Framework: **Vite**
4. Build: `npm run build`  Output: `dist` (`frontend/vercel.json`)
5. Env Var (dynamic, no hardcoded previous URL):
   ```
   VITE_API_URL=https://your-backend.onrender.com
   ```
   (no trailing slash; app appends `/api/...`)
6. Deploy → visit your Vercel URL.

### Local preview with prod backend
```bash
cd frontend
echo "VITE_API_URL=https://your-backend.onrender.com" > .env.local
npm run dev
```

---

## 4) Wiring OAuth for Production

1. Backend `REDIRECT_URI` must exactly match Google Console redirect.
2. Backend `FRONTEND_URL` must be your Vercel URL (CORS `allow_origins` from `FRONTEND_URL` + `http://localhost:5173`).
3. Frontend `VITE_API_URL` must be your Render backend URL (frontend `App.jsx:20` reads `VITE_API_URL` or `http://localhost:8000`).
4. After env changes: Render auto-redeploys; Vercel needs redeploy (Vite bakes env).

Test flow:
- Open frontend → `Connect Gmail` → Google consent → redirects `FRONTEND_URL?auth=success&email=...` → badge shows.

---

## 5) Verification

```bash
# backend health (Render, dynamic URL)
curl https://your-backend.onrender.com/api/health
# -> {"status":"ok"}

# CORS check from frontend origin
curl -H "Origin: https://your-frontend.vercel.app" -I https://your-backend.onrender.com/api/health

# frontend build
cd frontend && npm run build && npm run preview
```

---

## 6) Common Pitfalls

| Issue | Fix |
|-------|-----|
| `CORS blocked` | Ensure `FRONTEND_URL` is set to your Vercel URL; backend reads it dynamically via `os.environ.get("FRONTEND_URL", "http://localhost:5173")`. Check `CORS_ALLOWED_ORIGINS` extra. |
| `redirect_uri_mismatch` | `REDIRECT_URI` env must equal Google Console URI exactly (including `https` and trailing path). |
| `No refresh token` | OAuth URL uses `access_type=offline` & `prompt=consent` (`api.py:691`). Clear prior grant and re-auth. |
| `OOM on sync` | Uses remote `text-embedding-004` with `embedding_function=None` to avoid local ONNX/Torch; `SYNC_BATCH_SIZE=15` optimized for 512MB/0.1vCPU (was 100). |
| `VITE_API_URL not applied` | Vercel needs redeploy after env change; `VITE_` prefix required, no trailing slash. |
| `Render sleeps` | Free tier spins down → cold start ~30s; consider UptimeRobot or upgrade. |

---

## 7) Future Hardening (optional)

- Swap SQLite for Postgres (Neon/Supabase) + update `database.py` / `api.py` with persistent `DB_PATH`.
- Hosted vector DB (Chroma Cloud/Pinecone) for persistence.
- `uvicorn --workers 2` behind gunicorn for concurrency.
- Set `OAUTHLIB_INSECURE_TRANSPORT=0` in prod.
