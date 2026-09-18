# Deployment Guide — Vercel (Frontend) + Render (Backend)

This repo is split:
- **Frontend** (`/frontend`): Vite + React → **Vercel**
- **Backend** (`/`): FastAPI (`api.py`) → **Render**

---

## 1) Prerequisites

- Google Cloud OAuth 2.0 **Web** client (not Desktop)
  - Console → APIs & Services → Credentials → Create OAuth client → **Web application**
  - Add **Authorized redirect URIs**:
    - Local: `http://localhost:8000/api/auth/callback`
    - Prod: `https://your-api.onrender.com/api/auth/callback`
  - Add **Authorized JavaScript origins**:
    - `https://your-frontend.vercel.app`
- Gemini API key

---

## 2) Backend — Render

### Option A: `render.yaml` (Infrastructure as Code)
1. Push this repo to GitHub.
2. Render → New → Blueprint → select repo (it reads `render.yaml`).
3. Set env vars in Render dashboard (they are `sync: false` in yaml, so you must fill them):
   ```
   GEMINI_API_KEY=...
   GOOGLE_CLIENT_ID=...
   GOOGLE_CLIENT_SECRET=...
   REDIRECT_URI=https://your-api.onrender.com/api/auth/callback
   FRONTEND_URL=https://your-frontend.vercel.app
   DB_PATH=/data/emails.db
   CHROMA_DB_PATH=/data/chroma_db
   ```
4. Deploy → check `https://your-api.onrender.com/api/health` and `/docs`.

### Option B: Manual Render Web Service
- Runtime: **Python 3.11**
- Build: `pip install -r requirements.txt`
- Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Health check: `/api/health`
- **Disk** (critical for SQLite + Chroma): Add disk → Name `data`, Mount `/data`, 1GB → set `DB_PATH=/data/emails.db`, `CHROMA_DB_PATH=/data/chroma_db`
- Add same env vars as above.

> **Ephemeral FS warning:** Without a disk, `emails.db` and `chroma_db` are wiped on each deploy/restart. Free Render has ephemeral FS; paid adds disks. Alternative: migrate to Postgres + hosted vector DB later.

---

## 3) Frontend — Vercel

### Deploy
1. Vercel → Add New → Project → Import GitHub repo.
2. **Root Directory**: `frontend`  (very important — or Vercel will try to build the Python backend)
3. Framework preset: **Vite** (auto-detected)
4. Build Command: `npm run build`  Output: `dist`  (already in `frontend/vercel.json`)
5. Env Var:
   ```
   VITE_API_URL=https://your-api.onrender.com
   ```
   (no trailing slash; the app appends `/api/...`)
6. Deploy → visit `https://your-frontend.vercel.app`.

### Local preview with prod backend
```bash
cd frontend
echo "VITE_API_URL=https://your-api.onrender.com" > .env.local
npm run dev
```

---

## 4) Wiring OAuth for Production

After both are deployed:

1. Backend env `REDIRECT_URI` must equal the **exact** URI in Google Console.
2. Backend env `FRONTEND_URL` must be your Vercel URL (CORS + post-login redirect).
3. Frontend env `VITE_API_URL` must be your backend URL.
4. Redeploy backend after env changes (Render auto-redeploys).

Test flow:
- Open frontend → `Connect Gmail` → Google consent → redirects back to `FRONTEND_URL?auth=success&email=...` → badge shows.

---

## 5) Verification

```bash
# backend health
curl https://your-api.onrender.com/api/health
# -> {"status":"ok"}

# CORS check from frontend origin
curl -H "Origin: https://your-frontend.vercel.app" -I https://your-api.onrender.com/api/health

# frontend build
cd frontend && npm run build && npm run preview
```

---

## 6) Common Pitfalls

| Issue | Fix |
|-------|-----|
| `CORS blocked` | Add your Vercel URL to `FRONTEND_URL` (comma-separated if multiple). Backend `api.py` parses `FRONTEND_URL` + `CORS_ALLOWED_ORIGINS`. |
| `redirect_uri_mismatch` | `REDIRECT_URI` env + Google Console URI must match **including** `https` and trailing path. Re-save credentials and wait 5 min. |
| `No refresh token` | Ensure OAuth URL uses `access_type=offline` & `prompt=consent` (already in `api.py`). Re-auth after clearing Google App permissions. |
| `SQLite read-only` on Render | You used `/data` but didn't attach a Disk. Add Disk or change `DB_PATH` to `/tmp` (ephemeral) for testing. |
| `VITE_API_URL not applied` | Vercel needs redeploy after env change; Vite envs are baked at build time. Also ensure var starts with `VITE_`. |
| Free Render sleeps | Free instance spins down → first request ~30s. Use UptimeRobot or upgrade. |

---

## 7) Future Production Hardening (optional)

- Swap SQLite for Postgres (Render Postgres / Neon) + update `database.py` / `api.py`.
- Move Chroma to persistent hosted (Chroma Cloud, Pinecone, Qdrant).
- Add `uvicorn --workers 2` behind gunicorn for concurrency.
- Set `OAUTHLIB_INSECURE_TRANSPORT=0` in prod.
