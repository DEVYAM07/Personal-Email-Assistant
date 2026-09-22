# Personal Email Assistant — RAG + Gmail OAuth

Backend: FastAPI (`api.py` / `main.py` on `0.0.0.0:$PORT`) → Render (512MB, free)
Frontend: Vite + React → Vercel

## Entry Points
- `main.py` — primary entry, re-exports `api.py:623` `app` and handles dynamic `PORT` (Render injects `PORT`, defaults `8000`)
- `api.py:623` — core FastAPI with `/api/health`, `/api/auth/*`, `/api/sync` (202 background), `/api/query`, dynamic CORS via `FRONTEND_URL`
- CORS: `os.environ.get("FRONTEND_URL", "http://localhost:5173")` + `http://localhost:5173` (no hardcoded Vercel/Render URLs; inject via env)

## Deploy

See `DEPLOYMENT.md` for Render + Vercel guide. Summary:
1. Render → New Web Service → connect repo, `pip install -r requirements.txt`, `uvicorn main:app --host 0.0.0.0 --port $PORT`, health `/api/health`
2. Set Render env: `GEMINI_API_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `REDIRECT_URI` (your Render URL `/api/auth/callback`), `FRONTEND_URL` (your Vercel URL), `SYNC_BATCH_SIZE=100`
3. Google Console → OAuth Client → add redirect `REDIRECT_URI` + origin `FRONTEND_URL`
4. Vercel → import `frontend/`, set `VITE_API_URL` (your Render URL) → redeploy

## Local Dev
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000  # or api:app
# env-driven URLs:
# FRONTEND_URL=http://localhost:5173 REDIRECT_URI=http://localhost:8000/api/auth/callback
cd frontend && echo "VITE_API_URL=http://localhost:8000" > .env.local && npm run dev
```

## Sync & Memory
- `SYNC_BATCH_SIZE` env default `100` in `api.py:203` and `fetch_emails.py:69` (full sync, remote Gemini embeddings keep RAM low)
- Background `202 Accepted` on `POST /api/sync` (poll `GET /api/sync/status`) avoids proxy timeouts
- Remote Gemini embeddings `models/text-embedding-004` (`api.py:363`) keeps ChromaDB off ONNX/PyTorch for low RAM

