import os
import sys
from fastapi import FastAPI, BackgroundTasks
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

# --- Task: Async Background Processing for 512MB (fix 3/15 stall) ---
# Use FastAPI BackgroundTasks to return 202 immediately, avoid 30s polling timeout
def run_email_sync_pipeline():
    """Background sync pipeline - delegates to optimized sync.py (batch Gemini)."""
    try:
        # Import here to avoid circular
        from sync import sync_emails_batch
        import sqlite3
        DB_PATH = os.getenv("DB_PATH") or os.path.join(os.path.dirname(__file__), "emails.db")
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, subject, from_addr, date, body, snippet FROM emails LIMIT 15")
        rows = cur.fetchall()
        conn.close()
        if rows:
            emails = [
                {"id": r[0], "subject": r[1], "from_addr": r[2], "date": r[3], "body": r[4], "snippet": r[5]}
                for r in rows
            ]
            sync_emails_batch(emails)
    except Exception as e:
        print(f"Background sync error: {e}")


@app.post("/api/sync")
async def sync_emails(background_tasks: BackgroundTasks):
    # Trigger background processing without blocking HTTP response
    background_tasks.add_task(run_email_sync_pipeline)
    return {"status": "started", "message": "Syncing emails in background"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
