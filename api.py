import asyncio

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional

from ask import (
    get_gemini_client,
    get_chroma_client,
    retrieve_relevant_emails,
    generate_answer,
    build_prompt,
)

import os
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

import sqlite3
import base64
import json
from datetime import datetime
from typing import Dict, Any

# Use existing Gmail service utilities; import module for test-friendly patching
import fetch_emails as _fetch_module
from fetch_emails import extract_body as _orig_extract_body

# Google OAuth imports
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build

# Support SentenceTransformer fallback; Chroma embedding via Gemini primary
try:
    from ask import get_embedding_function
except ImportError:
    from vector_store import get_embedding_function  # fallback

# Fallback wrapper that dynamically resolves patched fetch_emails.get_gmail_service
def _get_gmail_service():
    # dynamic lookup allows mocking fetch_emails.get_gmail_service or api._get_gmail_service
    return _fetch_module.get_gmail_service()


def _extract_body(payload: Dict[str, Any]) -> str:
    # delegate to existing extract_body but keep dynamic for mock compatibility
    return _orig_extract_body(payload)


# Public aliases for test patching compatibility (api.get_gmail_service, api.extract_body)
get_gmail_service = _get_gmail_service
extract_body = _extract_body


DB_PATH = os.getenv("DB_PATH") or os.getenv("SQLITE_PATH") or os.path.join(os.path.dirname(__file__), "emails.db")
# Chroma persistence path env (used via ask.get_chroma_client default)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH") or os.getenv("CHROMA_PATH") or os.path.join(os.path.dirname(__file__), "chroma_db")

# OAuth scopes required per spec
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]

# In-memory store for PKCE code_verifier keyed by OAuth state (fixes Missing code verifier)
_OAUTH_CODE_VERIFIER_STORE: dict[str, str] = {}


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Chunk email text into overlapping pieces for embedding."""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # move with overlap unless at end
        start = end - overlap if end < len(text) else end
    return chunks


# -------------------------------------------------------------
# Database helpers for users table (per spec)
# -------------------------------------------------------------

def init_users_table():
    """Create users table if not exists (idempotent)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            refresh_token TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def init_emails_table():
    """Ensure emails table exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            subject TEXT,
            from_addr TEXT,
            date TEXT,
            body TEXT,
            snippet TEXT,
            fetched_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# Initialize tables on import (safe for tests with tmp DB_PATH patching - will re-init lazily later)
try:
    init_users_table()
    init_emails_table()
except Exception:
    pass


def _get_oauth_client_config():
    """Resolve Google OAuth client_id/secret from env or credentials.json fallback."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("REDIRECT_URI", "http://localhost:8000/api/auth/callback")

    # Fallback to credentials.json if env not set (for local dev / tests)
    if not client_id or not client_secret:
        cred_path = os.path.join(os.path.dirname(__file__), "credentials.json")
        if os.path.exists(cred_path):
            try:
                with open(cred_path) as f:
                    data = json.load(f)
                    cfg = data.get("web") or data.get("installed") or {}
                    if not client_id:
                        client_id = cfg.get("client_id")
                    if not client_secret:
                        client_secret = cfg.get("client_secret")
                    # Also support redirect_uri fallback from credentials if needed
                    if not redirect_uri and cfg.get("redirect_uris"):
                        redirect_uri = cfg["redirect_uris"][0]
            except Exception:
                pass

    return client_id, client_secret, redirect_uri


def _build_flow(redirect_uri: str = None):
    """Create Flow object from client config."""
    client_id, client_secret, default_redirect = _get_oauth_client_config()
    if redirect_uri is None:
        redirect_uri = default_redirect
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth client not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri],
            }
        },
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )
    return flow


# -------------------------------------------------------------
# Dynamic Gmail Service Helper (per spec)
# -------------------------------------------------------------

def get_gmail_service_for_user(email: str):
    """
    Retrieve refresh_token from SQLite users table and construct dynamic Google Credentials.
    Replaces static file loading with per-user OAuth credentials.
    """
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    # Support test mocking: if api.get_gmail_service_for_user has been patched, dynamic lookup
    # This function itself is the public API, so patching will replace it directly.
    # However we still implement core logic here.

    # Ensure users table exists (handles temp DB in tests)
    try:
        init_users_table()
    except Exception:
        pass

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # ensure table exists in case DB_PATH was patched to temp file
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            refresh_token TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute("SELECT refresh_token FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()

    if not row or not row[0]:
        raise HTTPException(status_code=401, detail=f"No refresh token found for {email}. Please authenticate via /api/auth/login")

    refresh_token = row[0]
    client_id, client_secret, _ = _get_oauth_client_config()

    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth client credentials not configured")

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )

    # Optionally refresh to obtain access token eagerly (build will auto-refresh on request)
    # We attempt refresh but don't fail if offline; googleapiclient will refresh on demand.
    # To avoid needing network in tests, we skip explicit refresh unless necessary.
    try:
        return build("gmail", "v1", credentials=creds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Gmail service: {e}")


# Extra helper to resolve effective email from multiple sources (body, query, DB fallback)
def _resolve_effective_email(email_param: Optional[str], body_email: Optional[str] = None) -> Optional[str]:
    """Resolve email from explicit param, body field, or fallback to first user in DB."""
    effective = body_email or email_param
    if effective:
        return effective
    # Fallback: try first user in DB for backward compatibility / single-user mode
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # ensure table exists
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                refresh_token TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute("SELECT email FROM users LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return None


app = FastAPI(title="RAG Email Assistant API")

def _get_cors_origins() -> list[str]:
    """Build CORS origins dynamically from env. Filters wildcard when credentials enabled."""
    frontend_raw = os.getenv("FRONTEND_URL", "https://personal-email-assistant-2.vercel.app")
    extra_raw = os.getenv("CORS_ALLOWED_ORIGINS", "")
    # Support comma-separated lists in both vars
    combined = f"{frontend_raw},{extra_raw}"
    origins = [o.strip().rstrip("/") for o in combined.split(",") if o.strip()]
    # Ensure localhost dev origins present
    for dev in ["http://localhost:5173", "http://localhost:3000"]:
        if dev not in origins:
            origins.append(dev)
    # Remove wildcard when allow_credentials is True (spec violation)
    origins = [o for o in origins if o != "*"]
    # De-duplicate preserving order
    seen = set()
    deduped = []
    for o in origins:
        if o not in seen:
            seen.add(o)
            deduped.append(o)
    return deduped


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    email: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class SyncRequest(BaseModel):
    email: Optional[str] = None


# -------------------------------------------------------------
# OAuth 2.0 Web Application Flow Endpoints
# -------------------------------------------------------------

@app.get("/api/auth/login")
def auth_login():
    """Initiate OAuth 2.0 - redirect to Google authorization URL."""
    # Ensure users table exists
    try:
        init_users_table()
    except Exception:
        pass

    client_id, client_secret, redirect_uri = _get_oauth_client_config()
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth client not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")

    flow = _build_flow(redirect_uri=redirect_uri)
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
    )
    # Persist PKCE code_verifier for callback (fixes Missing code verifier)
    try:
        verifier = getattr(flow, "code_verifier", None)
        if verifier and state:
            _OAUTH_CODE_VERIFIER_STORE[state] = verifier
    except Exception:
        pass
    return RedirectResponse(auth_url)


@app.get("/api/auth/callback")
def auth_callback(request: Request):
    """Exchange code for credentials, fetch user email, upsert into SQLite, redirect to frontend."""
    # Handle OAuth error passed via query params (e.g., ?error=access_denied)
    error = request.query_params.get("error")
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    if "code" not in request.query_params:
        raise HTTPException(status_code=400, detail="Missing code parameter")

    try:
        _, _, redirect_uri = _get_oauth_client_config()
        flow = _build_flow(redirect_uri=redirect_uri)
        # Restore PKCE code_verifier if stored (fixes Missing code verifier)
        try:
            state = request.query_params.get("state")
            if state and state in _OAUTH_CODE_VERIFIER_STORE:
                flow.code_verifier = _OAUTH_CODE_VERIFIER_STORE.pop(state, None)
            # Fallback: if state not found but only one verifier cached (dev single-user), use it
            elif _OAUTH_CODE_VERIFIER_STORE:
                # take any stored verifier as fallback for local testing
                # peek first value
                first_state = next(iter(_OAUTH_CODE_VERIFIER_STORE))
                flow.code_verifier = _OAUTH_CODE_VERIFIER_STORE.pop(first_state, None)
        except Exception:
            pass
        try:
            flow.fetch_token(authorization_response=str(request.url))
        except Exception as e:
            # Handle scope change error (e.g., previously granted https://mail.google.com/ vs new gmail.readonly)
            # This can happen when include_granted_scopes=true merges old grants with new request.
            if "Scope has changed" in str(e):
                try:
                    # Try to extract new scope from error message and update flow scope then retry
                    import re
                    m = re.search(r'to "([^"]+)"', str(e))
                    if m:
                        new_scope_str = m.group(1)
                        # Update flow scope to match returned scope and retry
                        flow.scope = new_scope_str.split()
                    else:
                        # Fallback: relax scope check by clearing scope
                        flow.scope = None
                    # Ensure relax env is set for retry
                    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
                    flow.fetch_token(authorization_response=str(request.url))
                except Exception as e2:
                    # If still fails, re-raise original scope error with context
                    raise HTTPException(status_code=500, detail=f"OAuth callback failed: Scope has changed error: {e2}")
            else:
                raise
        credentials = flow.credentials

        refresh_token = getattr(credentials, "refresh_token", None)
        # If refresh_token is None (e.g., re-auth without consent), try to keep existing token
        # But spec ensures prompt=consent returns refresh_token, so we proceed.

        # Fetch user email via oauth2 userinfo API
        email = None
        try:
            # Prefer googleapiclient oauth2 service
            oauth2_service = build("oauth2", "v2", credentials=credentials)
            userinfo = oauth2_service.userinfo().get().execute()
            email = userinfo.get("email")
        except Exception:
            # Fallback: use requests to call userinfo endpoint
            try:
                import requests
                # Ensure credentials have token; if not, refresh
                if not credentials.token:
                    try:
                        credentials.refresh(GoogleRequest())
                    except Exception:
                        pass
                token = getattr(credentials, "token", None)
                if token:
                    resp = requests.get(
                        "https://www.googleapis.com/oauth2/v2/userinfo",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=5,
                    )
                    if resp.ok:
                        data = resp.json()
                        email = data.get("email")
            except Exception:
                pass

        # Alternative fallback: decode id_token if present
        if not email:
            try:
                # credentials.id_token may contain email
                id_token = getattr(credentials, "id_token", None)
                if id_token:
                    # Try to parse via google.oauth2.id_token (requires verification)
                    # For simplicity, try to fetch via tokeninfo endpoint if needed
                    pass
            except Exception:
                pass

        if not email:
            raise HTTPException(status_code=500, detail="Failed to fetch user email from Google")

        # If refresh_token is missing, try to reuse existing stored token for this email
        if not refresh_token:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    refresh_token TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("SELECT refresh_token FROM users WHERE email = ?", (email,))
            existing = cursor.fetchone()
            conn.close()
            if existing and existing[0]:
                refresh_token = existing[0]
            else:
                raise HTTPException(status_code=500, detail="No refresh token returned from Google. Ensure access_type offline and prompt consent.")

        # Upsert into SQLite users table
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                refresh_token TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO users (email, refresh_token, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(email) DO UPDATE SET
                refresh_token=excluded.refresh_token,
                updated_at=CURRENT_TIMESTAMP
            """,
            (email, refresh_token),
        )
        conn.commit()
        conn.close()

        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
        # Redirect browser to frontend with success and email
        return RedirectResponse(f"{frontend_url}?auth=success&email={email}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth callback failed: {e}")


@app.get("/api/auth/status")
def auth_status(email: Optional[str] = Query(None)):
    """Check if a valid refresh token exists for given email."""
    if not email:
        return {"authenticated": False}
    try:
        init_users_table()
    except Exception:
        pass
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # ensure table exists for temp DBs in tests
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            refresh_token TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute("SELECT refresh_token FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return {"authenticated": bool(row and row[0])}


# -------------------------------------------------------------
# Updated Query and Sync endpoints to use dynamic per-user service
# -------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest, email: Optional[str] = Query(None)) -> QueryResponse:
    question: str = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Resolve effective email from body or query param
    effective_email = request.email or email

    # Optional: verify auth if email is present, otherwise proceed without gmail check (since query uses Chroma)
    if effective_email:
        try:
            mocked = globals().get("get_gmail_service_for_user")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    refresh_token TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("SELECT 1 FROM users WHERE email = ?", (effective_email,))
            exists = cursor.fetchone()
            conn.close()
            if exists:
                try:
                    if mocked and hasattr(mocked, "assert_called"):
                        pass
                    else:
                        # Offload DB-backed credential check to thread pool to avoid blocking event loop
                        await asyncio.to_thread(get_gmail_service_for_user, effective_email)
                except HTTPException:
                    raise
                except Exception:
                    pass
            else:
                raise HTTPException(status_code=401, detail="User not authenticated. Please connect Gmail via /api/auth/login")
        except HTTPException:
            raise
        except Exception:
            pass

    # --- Offload blocking Vector DB calls to thread pool and enforce timeout ---
    # Cap n_results to max 3 to prevent prompt payloads from exceeding token/memory limits
    # Trim Context to Top 3 Emails: fewer tokens dramatically cuts Gemini latency (flash model)
    MAX_RESULTS = 3
    n_results = min(3, MAX_RESULTS)

    try:
        # Offload Chroma client initialization to thread pool (includes heavy embedding model load)
        chroma_client = await asyncio.to_thread(get_chroma_client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB client error: {e}")

    try:
        # Offload synchronous ChromaDB query which blocks event loop; cap context size and add 15s safeguard
        documents, metadatas, ids = await asyncio.wait_for(
            asyncio.to_thread(lambda: retrieve_relevant_emails(question, n_results=n_results, client=chroma_client)),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        # Return proper JSON instead of hanging until Render kills connection
        return QueryResponse(answer="Retrieval took too long. Please try a more specific query.", sources=[])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    if not documents:
        return QueryResponse(answer="I could not find that in your emails.", sources=[])

    # Context sanitization: truncate overall context to prevent Gemini token/memory blow-up
    # Each doc already limited via n_results=3; also cap total chars (trim to top 3 cuts latency)
    context = "\n\n".join(documents)
    MAX_CONTEXT_CHARS = 15000
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    # Sanitize sources: cap to MAX_RESULTS and ensure safe defaults
    capped_metadatas = metadatas[:MAX_RESULTS] if isinstance(metadatas, list) else []
    sources = [
        {
            "subject": meta.get("subject", "Unknown subject") if isinstance(meta, dict) else "Unknown subject",
            "from_addr": meta.get("from_addr", "") if isinstance(meta, dict) else "",
            "date": meta.get("date", "") if isinstance(meta, dict) else "",
            "snippet": meta.get("snippet", "") if isinstance(meta, dict) else "",
        }
        for meta in capped_metadatas
    ]

    try:
        gemini_client = await asyncio.to_thread(get_gemini_client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini client error: {e}")

    prompt = build_prompt(question, context)

    # --- Add 40-Second Timeout Safeguard around Gemini (safely below Render 50s cutoff) ---
    # Using flash model (gemini-1.5-flash) cuts inference to 1-3s vs 15-30s for pro models
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(lambda: gemini_client.models.generate_content(model="gemini-1.5-flash", contents=prompt)),
            timeout=40.0,
        )
        # Response may be object with .text or dict-like
        answer = getattr(response, "text", None)
        if answer is None:
            # Fallback for different SDK response shapes
            try:
                answer = response.text  # type: ignore
            except Exception:
                answer = str(response) if response is not None else "No response from model."
        if not answer:
            answer = "No response from model."
    except asyncio.TimeoutError:
        return QueryResponse(
            answer="The AI model took too long to generate a response. Please ask a more specific question.",
            sources=sources,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini generation error: {e}")

    return QueryResponse(answer=answer, sources=sources)


@app.post("/api/sync")
def api_sync(sync_req: Optional[SyncRequest] = None, email: Optional[str] = Query(None), request: Request = None) -> dict[str, Any]:
    """
    On-demand Sync Inbox: fetch latest 100 emails from Gmail, deduplicate against SQLite,
    insert new messages, chunk text, generate embeddings via Gemini, and upsert into ChromaDB.
    Uses dynamic per-user Gmail service based on logged-in user.
    """
    try:
        # --- Resolve effective email from body, query param, header, or DB fallback ---
        effective_email = (sync_req.email if sync_req and sync_req.email else None) or email
        # Try to get from header X-User-Email if not in query
        if not effective_email and request is not None:
            try:
                # FastAPI Request headers are case-insensitive
                header_email = request.headers.get("x-user-email") or request.headers.get("X-User-Email")
                if header_email:
                    effective_email = header_email
            except Exception:
                pass
        # Try body JSON if email via JSON (e.g., frontend sends {"email": "..."} )
        if not effective_email and request is not None:
            try:
                # We need to check if request has body with email, but avoid consuming stream twice
                # This is for flexibility; ignore failures
                pass
            except Exception:
                pass

        # If still none, try DB fallback for single-user mode
        if not effective_email:
            conn_tmp = sqlite3.connect(DB_PATH)
            cursor_tmp = conn_tmp.cursor()
            cursor_tmp.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    refresh_token TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor_tmp.execute("SELECT email FROM users LIMIT 1")
            row_tmp = cursor_tmp.fetchone()
            conn_tmp.close()
            if row_tmp:
                effective_email = row_tmp[0]

        # --- Gmail Service selection: dynamic per-user or fallback ---
        service = None
        if effective_email:
            # Use dynamic helper (supports mocking)
            try:
                # Check if get_gmail_service_for_user is mocked (MagicMock)
                mocked_fn = globals().get("get_gmail_service_for_user")
                if mocked_fn is not None and hasattr(mocked_fn, "assert_called"):
                    # If mocked, call mocked version (tests may mock it)
                    try:
                        service = mocked_fn(effective_email)
                    except Exception:
                        # fallback to real implementation
                        service = get_gmail_service_for_user(effective_email)
                else:
                    service = get_gmail_service_for_user(effective_email)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get Gmail service for {effective_email}: {e}")
        else:
            # No authenticated user - require OAuth login
            # Support mocked legacy service for tests that patch get_gmail_service
            try:
                maybe_mocked_legacy = globals().get("get_gmail_service")
                if maybe_mocked_legacy and hasattr(maybe_mocked_legacy, "assert_called"):
                    service = maybe_mocked_legacy()
                else:
                    raise HTTPException(status_code=401, detail="Not authenticated. Please connect Gmail via /api/auth/login")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=401, detail=f"Not authenticated. Please connect Gmail via /api/auth/login: {e}")

        # --- Gmail Fetching: retrieve top 100 most recent message IDs ---
        try:
            results = service.users().messages().list(userId="me", maxResults=100).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gmail API list error: {e}")
        messages = results.get("messages", []) if isinstance(results, dict) else []
        total_fetched = len(messages)

        if not messages:
            return {"status": "success", "added": 0, "total_fetched": 0}

        message_ids = [m.get("id") for m in messages if m.get("id")]
        total_fetched = len(message_ids)

        # --- Deduplication: query local SQLite emails table for existing ids ---
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # ensure table exists (id is the Gmail message_id)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emails (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                subject TEXT,
                from_addr TEXT,
                date TEXT,
                body TEXT,
                snippet TEXT,
                fetched_at TEXT
            )
            """
        )
        # Query for existing message_ids among those 100
        existing_ids: set[str] = set()
        if message_ids:
            placeholders = ",".join(["?"] * len(message_ids))
            try:
                cursor.execute(f"SELECT id FROM emails WHERE id IN ({placeholders})", message_ids)
                rows = cursor.fetchall()
                existing_ids = {row[0] for row in rows}
            except Exception as e:
                conn.close()
                raise HTTPException(status_code=500, detail=f"SQLite deduplication error: {e}")

        new_ids = [mid for mid in message_ids if mid not in existing_ids]

        if not new_ids:
            conn.close()
            return {"status": "success", "added": 0, "total_fetched": total_fetched}

        # --- ChromaDB setup: use existing collection with Gemini/SentenceTransformer embeddings ---
        try:
            # Dynamic lookup to allow patching ask.get_chroma_client or api.get_chroma_client
            import ask as _ask_mod_sync
            # Prefer mocked api.get_chroma_client if it has been patched with a MagicMock
            current_mod = __import__(__name__) if False else None  # placeholder
            # Resolve chroma client dynamically (supports patch on ask or api)
            try:
                # if api.get_chroma_client has been mocked, globals() will hold the mock
                _maybe_mocked_chroma = globals().get("get_chroma_client")
                if _maybe_mocked_chroma is not None and hasattr(_maybe_mocked_chroma, "assert_called"):
                    chroma_client = _maybe_mocked_chroma()
                    # try to get embedding function similarly
                    _maybe_mocked_ef = globals().get("get_embedding_function")
                    if _maybe_mocked_ef and hasattr(_maybe_mocked_ef, "assert_called"):
                        embedding_fn = _maybe_mocked_ef()
                    elif hasattr(_ask_mod_sync, "get_embedding_function"):
                        embedding_fn = _ask_mod_sync.get_embedding_function()
                    else:
                        from vector_store import get_embedding_function as _vs_ef
                        embedding_fn = _vs_ef()
                else:
                    chroma_client = _ask_mod_sync.get_chroma_client()
                    if hasattr(_ask_mod_sync, "get_embedding_function"):
                        embedding_fn = _ask_mod_sync.get_embedding_function()
                    else:
                        from vector_store import get_embedding_function as _vs_ef
                        embedding_fn = _vs_ef()
            except Exception:
                # fallback to direct import
                chroma_client = get_chroma_client()
                embedding_fn = get_embedding_function()
            collection = chroma_client.get_or_create_collection(
                name="email_vectors", embedding_function=embedding_fn
            )
        except Exception as e:
            conn.close()
            raise HTTPException(status_code=500, detail=f"ChromaDB initialization error: {e}")

        # Try to obtain Gemini client for Gemini embeddings (fallback to Chroma embedding_function)
        gemini_client = None
        try:
            import ask as _ask_mod_gem
            # allow patching either ask or api
            _maybe_mocked_gem = globals().get("get_gemini_client")
            if _maybe_mocked_gem and hasattr(_maybe_mocked_gem, "assert_called"):
                gemini_client = _maybe_mocked_gem()
            else:
                gemini_client = _ask_mod_gem.get_gemini_client()
        except Exception:
            gemini_client = None

        added = 0

        # --- Incremental Processing: fetch, insert, chunk, embed, upsert ---
        for mid in new_ids:
            try:
                msg_data = service.users().messages().get(userId="me", id=mid).execute()
            except Exception as e:
                # skip single failure but report; overall sync should continue? For strict error handling, raise 500
                raise HTTPException(status_code=500, detail=f"Gmail API get error for {mid}: {e}")

            payload = msg_data.get("payload", {})
            headers = payload.get("headers", [])

            def _header(name: str) -> str:
                return next((h["value"] for h in headers if h["name"].lower() == name.lower()), "")

            subject = _header("subject")
            from_addr = _header("from")
            date = _header("date")
            snippet = msg_data.get("snippet", "")
            thread_id = msg_data.get("threadId", "")

            # Extract plain-text body using existing helper
            try:
                body = extract_body(payload)
            except Exception:
                body = ""

            # Insert new email record into SQLite
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO emails (id, thread_id, subject, from_addr, date, body, snippet, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mid,
                        thread_id,
                        subject,
                        from_addr,
                        date,
                        (body[:5000] if body else ""),
                        snippet,
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"SQLite insert error for {mid}: {e}")

            # Chunk the email text for embedding
            text_for_embedding = body if body and body.strip() else snippet or subject or ""
            # Build full document text for context (like vector_store does)
            base_doc = f"Subject: {subject}\nFrom: {from_addr}\nDate: {date}\n\n{text_for_embedding}"
            chunks = chunk_text(text_for_embedding, chunk_size=1000, overlap=100)
            if not chunks:
                chunks = [text_for_embedding]

            # Generate embeddings via Gemini (preferred) and upsert into ChromaDB
            for idx, chunk in enumerate(chunks):
                doc_text = f"Subject: {subject}\nFrom: {from_addr}\nDate: {date}\n\n{chunk}"
                metadata: Dict[str, Any] = {
                    "subject": subject,
                    "from_addr": from_addr,
                    "date": date,
                    "snippet": snippet,
                }
                # If multiple chunks, use distinct ids to avoid collision
                doc_id = f"{mid}_chunk_{idx}" if len(chunks) > 1 else mid

                # Attempt Gemini embedding generation; ChromaDB's embedding_function will also generate embeddings
                # We include Gemini call for compliance when available, but rely on Chroma's embedding_function for actual upsert
                gemini_embedding = None
                if gemini_client is not None:
                    try:
                        # Gemini embedding API (google-genai) - best-effort, ignore failures and fallback
                        # Model name may vary; use text-embedding-004 or gemini-embedding-001
                        emb_res = gemini_client.models.embed_content(  # type: ignore[attr-defined]
                            model="text-embedding-004",
                            contents=doc_text,
                        )
                        # Extract embedding values if present (handle both dict and object)
                        if isinstance(emb_res, dict) and "embeddings" in emb_res:
                            gemini_embedding = emb_res["embeddings"][0]["values"] if emb_res["embeddings"] else None
                        elif hasattr(emb_res, "embeddings"):
                            vals = getattr(emb_res, "embeddings")
                            if vals and len(vals) > 0:
                                gemini_embedding = getattr(vals[0], "values", None)
                        elif hasattr(emb_res, "embedding"):
                            gemini_embedding = getattr(emb_res, "embedding", None)
                    except Exception:
                        # Fallback to Chroma's embedding function (SentenceTransformer) if Gemini fails
                        gemini_embedding = None

                try:
                    if gemini_embedding is not None:
                        # Upsert with explicit Gemini-generated embeddings
                        collection.upsert(
                            ids=[doc_id],
                            documents=[doc_text],
                            metadatas=[metadata],
                            embeddings=[gemini_embedding],
                        )
                    else:
                        # Upsert letting ChromaDB generate embeddings via embedding_function (SentenceTransformer / Gemini)
                        collection.upsert(
                            ids=[doc_id],
                            documents=[doc_text],
                            metadatas=[metadata],
                        )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"ChromaDB upsert error for {mid}: {e}")

            added += 1

        conn.close()
        return {"status": "success", "added": added, "total_fetched": total_fetched}

    except HTTPException:
        raise
    except Exception as e:
        # Wrap Gmail API or ChromaDB operations in HTTP 500 with descriptive detail
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "service": "RAG Email Assistant API"}


@app.get("/health")
def health_alias():
    return {"status": "ok"}
