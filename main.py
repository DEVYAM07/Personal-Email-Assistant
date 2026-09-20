"""
Main entry point for Render deployment compatibility.
Re-exports FastAPI app from api.py with correct CORS configuration.

CORS configuration (must match api.py):
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://personal-email-assistant-2.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "*"  # Fallback wildcard for staging/preview deployments
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)
"""

from api import app  # noqa: F401

# Ensure app is correctly configured for CORS if imported via main:app
# The actual CORS middleware is configured in api.py; this re-export ensures
# both api:app and main:app work with Render's startCommand variations.
