"""Milestone 2: Gmail Reader

Fetches emails from Gmail via OAuth 2.0, cleans/store them to SQLite.

Usage:
    python fetch_emails.py --max 10 --days 7

Requires:
    - credentials.json (OAuth 2.0 Desktop client from Google Cloud Console)
    - Gmail API enabled in Google Cloud project
"""

import os
import argparse
import base64
import sqlite3
from datetime import datetime, timedelta

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


# Paths
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "token.json")
SCOPES = ["https://mail.google.com/"]
DB_PATH = os.path.join(os.path.dirname(__file__), "emails.db")


def get_gmail_service():
    """Authenticate and return a Gmail API service instance."""
    creds = None

    # Load existing token
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    # If no valid credentials, run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for next run
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def extract_body(payload):
    """Extract plain-text body from email payload."""
    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    elif payload.get("body", {}).get("data"):
        data = payload["body"].get("data", "")
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    return ""


def fetch_recent_emails(service, max_results=None, days_back=7):
    """Fetch emails from the last N days."""
    # Full sync batch 100 (SYNC_BATCH_SIZE env, default 100)
    default_batch = int(os.getenv("SYNC_BATCH_SIZE", "100"))
    if max_results is None:
        max_results = default_batch
    else:
        max_results = min(int(max_results), default_batch)
    now = datetime.utcnow()
    since = now - timedelta(days=days_back)
    since_timestamp = int(datetime.timestamp(since))

    query = f"after:{since_timestamp}"

    results = service.users().messages().list(
        userId="me",
        maxResults=max_results,
        q=query,
    ).execute()

    messages = results.get("messages", [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        payload = msg_data.get("payload", {})
        headers = payload.get("headers", [])

        subject = next(
            (h["value"] for h in headers if h["name"].lower() == "subject"), ""
        )
        from_addr = next(
            (h["value"] for h in headers if h["name"].lower() == "from"), ""
        )
        date = next(
            (h["value"] for h in headers if h["name"].lower() == "date"), ""
        )

        body = extract_body(payload)

        emails.append(
            {
                "id": msg["id"],
                "thread_id": msg_data.get("threadId"),
                "subject": subject,
                "from": from_addr,
                "date": date,
                "body": body,
                "snippet": msg_data.get("snippet", ""),
            }
        )

    return emails


def store_emails(emails):
    """Store cleaned emails into SQLite database."""
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

    for email in emails:
        cursor.execute(
            """
            INSERT OR REPLACE INTO emails (id, thread_id, subject, from_addr, date, body, snippet, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email["id"],
                email.get("thread_id"),
                email["subject"],
                email["from"],
                email["date"],
                email["body"][:5000],
                email["snippet"],
                datetime.utcnow().isoformat(),
            ),
        )

    conn.commit()
    conn.close()
    print(f"Stored {len(emails)} emails to {DB_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch emails from Gmail and store to SQLite"
    )
    parser.add_argument(
        "--max", type=int, default=int(os.getenv("SYNC_BATCH_SIZE", "100")), help="Max emails to fetch"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Fetch emails from last N days"
    )
    args = parser.parse_args()

    print("🔐 Authenticating with Gmail...")
    service = get_gmail_service()

    print(
        f"📧 Fetching recent emails (last {args.days} days, max {args.max})..."
    )
    emails = fetch_recent_emails(service, max_results=args.max, days_back=args.days)

    print(f"💾 Storing {len(emails)} emails to SQLite...")
    store_emails(emails)

    print(f"✅ Done! Fetched {len(emails)} emails.")


if __name__ == "__main__":
    main()