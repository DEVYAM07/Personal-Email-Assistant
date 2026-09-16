import sqlite3
from typing import Any, Dict, List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def fetch_emails(db_path: str = "emails.db") -> List[Dict[str, Any]]:
    """Read all email rows from the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, subject, from_addr, date, body, snippet FROM emails")
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "subject": row[1],
            "from_addr": row[2],
            "date": row[3],
            "body": row[4],
            "snippet": row[5],
        }
        for row in rows
    ]


def init_chroma_client(db_path: str = "./chroma_db") -> chromadb.PersistentClient:
    """Initialize a ChromaDB PersistentClient."""
    return chromadb.PersistentClient(path=db_path)


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Return a SentenceTransformerEmbeddingFunction with the all-MiniLM-L6-v2 model."""
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def upsert_emails(
    client: chromadb.PersistentClient,
    emails: List[Dict[str, Any]],
    collection_name: str = "email_vectors",
) -> None:
    """Format and upsert emails into the ChromaDB collection."""
    embedding_fn = get_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )

    for email in emails:
        doc_text = (
            f"Subject: {email['subject']}\n"
            f"From: {email['from_addr']}\n"
            f"Date: {email['date']}\n\n"
            f"{email['body']}"
        )
        metadata = {
            "subject": email["subject"],
            "from_addr": email["from_addr"],
            "date": email["date"],
            "snippet": email["snippet"],
        }
        collection.upsert(
            ids=[email["id"]],
            documents=[doc_text],
            metadatas=[metadata],
        )


def main(db_path: str = "emails.db", chroma_path: str = "./chroma_db") -> None:
    """Main entry point: fetch emails from SQLite and upsert into ChromaDB."""
    emails = fetch_emails(db_path)

    if not emails:
        print("No emails found in the database. Nothing to upsert.")
        return

    client = init_chroma_client(chroma_path)
    upsert_emails(client, emails)
    print(f"Upserted {len(emails)} emails into ChromaDB collection 'email_vectors'.")


if __name__ == "__main__":
    main()