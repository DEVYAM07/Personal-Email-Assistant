import os
import sys
from typing import List, Tuple, Optional

from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.genai as genai

load_dotenv()


def get_chroma_client(path: str = "./chroma_db") -> chromadb.PersistentClient:
    """Initialize and return a ChromaDB PersistentClient."""
    return chromadb.PersistentClient(path=path)


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Return a SentenceTransformerEmbeddingFunction with all-MiniLM-L6-v2."""
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def get_gemini_client() -> genai.Client:
    """Initialize and return the Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=api_key)


def retrieve_relevant_emails(
    query: str,
    n_results: int = 3,
    client: Optional[chromadb.PersistentClient] = None,
) -> Tuple[List[str], List[dict], List[str]]:
    """Query ChromaDB for the top matching email documents and metadata.

    Returns a tuple of (documents, metadatas, ids).
    """
    if not query or not query.strip():
        print("⚠️ Warning: Empty query provided.", file=sys.stderr)
        return [], [], []

    chroma_client = client or get_chroma_client()
    embedding_fn = get_embedding_function()
    collection = chroma_client.get_collection(
        name="email_vectors", embedding_function=embedding_fn
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    documents: List[str] = results.get("documents", [[]])[0] if results.get("documents") else []
    metadatas: List[dict] = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    ids: List[str] = results.get("ids", [[]])[0] if results.get("ids") else []

    return documents, metadatas, ids


def build_prompt(query: str, context: str) -> str:
    """Build the structured prompt for Gemini with the email context."""
    return (
        "You are an AI email assistant. Answer the user's question using ONLY the retrieved email context provided. "
        "If the answer is not in the context, say 'I could not find that in your emails.'\n\n"
        f"User question: {query}\n\n"
        f"Email context:\n{context}\n\n"
        "Answer:"
    )


def generate_answer(query: str, context: str) -> Optional[str]:
    """Call gemini-3.6-flash with the structured prompt and return the answer."""
    if not context or context.strip() == "":
        return "I could not find that in your emails."

    client = get_gemini_client()
    prompt = build_prompt(query, context)

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"❌ Error generating answer from Gemini: {e}", file=sys.stderr)
        return None


def format_subjects(metadatas: List[dict]) -> str:
    """Format email subjects from metadata for display."""
    if not metadatas:
        return "No matching emails found."
    lines = []
    for i, meta in enumerate(metadatas, 1):
        subject = meta.get("subject", "Unknown subject")
        lines.append(f"{i}. {subject}")
    return "\n".join(lines)


def main():
    """Interactive CLI loop for querying emails via RAG with Gemini."""
    print("=" * 60)
    print("RAG Email Assistant (type 'exit' to quit)")
    print("=" * 60)

    while True:
        query = input("\n🔍 Enter your question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        if not query:
            print("⚠️ Please enter a non-empty question.")
            continue

        documents, metadatas, ids = retrieve_relevant_emails(query, n_results=3)

        if not documents:
            print("⚠️ No matching emails found in the database.")
            continue

        print("\n📧 Relevant email subjects:")
        print(format_subjects(metadatas))

        # Combine all document texts into a single context string
        context = "\n\n".join(documents)

        print("\n🔎 Generating answer from Gemini...")
        answer = generate_answer(query, context)

        if answer is None:
            print("❌ Failed to generate answer.")
            continue

        print("\n💡 Gemini's answer:")
        print(answer)


if __name__ == "__main__":
    main()