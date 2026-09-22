import os
import sys
from typing import List, Tuple, Optional

from dotenv import load_dotenv

import chromadb
import google.genai as genai

load_dotenv()

# Module-level caches to avoid re-loading heavy clients/models per request (critical for 512MB Render free tier)
_CHROMA_CLIENT_CACHE: Optional[chromadb.PersistentClient] = None
_CHROMA_CLIENT_PATH: Optional[str] = None
_GEMINI_CLIENT_CACHE: Optional[genai.Client] = None
_EMBEDDING_FN_CACHE = None
_EMBEDDING_FN_NAME: Optional[str] = None


def get_chroma_client(path: str = None) -> chromadb.PersistentClient:
    """Initialize and return a cached ChromaDB PersistentClient (512MB-friendly)."""
    global _CHROMA_CLIENT_CACHE, _CHROMA_CLIENT_PATH
    if path is None:
        path = os.getenv("CHROMA_DB_PATH") or os.getenv("CHROMA_PATH") or "./chroma_db"
    # Reuse cached client if same path
    if _CHROMA_CLIENT_CACHE is not None and _CHROMA_CLIENT_PATH == path:
        return _CHROMA_CLIENT_CACHE
    # Optional warm cache invalidation if path changes
    client = chromadb.PersistentClient(path=path)
    _CHROMA_CLIENT_CACHE = client
    _CHROMA_CLIENT_PATH = path
    return client


def get_embedding_function(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load SentenceTransformerEmbeddingFunction (avoids torch load at import)."""
    global _EMBEDDING_FN_CACHE, _EMBEDDING_FN_NAME
    if _EMBEDDING_FN_CACHE is not None and _EMBEDDING_FN_NAME == model_name:
        return _EMBEDDING_FN_CACHE
    # Lazy import to avoid importing torch/sentence_transformers at startup when Gemini is primary
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
    _EMBEDDING_FN_CACHE = fn
    _EMBEDDING_FN_NAME = model_name
    return fn


def _try_get_gemini_client_silent() -> Optional[genai.Client]:
    """Try to get Gemini client without sys.exit - for embedding fallback."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        # Use cached
        global _GEMINI_CLIENT_CACHE
        if _GEMINI_CLIENT_CACHE is not None:
            return _GEMINI_CLIENT_CACHE
        client = genai.Client(api_key=api_key)
        _GEMINI_CLIENT_CACHE = client
        return client
    except Exception:
        return None


def _embed_query_gemini(query: str, gemini_client: Optional[genai.Client] = None) -> Optional[List[float]]:
    """Embed query via Gemini text-embedding-004; returns vector or None on failure."""
    try:
        client = gemini_client or _try_get_gemini_client_silent()
        if client is None:
            return None
        # genai SDK: embed_content with model text-embedding-004
        res = client.models.embed_content(model="text-embedding-004", contents=query)  # type: ignore[attr-defined]
        # Handle multiple SDK response shapes
        if isinstance(res, dict) and "embeddings" in res:
            vals = res["embeddings"]
            if vals and isinstance(vals, list):
                first = vals[0]
                if isinstance(first, dict) and "values" in first:
                    return first["values"]
                if hasattr(first, "values"):
                    return getattr(first, "values")
        if hasattr(res, "embeddings"):
            vals = getattr(res, "embeddings")
            if vals and len(vals) > 0:
                first = vals[0]
                if hasattr(first, "values"):
                    return getattr(first, "values")
                if isinstance(first, dict) and "values" in first:
                    return first["values"]
        if hasattr(res, "embedding"):
            emb = getattr(res, "embedding")
            if isinstance(emb, dict) and "values" in emb:
                return emb["values"]
            if hasattr(emb, "values"):
                return getattr(emb, "values")
            if isinstance(emb, list):
                return emb
        # Some SDKs return .embeddings[0].values directly
        # Fallback: try to extract from res directly if it's a list
        if isinstance(res, list) and len(res) > 0:
            return res[0]  # type: ignore
    except Exception as e:
        print(f"⚠️ Gemini embed_content failed, falling back to ST: {e}", file=sys.stderr)
    return None


def get_gemini_client() -> genai.Client:
    """Initialize and return the Gemini client (cached for 512MB efficiency)."""
    global _GEMINI_CLIENT_CACHE
    if _GEMINI_CLIENT_CACHE is not None:
        return _GEMINI_CLIENT_CACHE
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    _GEMINI_CLIENT_CACHE = client
    return client


def retrieve_relevant_emails(
    query: str,
    n_results: int = 3,
    client: Optional[chromadb.PersistentClient] = None,
    gemini_client: Optional[genai.Client] = None,
) -> Tuple[List[str], List[dict], List[str]]:
    """Query ChromaDB for the top matching email documents and metadata.

    Returns a tuple of (documents, metadatas, ids).
    Caps n_results to max 3 to prevent prompt payloads from exceeding token/memory limits.
    Trim Context to Top 3 Emails: fewer tokens dramatically cuts Gemini latency.
    Prefers Gemini text-embedding-004 for query (avoids loading 200MB+ ST model on 512MB free tier).
    """
    if not query or not query.strip():
        print("⚠️ Warning: Empty query provided.", file=sys.stderr)
        return [], [], []

    # Sanitize n_results: cap to 3 as per bug fix (Render timeout safeguard + latency cut)
    try:
        n_results = int(n_results)
    except Exception:
        n_results = 3
    n_results = max(1, min(n_results, 3))

    chroma_client = client or get_chroma_client()

    # Prefer Gemini embeddings (512MB-friendly, matches sync's primary path)
    # This avoids loading SentenceTransformer/torch which spikes RSS by ~230MB
    query_embedding = _embed_query_gemini(query, gemini_client=gemini_client)

    if query_embedding is not None:
        # Query via precomputed embedding - no embedding_function needed (avoids ST load)
        try:
            try:
                collection = chroma_client.get_collection(name="email_vectors")
            except Exception:
                # Fallback if collection needs embedding_function for metadata
                collection = chroma_client.get_or_create_collection(name="email_vectors")
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
        except Exception as e:
            # Handle dimension mismatch (existing 384-dim ST vectors vs new 768-dim Gemini)
            # Fallback to ST query so existing DB still works, avoiding 500
            err_msg = str(e).lower()
            if "dimension" in err_msg or "expecting embedding" in err_msg:
                print(f"⚠️ Gemini query dimension mismatch ({e}), falling back to ST", file=sys.stderr)
                try:
                    embedding_fn = get_embedding_function()
                    collection = chroma_client.get_collection(
                        name="email_vectors", embedding_function=embedding_fn
                    )
                    results = collection.query(
                        query_texts=[query],
                        n_results=n_results,
                    )
                except Exception as e2:
                    # If collection empty or not found, try or_create
                    try:
                        embedding_fn = get_embedding_function()
                        collection = chroma_client.get_or_create_collection(
                            name="email_vectors", embedding_function=embedding_fn
                        )
                        results = collection.query(
                            query_texts=[query],
                            n_results=n_results,
                        )
                    except Exception:
                        raise e2
            else:
                raise
    else:
        # Fallback: lazy ST (only if GEMINI_API_KEY missing or embed fails)
        try:
            embedding_fn = get_embedding_function()
            collection = chroma_client.get_collection(
                name="email_vectors", embedding_function=embedding_fn
            )
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
            )
        except Exception as e:
            # Handle missing collection
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                embedding_fn = get_embedding_function()
                collection = chroma_client.get_or_create_collection(
                    name="email_vectors", embedding_function=embedding_fn
                )
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                )
            else:
                raise

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


GEMINI_FLASH_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("GEMINI_FLASH_MODEL") or "gemini-2.0-flash"


def generate_answer(query: str, context: str) -> Optional[str]:
    """Call Gemini flash with the structured prompt and return the answer."""
    if not context or context.strip() == "":
        return "I could not find that in your emails."

    client = get_gemini_client()
    prompt = build_prompt(query, context)

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"❌ Error generating answer from Gemini ({GEMINI_FLASH_MODEL}): {e}", file=sys.stderr)
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