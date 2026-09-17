from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from ask import (
    get_gemini_client,
    get_chroma_client,
    retrieve_relevant_emails,
    generate_answer,
    build_prompt,
)


app = FastAPI(title="RAG Email Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.post("/api/query", response_model=QueryResponse)
def api_query(request: QueryRequest) -> QueryResponse:
    question: str = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chroma_client = get_chroma_client()
    documents, metadatas, ids = retrieve_relevant_emails(question, n_results=3, client=chroma_client)

    if not documents:
        return QueryResponse(answer="I could not find that in your emails.", sources=[])

    context = "\n\n".join(documents)

    gemini_client = get_gemini_client()
    prompt = build_prompt(question, context)

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini generation error: {e}")

    sources = [
        {
            "subject": meta.get("subject", "Unknown subject"),
            "from_addr": meta.get("from_addr", ""),
            "date": meta.get("date", ""),
            "snippet": meta.get("snippet", ""),
        }
        for meta in metadatas
    ]

    return QueryResponse(answer=answer, sources=sources)


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}