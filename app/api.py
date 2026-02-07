from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import time

from app.services.embedding import EmbeddingService
from app.services.store import DocumentStore
from app.services.rag import RagWorkflow

embedder = EmbeddingService()
store = DocumentStore()
workflow = RagWorkflow(embedder, store)

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

class DocumentRequest(BaseModel):
    text: str

@router.post("/add")
def add_document(req: DocumentRequest):
    """Tambah dokumen ke sistem (Qdrant / memory)."""
    try:
        vector = embedder.fake_embed(req.text)
        store.add_document(req.text, vector)
        return {"status": "added", "text": req.text[:50]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
def ask_question(req: QuestionRequest):
    """Menjawab pertanyaan berdasarkan dokumen yang disimpan."""
    start = time.time()
    try:
        result = workflow.run(req.question)
        return {
            "question": req.question,
            "answer": result["answer"],
            "context": result.get("context", []),
            "latency_sec": round(time.time() - start, 3),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def status():
    """Cek apakah sistem siap."""
    return {
        "qdrant_ready": store.use_qdrant,
        "graph_ready": True
    }
