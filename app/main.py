from fastapi import FastAPI
from app.api import router

app = FastAPI(title="Learning RAG Refactored")
app.include_router(router)
