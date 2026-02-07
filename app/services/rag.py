from langgraph.graph import StateGraph, END
from app.services.store import DocumentStore
from app.services.embedding import EmbeddingService

class RagWorkflow:
    def __init__(self, embedder: EmbeddingService, store: DocumentStore):
        self.embedder = embedder
        self.store = store
        self.chain = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("answer", self.answer)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        return workflow.compile()

    def retrieve(self, state):
        emb = self.embedder.fake_embed(state["question"])
        results = self.store.search(emb, state["question"])
        state["context"] = results
        return state

    def answer(self, state):
        ctx = state.get("context", [])
        state["answer"] = f"I found this: '{ctx[0][:100]}...'" if ctx else "Sorry, I don't know."
        return state
    
    def run(self, question: str) -> dict:
        """Run RAG flow from question input"""
        return self.chain.invoke({"question": question})