"""support_rag.py - Customer Support RAG pipeline (Experiment 7).

Document loading, chunking, embeddings, vector search, prompt construction
and the LLM backend live here. app.py only exposes this over HTTP.
"""
import logging
import os
from typing import List, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()
logger = logging.getLogger("support_rag")

# Configuration (from environment / .env) 
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()  

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://router.huggingface.co/v1")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")

FAQ_PATH = os.getenv("FAQ_PATH", "support_faq.txt")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
# FAISS L2 distance: lower = more similar. Chunks farther than this are ignored.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.2"))

INSUFFICIENT = "INSUFFICIENT_CONTEXT"
FALLBACK_MESSAGE = (
    "I couldn't find a reliable answer in the available FAQ. "
    "Please contact the support team through the Help section. "
    "Do not share your password, one-time code, or full payment-card details."
)

class LLMBackendError(Exception):
    """Raised when the model backend (Ollama / cloud API) fails."""

# LLM backend
def generate(prompt: str) -> str:
    """Send a prompt to the chosen backend and return the text."""
    try:
        if LLM_BACKEND == "ollama":
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 160},
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

        if LLM_BACKEND == "cloud":
            if not CLOUD_API_KEY or not CLOUD_MODEL:
                raise LLMBackendError("Cloud backend not configured")
            resp = requests.post(
                f"{CLOUD_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {CLOUD_API_KEY}"},
                json={
                    "model": CLOUD_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 160,
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        raise LLMBackendError(f"Unknown LLM_BACKEND '{LLM_BACKEND}'")

    except LLMBackendError:
        raise
    except (requests.RequestException, KeyError, ValueError) as exc:
        # Log only the exception type - never headers/keys.
        logger.error("LLM backend call failed: %s", type(exc).__name__)
        raise LLMBackendError("LLM backend call failed") from exc


# Graph state 
class SupportState(TypedDict, total=False):
    question: str
    intent: str
    search_query: str
    context: str
    sources: List[str]
    draft: str
    retry_count: int
    final_answer: str

class SupportRAG:
    """Builds the knowledge base once; call .ask() for each request."""

    def __init__(self):
        logger.info("Loading FAQ from %s", FAQ_PATH)
        with open(FAQ_PATH, encoding="utf-8") as f:
            text = f.read()
        doc = Document(page_content=text, metadata={"source": FAQ_PATH})

        splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
        self.chunks = splitter.split_documents([doc])
        logger.info("Created %d chunks", len(self.chunks))

        logger.info("Loading embedding model (first run downloads it)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.from_documents(self.chunks, embeddings)
        self.graph = self._build_graph()
        logger.info("RAG pipeline ready (backend=%s)", LLM_BACKEND)

    # nodes
    def _classify_intent(self, state: SupportState):
        q = state["question"].lower()
        if any(w in q for w in ["refund", "return", "exchange"]):
            intent = "returns_refunds"
        elif any(w in q for w in ["track", "tracking", "shipment", "delivery", "order status"]):
            intent = "order_tracking"
        elif any(w in q for w in ["password", "sign in", "login", "account", "verification"]):
            intent = "account_access"
        elif any(w in q for w in ["charged", "payment", "billing", "invoice", "card"]):
            intent = "billing_payments"
        elif any(w in q for w in ["website", "error", "loading", "browser", "technical"]):
            intent = "technical_support"
        else:
            intent = "general_support"
        return {
            "intent": intent,
            "search_query": state["question"],
            "retry_count": state.get("retry_count", 0),
        }

    def _retrieve(self, state: SupportState):
        results = self.vector_store.similarity_search_with_score(
            state["search_query"], k=3
        )
        logger.info("Retrieval distances: %s", [round(float(s), 3) for _, s in results])
        kept = [d for d, s in results if s <= MAX_DISTANCE]
        return {
            "context": "\n\n".join(d.page_content for d in kept),
            "sources": [d.metadata.get("source", "unknown") for d in kept],
        }

    def _draft_answer(self, state: SupportState):
        context = state.get("context", "")
        if not context.strip():
            # Nothing relevant retrieved: don't let the model guess.
            return {"draft": INSUFFICIENT}

        prompt = f"""You are a polite customer-support assistant.
Use only the FAQ CONTEXT to answer the CUSTOMER QUESTION.
Do not invent order status, refund approvals, timelines, or company policies.
If the context does not contain the answer, output exactly:
{INSUFFICIENT}
Do not ask for passwords, one-time codes, or full payment-card details.
Keep the answer clear and concise.

FAQ CONTEXT:
{context}

CUSTOMER QUESTION:
{state["question"]}

ANSWER:"""
        return {"draft": generate(prompt)}

    def _review_answer(self, state: SupportState):
        draft = state.get("draft", "").strip()
        if INSUFFICIENT in draft or not draft:
            return {"final_answer": ""}
        return {"final_answer": draft}

    @staticmethod
    def _route_after_review(state: SupportState):
        if state.get("final_answer"):
            return "finish"
        if state.get("retry_count", 0) < 1:
            return "retry"
        return "fallback"

    def _rewrite_query(self, state: SupportState):
        terms = {
            "returns_refunds": "return eligibility refund process",
            "order_tracking": "order dispatch tracking shipment status",
            "account_access": "account sign in password reset",
            "billing_payments": "billing payment transaction",
            "technical_support": "website technical troubleshooting",
            "general_support": "customer support help contact",
        }
        expanded = state["question"] + " " + terms.get(
            state.get("intent", "general_support"), "customer support"
        )
        return {
            "search_query": expanded,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # fallback
    @staticmethod
    def _fallback(state: SupportState):
        return {"final_answer": FALLBACK_MESSAGE}

    # built graph
    def _build_graph(self):
        g = StateGraph(SupportState)
        g.add_node("classify_intent", self._classify_intent)
        g.add_node("retrieve", self._retrieve)
        g.add_node("draft_answer", self._draft_answer)
        g.add_node("review_answer", self._review_answer)
        g.add_node("rewrite_query", self._rewrite_query)
        g.add_node("fallback", self._fallback)

        g.add_edge(START, "classify_intent")
        g.add_edge("classify_intent", "retrieve")
        g.add_edge("retrieve", "draft_answer")
        g.add_edge("draft_answer", "review_answer")
        g.add_conditional_edges(
            "review_answer",
            self._route_after_review,
            {"finish": END, "retry": "rewrite_query", "fallback": "fallback"},
        )
        g.add_edge("rewrite_query", "retrieve")
        g.add_edge("fallback", END)
        return g.compile()

    # public API 
    def ask(self, question: str) -> dict:
        result = self.graph.invoke({"question": question, "retry_count": 0})
        return {
            "answer": result.get("final_answer", FALLBACK_MESSAGE),
            "intent": result.get("intent"),
            "sources": sorted(set(result.get("sources", []))),
            "retries": result.get("retry_count", 0),
            "fallback_used": result.get("final_answer", "") == FALLBACK_MESSAGE,
        }

    def health(self) -> bool:
        return self.vector_store is not None
