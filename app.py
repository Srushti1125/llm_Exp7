import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from support_rag import LLMBackendError, SupportRAG, LLM_BACKEND

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

MAX_QUESTION_LENGTH = 1000
rag: SupportRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embeddings + vector index ONCE at startup, not per request.
    global rag
    rag = SupportRAG()
    yield

app = FastAPI(title="Customer Support RAG API", version="1.0", lifespan=lifespan)

# Schemas
class AskRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v

class AskResponse(BaseModel):
    answer: str
    intent: str | None = None
    sources: list[str] = []
    retries: int = 0
    fallback_used: bool = False
    latency_ms: float

# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # Client error (422). Return a short message, not the user's raw input.
    problems = [
        {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    logger.warning("Validation failed on %s", request.url.path)
    return JSONResponse(status_code=422, content={"detail": problems})

# Endpoints

# GET /health endpoint
@app.get("/health")
def health():
    # Limitation: only shows the API process is up, not that the LLM backend works.
    return {"status": "ok", "rag_loaded": bool(rag and rag.health()), "backend": LLM_BACKEND}

# POST /ask endpoint
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.perf_counter()
    try:
        result = rag.ask(req.question)
    except LLMBackendError:
        latency = (time.perf_counter() - start) * 1000
        logger.error("ask failed | reason=llm_backend | latency_ms=%.1f", latency)
        raise HTTPException(status_code=503, detail="The language model service is unavailable. Please try again later.")
    except Exception:
        latency = (time.perf_counter() - start) * 1000
        logger.exception("ask failed | reason=unexpected | latency_ms=%.1f", latency)
        raise HTTPException(status_code=500, detail="Internal server error.")

    latency = (time.perf_counter() - start) * 1000
    # Log metadata only - not the question text (may contain personal data).
    logger.info(
        "ask ok | intent=%s | fallback=%s | retries=%s | q_len=%d | latency_ms=%.1f",
        result["intent"], result["fallback_used"], result["retries"],
        len(req.question), latency,
    )
    return AskResponse(**result, latency_ms=round(latency, 1))
