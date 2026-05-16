"""
FastAPI Application
-------------------
Exposes:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse (stateless, full history per call)

Startup:
  - Loads catalog from data/catalog.json
  - Builds/loads Chroma vector index
  - Compiles LangGraph agent
  - All subsequent /chat calls reuse the compiled graph
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

from agent.state import (
    ChatRequest, ChatResponse, Recommendation,
    INTENT_COMPARE,
)
from agent.graph import build_graph, build_initial_state
from catalog.vector_store import SHLVectorStore


# ------------------------------------------------------------------
# App-level singletons (initialized at startup)
# ------------------------------------------------------------------

_vector_store: SHLVectorStore = None
_agent_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all resources at startup."""
    global _vector_store, _agent_graph

    catalog_path = os.getenv("CATALOG_JSON_PATH", "./data/catalog.json")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    logger.info("Initializing SHL Assessment Recommender...")

    # Build vector store
    _vector_store = SHLVectorStore(
        catalog_path=catalog_path,
        chroma_persist_dir=chroma_dir,
    )
    _vector_store.build_index(force_rebuild=False)

    # Build agent
    _agent_graph = build_graph(_vector_store)

    logger.info("SHL Recommender ready.")
    yield

    logger.info("Shutting down.")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment selection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    """Readiness check."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint.
    Accepts full conversation history, returns next agent reply.
    """
    if not _agent_graph:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    if not request.messages:
        raise HTTPException(status_code=422, detail="messages cannot be empty.")

    # Convert Pydantic models to dicts for LangGraph
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Validate last message is from user
    if messages[-1]["role"] != "user":
        raise HTTPException(status_code=422, detail="Last message must be from user.")

    # Build initial state
    initial_state = build_initial_state(messages)

    try:
        # Run the graph
        final_state = _agent_graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Extract reply
    reply = final_state.get("reply", "")
    if not reply:
        reply = "I'm sorry, I encountered an issue. Could you please rephrase your request?"

    # Build recommendations list (only from catalog — never hallucinated)
    recommendations = []
    shortlist = final_state.get("shortlist", [])

    # For comparison intents, don't return recommendations
    intent = final_state.get("intent", "")
    if intent != INTENT_COMPARE and shortlist:
        for a in shortlist:
            # Guard: only include if URL exists in catalog
            url = a.get("url", "")
            name = a.get("name", "")
            if not url or not name:
                continue
            # Validate URL is from shl.com
            if "shl.com" not in url.lower():
                logger.warning(f"Skipping non-SHL URL: {url}")
                continue

            test_type = a.get("test_types", "")
            recommendations.append(
                Recommendation(
                    name=name,
                    url=url,
                    test_type=test_type,
                )
            )

    end_of_conversation = final_state.get("end_of_conversation", False)

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
