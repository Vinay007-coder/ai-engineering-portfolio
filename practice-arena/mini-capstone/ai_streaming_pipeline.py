# ------------------------------------------------------------
# Real-World AI Streaming Architecture Scenario
# ------------------------------------------------------------

# We are building a production-style
# AI Content Generation API endpoint.

# The system combines:
#
#     • request validation
#     • async decorators
#     • streaming generators
#     • FastAPI SSE responses
#
# into a single high-performance pipeline.

# ============================================================
# 1. Request Validation Layer
# (Pydantic v2)
# ============================================================

# Incoming client request payloads
# are validated using Pydantic v2 models.

# Responsibilities:
#
#     • enforce strict schemas
#     • validate input data types
#     • reject malformed requests
#     • serialize structured output safely

# Example validation targets:
#
#     • prompts
#     • model identifiers
#     • token limits
#     • generation settings

# This creates predictable
# and reliable API behavior.

# ============================================================
# 2. Async Performance Decorator
# ============================================================

# A custom async decorator intercepts
# the endpoint execution before
# the actual business logic runs.

# Responsibilities:
#
#     • measure execution latency
#     • record diagnostics
#     • log request timing metrics
#     • monitor pipeline performance

# The decorator tracks execution time
# down to the millisecond level.

# Typical production usage:
#
#     • observability systems
#     • performance dashboards
#     • API monitoring
#     • latency optimization

# ============================================================
# 3. Async Generator Streaming Core
# ============================================================

# The AI generation engine is modeled
# as an async generator.

# Instead of waiting for the full
# response to finish generation,
# the system streams tokens incrementally.

# Uses:
#
#     async for
#     yield token

# Responsibilities:
#
#     • process data chunk-by-chunk
#     • reduce memory pressure
#     • enable real-time streaming
#     • improve responsiveness

# This simulates how real LLM systems
# generate text progressively.

# ============================================================
# 4. FastAPI + SSE Streaming Layer
# ============================================================

# FastAPI exposes the generator output
# through Server-Sent Events (SSE).

# SSE keeps the HTTP connection alive,
# allowing the backend to continuously
# push new AI-generated tokens
# directly into the client interface.

# This enables:
#
#     • live token streaming
#     • real-time UI updates
#     • interactive AI chat systems
#     • low-latency user experience

# ============================================================
# Complete Pipeline Flow
# ============================================================

# Client Request
#       ↓
#
# Pydantic Validation
#       ↓
#
# Async Performance Decorator
#       ↓
#
# Async AI Generator
#       ↓
#
# yield token
#       ↓
#
# FastAPI SSE Stream
#       ↓
#
# Live Browser Updates

# ============================================================
# Production Concepts Demonstrated
# ============================================================

# This architecture models
# real modern AI backend systems using:
#
#     • async programming
#     • structured validation
#     • decorators
#     • generators
#     • SSE streaming
#     • observability metrics
#     • low-memory streaming pipelines
#
# Similar architectural patterns exist in:
#
#     • ChatGPT-style interfaces
#     • AI copilots
#     • streaming inference APIs
#     • real-time chatbot systems
# ------------------------------------------------------------


import asyncio
import functools
import time
from typing import AsyncGenerator, Callable, Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Unified Async Production Core")

# ==========================================
# 1. PYDANTIC VALIDATION LAYER
# ==========================================
class AIQueryRequest(BaseModel):
    # Enforcing strict data parameters via Field() and Type Hints
    prompt: str = Field(min_length=5, max_length=200, description="The structural instruction input.")
    max_tokens: int = Field(default=100, ge=10, le=500, description="Upper resource budget limit.")
    client_id: str = Field(description="Internal system tracking identifier.")

    # Custom validation rule leveraging Pydantic v2 validation decorators
    @field_validator("client_id")
    @classmethod
    def verify_client_signature(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value.startswith("cli_"):
            raise ValueError("Invalid credentials layout. Identifier must start with 'cli_'")
        return cleaned_value

# ==========================================
# 2. METADATA PRESERVING ASYNC DECORATOR
# ==========================================
def async_performance_monitor(func: Callable[..., Any]) -> Callable[..., Any]:
    """An asynchronous decorator that prints execution metrics for non-blocking functions."""
    @functools.wraps(func)  # Critical step: Preserves method name, type annotations, and docstrings
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_marker: float = time.perf_counter()
        print(f"\n🚀 [Monitor Start] Intercepting execution path for route handler: '{func.__name__}'")
        
        # Execute the underlying asynchronous endpoint logic
        result: Any = await func(*args, **kwargs)
        
        duration: float = time.perf_counter() - start_marker
        print(f"⏱️ [Monitor Stop] Route '{func.__name__}' orchestrator completed in {duration:.4f} seconds.")
        return result
    return wrapper

# ==========================================
# 3. ASYNC STREAMING GENERATOR
# ==========================================
async def mock_llm_token_engine(prompt: str, limit: int) -> AsyncGenerator[str, None]:
    """An asynchronous generator that yields simulated AI words following the SSE wire format."""
    print(f"[Engine] Activating generation weights for prompt: '{prompt}'")
    chunks: list[str] = ["Asynchronous", " architectures", " maximize", " server", " hardware", " capacity."]
    
    for i, chunk in enumerate(chunks):
        if i >= limit:  # Break out if we hit the user token budget limit
            break
            
        await asyncio.sleep(0.2)  # Simulate non-blocking AI calculation latency
        
        # Format explicitly according to the Server-Sent Events (SSE) standard data: <payload>\n\n
        yield f"data: {chunk}\n\n"

# ==========================================
# 4. FASTAPI ENDPOINT WITH TYPE HINTS
# ==========================================
@app.post("/v1/ai/generate")
@async_performance_monitor  # Stacking our custom metrics engine on top of the endpoint
async def run_ai_generation_pipeline(request: AIQueryRequest) -> StreamingResponse:
    """
    Validates input payloads, monitors system metrics via custom decorators, 
    and streams tokens directly to the client browser in real-time.
    """
    print(f"[Pipeline] Processing validated request from account: {request.client_id}")
    
    # 1. Initialize our async generator stream recipe
    stream_generator: AsyncGenerator[str, None] = mock_llm_token_engine(
        prompt=request.prompt, 
        limit=request.max_tokens
    )
    
    # 2. Hand off the stream to StreamingResponse to keep the HTTP socket open
    return StreamingResponse(
        stream_generator, 
        media_type="text/event-stream"
    )

# Run via terminal: uvicorn main:app --reload


# ==========================================================
# End-to-End Operational Flow
# ==========================================================

#        [ Terminal Client Execution (curl) ]
#                         │
#                         ▼
#     [ Intercepted by @async_performance_monitor ]
#                         │
#                         ▼
#  1. Runs wrapper logic: Captures performance start timestamp.
#                         │
#                         ▼
#     [ Enters run_ai_generation_pipeline Route ]
#                         │
#                         ▼
#  2. Pydantic validates input. client_id is verified via field_validator.
#  3. Instantiates mock_llm_token_engine generator without triggering execution.
#                         │
#                         ▼
#  4. Returns StreamingResponse instantly. Wrapper concludes time metrics.
#                         │
#                         ▼
#     [ Streaming Pipeline stays open over the network ]
#  5. Generator yields tokens one-by-one -> "data: Asynchronous\n\n"
#  6. Browser renders words on-screen in real-time. Connection closes automatically.


# =========================================
# Verification Challenge
# ========================================

# curl -X POST "http://127.0.0.1:8000/v1/ai/generate" \
#      -H "Content-Type: application/json" \
#      -d '{"prompt": "Test async pipelines", "client_id": "cli_prod_9410", "max_tokens": 10}'

