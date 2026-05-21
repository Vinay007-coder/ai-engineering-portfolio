# ------------------------------------------------------------
# Project Architecture Overview
# ------------------------------------------------------------

# 1. Rate-Limiter (Semaphore)
#
#    - Controls concurrent AI generation jobs.
#
#    - Prevents:
#         • CPU overload
#         • memory exhaustion
#         • excessive parallel execution
#
#    - Ensures the system processes
#      only a limited number of tasks
#      simultaneously.
#
#    - Example:
#
#          asyncio.Semaphore(...)

# 2. Lifecycle Manager
#    (Async Context Manager)
#
#    - Handles resource initialization
#      and cleanup automatically.
#
#    - Manages:
#         • database connections
#         • socket setup
#         • session teardown
#
#    - Uses:
#
#          __aenter__()
#          __aexit__()
#
#    - Guarantees cleanup even during failures.

# 3. Asynchronous Generator (yield)
#
#    - Streams AI-generated text tokens
#      directly to the client in real time.
#
#    - Avoids waiting for the full response
#      before sending data.
#
#    - Enables:
#         • live token streaming
#         • lower memory usage
#         • responsive UI updates
#
#    - Uses:
#
#          yield token

# 4. Failure Containment
#    (TaskGroup + timeout)
#
#    - Isolates slow or failing tasks safely.
#
#    - Prevents one broken connection
#      from crashing the entire system.
#
#    - timeout:
#
#         • limits task execution time
#         • interrupts hanging operations
#
#    - TaskGroup:
#
#         • manages structured concurrency
#         • propagates cancellations safely
#         • cleans up leaked sub-tasks

# 5. Data Validation (Pydantic v2)
#
#    - Validates incoming request data.
#
#    - Enforces:
#         • correct types
#         • schema consistency
#         • safe parsing
#
#    - Produces clean serialized output
#      using JSON dump formatting.
#
#    - Helps maintain:
#         • API reliability
#         • predictable responses
#         • safer application logic


import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# ==========================================
# 1. PYDANTIC VALIDATION MODELS
# ==========================================
class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=5, max_length=500, description="The text prompt for the AI model.")
    temperature: float = Field(default=0.7, ge=0.0, le=1.5, description="Sampling temperature.")
    user_id: str = Field(description="Must be formatted as 'usr_XXXX'")

    @field_validator('user_id')
    @classmethod
    def validate_user_id_format(cls, value: str) -> str:
        if not value.startswith("usr_"):
            raise ValueError("Invalid identification token. Must start with 'usr_'")
        return value

class StreamTokenPayload(BaseModel):
    token: str
    index: int
    is_final: bool = False

# ==========================================
# 2. LIFECYCLE MANAGEMENT (Async Context Manager)
# ==========================================
class FakeDatabaseClient:
    def __init__(self):
        self.connected = False

    async def __aenter__(self):
        print("[DB] Open connection pool asynchronously...")
        await asyncio.sleep(0.5)  # Simulate network connection overhead
        self.connected = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("[DB] Draining pool and closing connection asynchronously...")
        await asyncio.sleep(0.2)
        self.connected = False

    async def log_request(self, user_id: str, prompt: str):
        if not self.connected:
            raise RuntimeError("Database connection not alive.")
        print(f"[DB Log] Tracking request for {user_id} -> Prompt length: {len(prompt)}")

# Global concurrency controller to simulate API rate-limiting (Max 2 simultaneous AI generations)
AI_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(2)

# Global FastAPI application lifecycle hook
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    print("🚀 --- Backend System Booted ---")
    yield
    print("🛑 --- Backend System Stopped ---")

app = FastAPI(lifespan=app_lifespan)

# ==========================================
# 3. CORE SERVICE LAYER (Async Generators & Concurrency)
# ==========================================
async def generate_ai_tokens(prompt: str) -> AsyncGenerator[str, None]:
    """Simulates a heavy LLM pipeline streaming text responses chunk-by-chunk."""
    mock_tokens = ["Deep", " Learning", " pipelines", " run", " highly", " efficiently", " under", " Asyncio."]
    
    # Enforce global rate-limiting protection
    async with AI_CONCURRENCY_SEMAPHORE:
        for i, token in enumerate(mock_tokens):
            await asyncio.sleep(0.25)  # Simulate AI generation processing time
            
            # Formulate outbound structural frame via Pydantic
            payload = StreamTokenPayload(
                token=token,
                index=i,
                is_final=(i == len(mock_tokens) - 1)
            )
            # Yield stringified validation format directly into network socket
            yield payload.model_dump_json() + "\n"

async def fetch_auxiliary_metadata(user_id: str) -> Dict[str, Any]:
    """Simulates a dependent metadata lookup with strict error timeout control."""
    try:
        # Enforce a 1.5-second SLA constraint on this secondary data fetch
        async with asyncio.timeout(1.5):
            if user_id == "usr_slow":
                await asyncio.sleep(3.0)  # Purposefully break SLA limit
            await asyncio.sleep(0.3)
            return {"tier": "enterprise", "quota_remaining": 942}
            
    except asyncio.TimeoutError:
        print(f"[Warning] Metadata lookup timed out for user: {user_id}")
        return {"tier": "standard_fallback", "quota_remaining": 0}
        
    except asyncio.CancelledError:
        print(f"[Cleanup] Metadata lookup task explicitly terminated mid-flight for user: {user_id}")
        raise

# ==========================================
# 4. API ROUTE ROUTERS (FastAPI App Layer)
# ==========================================
@app.post("/v1/chat/stream")
async def chat_stream_endpoint(request: GenerationRequest):
    """
    Executes an orchestrated AI streaming pipeline combining TaskGroups,
    Async Context Managers, and strict data parsing models.
    """
    try:
        # Use database context manager on-demand
        async with FakeDatabaseClient() as db:
            # Structured Concurrency execution using a TaskGroup
            async with asyncio.TaskGroup() as tg:
                # 1. Fire database logging asynchronously
                db_task = tg.create_task(db.log_request(request.user_id, request.prompt))
                # 2. Fire independent profile check concurrently
                meta_task = tg.create_task(fetch_auxiliary_metadata(request.user_id))
            
            # Confirm tasks completed successfully within the group block
            await db_task
            user_metadata = meta_task.result()
            print(f"[Pipeline] User metadata established: {user_metadata}")

        # Construct and deliver the asynchronous chunk response directly over HTTP
        return StreamingResponse(
            generate_ai_tokens(request.prompt),
            media_type="text/event-stream"
        )

    except* RuntimeError as eg:
        # Capture underlying architecture faults caught inside the ExceptionGroup framework
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Database Lifecycle Crash: {str(eg.exceptions)}"
        )
    except* Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ==========================================================
# Verify the Production Flows
# ==========================================================

# Scenario A: Successful Concurrent Stream Request

# curl -X POST "http://127.0.0.1:8000/v1/chat/stream" \
#      -H "Content-Type: application/json" \
#      -d '{"prompt": "Explain async design patterns", "user_id": "usr_9981"}'

# Scenario B: Input Schema Interception (Pydantic Layer Validation)

# curl -X POST "http://127.0.0.1:8000/v1/chat/stream" \
#      -H "Content-Type: application/json" \
#      -d '{"prompt": "Hi", "user_id": "bad_token_99"}'

# Scenario C: Isolated Timeout Processing (asyncio.timeout)

# curl -X POST "http://127.0.0.1:8000/v1/chat/stream" \
#      -H "Content-Type: application/json" \
#      -d '{"prompt": "Test query string pattern", "user_id": "usr_slow"}'


