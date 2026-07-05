"""
Phase 5: the FastAPI app. No model/tokenization logic lives here — this file
only wires HTTP requests to the existing load_model()/predict() functions in
api/inference.py.

Phase 6 adds /analyze/stream — a deliberately artificial streaming demo
endpoint for async/streaming practice, not a redesign of sentiment analysis.
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

# Same reasoning as api/inference.py: api/ and train/ are sibling directories
# with no __init__.py, so we add the project root to sys.path once here and
# import everything else (api.inference, api.schemas) as namespace packages,
# rather than relying on however this file happens to be launched.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from api.inference import load_model, predict
from api.schemas import AnalyzeRequest, AnalyzeResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Why `lifespan` instead of the older @app.on_event("startup")/
    # ("shutdown") decorators?
    # on_event registers two INDEPENDENT callbacks with no shared scope and
    # no structural link between them — nothing forces "whatever startup
    # acquired" to be the same thing shutdown releases, you just have to keep
    # two separate functions in sync by hand. It's also deprecated by FastAPI
    # in favor of lifespan.
    # lifespan is a single async context manager: everything before `yield`
    # runs once at startup, everything after `yield` runs once at shutdown,
    # and both share the same local scope/variables naturally — the
    # acquire/release pairing is enforced by the language's own
    # try/finally-like structure, not by developer discipline. It also
    # composes correctly with async lifecycles other startup patterns don't
    # (e.g. awaiting an async resource, or FastAPI's TestClient triggering
    # the same lifespan during tests, so tests see realistic startup behavior).
    model, vocab = load_model()

    # app.state is a plain namespace object FastAPI attaches to the app
    # instance specifically for this purpose — holding application-wide
    # objects that live for the whole process, as opposed to per-request
    # data. We store the loaded (model, vocab) here ONCE, at startup, so
    # every request handler can reuse the same objects instead of reloading
    # them (see Phase 4's summary on why loading once matters).
    app.state.model = model
    app.state.vocab = vocab

    yield  # server runs and handles requests while suspended here

    # Nothing to release on shutdown: no open file handles, DB connections,
    # or background tasks were started above, so there's no cleanup step —
    # but the structure is here so adding one later is a one-line change in
    # an obvious place, not a new mechanism to introduce.


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    # Deployed APIs conventionally expose a trivial, dependency-free health
    # endpoint like this because the infrastructure around the API (a load
    # balancer, a container orchestrator's liveness/readiness probes, uptime
    # monitoring) needs a cheap, fast way to ask "is this process up and
    # responding at all?" without exercising real business logic (no model
    # inference, no vocab lookups) that could be slow or have its own
    # failure modes unrelated to "is the server alive." This becomes directly
    # relevant in Phase 8: deployment platforms typically restart or stop
    # routing traffic to instances that fail a health check repeatedly.
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest, request: Request):
    # FastAPI inspects each route parameter's TYPE ANNOTATION, not its name,
    # to decide how to fill it in: `payload: AnalyzeRequest` gets built from
    # the parsed+validated JSON request body, while `request: Request`
    # (Starlette's Request type) gets injected as the actual live request
    # object for this call. `request.app` is the same FastAPI `app` instance
    # defined above, so `request.app.state.model`/`.vocab` retrieves exactly
    # what `lifespan` stored at startup — this is the mechanism that lets a
    # route handler reach state that was set up once, outside of any request.
    result = predict(payload.text, request.app.state.model, request.app.state.vocab)
    return result


# --- Phase 6: streaming demo, self-contained and separate from /analyze. ---
# This ML task has no real reason to stream — a sentiment prediction is one
# small, already-fast result. This exists purely to practice the async
# streaming mechanism itself (relevant to real LLM-serving endpoints, which
# stream tokens as they're generated). The actual model work is NOT being
# streamed; only the delivery of an already-computed result is simulated as
# streamed, so the mechanism can be observed in isolation.


async def stream_result(sentiment: str, confidence: float):
    """
    Async generator yielding the result in 3 separate chunks with a delay
    between each, so the streaming behavior is visibly observable (e.g. via
    `curl --no-buffer`) rather than arriving all at once.
    """
    yield json.dumps({"status": "analyzing..."}) + "\n"

    # asyncio.sleep(), not time.sleep(): this function runs inside the
    # single-threaded asyncio event loop that FastAPI uses to handle EVERY
    # concurrent request. asyncio.sleep() yields control back to that event
    # loop for the duration of the sleep, so the loop is free to go work on
    # other requests (or other chunks of other streaming responses) in the
    # meantime — that's the whole point of "async." time.sleep() is a
    # BLOCKING call: it freezes the entire OS thread the event loop runs on,
    # which means every other request the server is handling — not just this
    # one — would simply stall for that same duration, because there is no
    # other thread free to serve them. In an async def endpoint, a stray
    # time.sleep() doesn't just slow down its own request, it takes the
    # whole server down with it for that interval.
    await asyncio.sleep(0.5)
    yield json.dumps({"sentiment": sentiment}) + "\n"

    await asyncio.sleep(0.5)
    yield json.dumps({"confidence": confidence}) + "\n"


@app.post("/analyze/stream")
async def analyze_stream(payload: AnalyzeRequest, request: Request):
    # predict() is called once, synchronously, up front — inference itself
    # is fast and NOT what's being streamed. Only the delivery of the
    # already-known result is artificially spread out below.
    result = predict(payload.text, request.app.state.model, request.app.state.vocab)

    # What StreamingResponse does differently from a normal `return`:
    # A normal FastAPI return value (like /analyze's `return result` above)
    # is fully serialized to a complete response body BEFORE anything is
    # sent to the client — the client receives the entire response at once,
    # only after the route handler has finished running end-to-end.
    # StreamingResponse instead takes an async generator and sends each
    # yielded chunk to the client AS SOON AS it's yielded, without waiting
    # for the generator to finish. The client here starts receiving bytes
    # the moment the first `yield` fires (after the "analyzing..." chunk),
    # not after all three chunks (and the 1-second total delay) have
    # completed — the response is being built and transmitted incrementally
    # while the generator is still running, not handed over as one finished
    # object at the end.
    return StreamingResponse(
        stream_result(result["sentiment"], result["confidence"]),
        media_type="application/x-ndjson",
    )
