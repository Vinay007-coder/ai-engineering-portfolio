"""
Phase 5: the FastAPI app. No model/tokenization logic lives here — this file
only wires HTTP requests to the existing load_model()/predict() functions in
api/inference.py.

No streaming endpoint here — that's Phase 6.
"""

import os
import sys
from contextlib import asynccontextmanager

# Same reasoning as api/inference.py: api/ and train/ are sibling directories
# with no __init__.py, so we add the project root to sys.path once here and
# import everything else (api.inference, api.schemas) as namespace packages,
# rather than relying on however this file happens to be launched.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Request

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
