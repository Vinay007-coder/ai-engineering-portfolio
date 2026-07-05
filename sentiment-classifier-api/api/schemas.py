"""
Phase 5: request/response contracts for the /analyze endpoint.

No model, tokenization, or inference logic here at all — just the shape of
data going in and out of the API.
"""

from pydantic import BaseModel, field_validator


class AnalyzeRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value):
        # Why validate here instead of letting predict() deal with it?
        # predict() would technically "succeed" on "" or "   " — encode()
        # would just tokenize it into an empty token list, which gets padded
        # to 256 <PAD> tokens, run through the model, and produce SOME
        # sigmoid output. That's the problem: it wouldn't crash, it would
        # silently return a confident-looking sentiment/confidence pair for
        # input that was never a real review to begin with — garbage in,
        # plausible-looking garbage out, with no signal to the caller that
        # anything was wrong. Rejecting blank input at the schema boundary
        # means the API responds with a clear 422 "this input was invalid"
        # instead of a 200 that looks fine but means nothing.
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value


class AnalyzeResponse(BaseModel):
    sentiment: str
    confidence: float

    # Declaring this as the route's response_model (see main.py) buys us
    # more than just "returning a dict already gets the job done":
    # 1. Output validation/coercion: if predict() ever returned an unexpected
    #    shape (e.g. a numpy float instead of a plain float, or a typo'd key),
    #    FastAPI would raise a clear server-side error immediately rather
    #    than silently sending malformed JSON to the client.
    # 2. Filtering: only the fields declared here are ever serialized — if
    #    predict() later returns extra internal fields (e.g. debug info),
    #    they're automatically dropped rather than leaking to callers.
    # 3. Auto-generated OpenAPI docs: FastAPI uses this model to document the
    #    exact response shape at /docs, so API consumers (and future you)
    #    don't have to read the route's implementation to know what comes back.
