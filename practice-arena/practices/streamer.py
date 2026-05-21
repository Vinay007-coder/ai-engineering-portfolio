# ----------------------------------------------------------------------------
# 3. Practice: FastAPI Word Streamer (async generator)
# ----------------------------------------------------------------------------

import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def word_token_generator(text_prompt: str):
    """An async generator splitting text strings into word chunks over time."""
    words = text_prompt.split(" ")
    
    for i, word in enumerate(words):
        await asyncio.sleep(0.3)  # Simulate network latency or AI model processing delay
        
        # Format explicitly according to the Server-Sent Events (SSE) standard protocol:
        # Format pattern must be: data: <payload>\n\n
        yield f"data: {word}\n\n"

@app.get("/v1/stream")
async def stream_words_endpoint(prompt: str = "Asyncio and generators power high performance web streaming architectures"):
    # StreamingResponse hooks the async generator output directly into the HTTP channel
    return StreamingResponse(
        word_token_generator(prompt),
        media_type="text/event-stream"
    )

# Test it using curl in another terminal: curl -X GET "http://127.0.0.1:8000/v1/stream"


# ==========================================================
# 4. Architecture Deep Dive: Why SSE Requires Generators
# ==========================================================

# To understand why Server-Sent Events (SSE) must use generators to push data to 
# browsers, look at how the data moves through memory and the network:

# TRADITIONAL RETURN PATTERN (Memory Hog / High Latency)
# [Server Processes 5s] ──► [Compiles whole array in RAM] ──► [Sends 100% Data at once] ──► [Connection Closes]

# SSE GENERATOR STREAMING PATTERN (Low Memory / Zero Latency)
# [Token 1] ──► Yield ──► Pushed to Socket ──► Browser displays word instantly
# [Token 2] ──► Yield ──► Pushed to Socket ──► Browser displays word instantly
# (Network channel stays open throughout the iteration)

# ------------------------------------------------------------
# Technical Reasons Behind SSE + Generators
# ------------------------------------------------------------

# 1. The Persistent HTTP Pipe
#
#    - A normal HTTP connection closes immediately
#      after a function exits using `return`.
#
#    - Server-Sent Events (SSE) keeps the connection alive:
#
#          Connection: keep-alive
#
#    - Because generators pause at `yield`
#      instead of terminating completely,
#      they can continuously stream data
#      through the same open connection.
#
#    - This enables incremental delivery
#      of data over time.

# 2. No Server Buffer Bloat
#
#    - Returning a huge payload at once
#      forces the server to hold the entire
#      response in RAM before transmission.
#
#    - Example:
#
#         • massive CSV exports
#         • large log streams
#         • AI-generated responses
#
#    - Generators avoid this problem by:
#
#         • loading a small chunk
#         • yielding it immediately
#         • releasing memory afterward
#
#    - This dramatically reduces
#      memory pressure on the server.

# 3. Real-Time Push Dynamics
#
#    - Servers cannot easily push data
#      to browsers without an existing
#      client connection.
#
#    - With SSE:
#
#         • the client opens the stream first
#         • the connection remains active
#
#    - The generator then behaves like
#      a reactive streaming engine,
#      sending new data fragments
#      the exact moment they become available.
#
#    - This enables:
#
#         • live AI token streaming
#         • real-time dashboards
#         • event feeds
#         • instant UI updates


