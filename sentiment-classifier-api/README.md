# Sentiment Classifier API

A sentiment classifier trained from scratch in plain PyTorch, served via FastAPI.

Live API: **https://ai-engineering-portfolio-b4yd.onrender.com**

> This runs on Render's free tier, which sleeps the instance after 15 minutes
> of inactivity. The first request after a period of idleness will take
> **30-60s** to respond (cold start) while the instance spins back up.
> Subsequent requests are fast.

## What this is

This is a sentiment classifier trained from scratch in plain PyTorch (no
HuggingFace `transformers`, no pretrained embeddings, no pretrained
tokenizer) on the IMDB movie review dataset, served through a FastAPI
backend and deployed on Render. It was built as a first hands-on AI
engineering learning project — the architecture is deliberately simple
(a "bag of embeddings" model, not an RNN or Transformer) so that every step
of the pipeline, from tokenization to training to serving, could be
understood and traced by hand before reaching for more advanced models.

## Architecture

```
raw text
  -> tokenize (lowercase, strip punctuation, split on whitespace)
  -> vocab lookup (min_freq=5; index 0 = <PAD>, index 1 = <UNK>)
  -> nn.Embedding (embedding_dim=100)
  -> masked mean pooling over the sequence (padding excluded from the average)
  -> nn.Linear(embedding_dim, 1)
  -> sigmoid -> probability
```

This is an **order-invariant "bag of embeddings" model** by deliberate
choice — there is no RNN, attention, or any mechanism for word order or
context to influence the result; each token's embedding is looked up
independently and then averaged. Trained for 5 epochs, reaching **~87%
validation accuracy**.

## API Reference

### `GET /health`

Trivial liveness check — no model inference involved.

```bash
curl https://ai-engineering-portfolio-b4yd.onrender.com/health
```

```json
{"status": "ok"}
```

### `POST /analyze`

Runs the model once and returns a single JSON result.

```bash
curl -X POST https://ai-engineering-portfolio-b4yd.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely wonderful, I loved it!"}'
```

```json
{"sentiment": "positive", "confidence": 0.9999}
```

### `POST /analyze/stream`

Same request shape as `/analyze`, but streams the result back as 3
newline-delimited JSON chunks (`status` -> `sentiment` -> `confidence`) with
an artificial delay between each. This endpoint was built purely as
async/streaming practice — this specific classification task doesn't
benefit from streaming (the result is known before the first chunk is
sent); a real LLM-serving endpoint streaming generated tokens as they're
produced is the actual motivating use case for this pattern.

```bash
curl -N -X POST https://ai-engineering-portfolio-b4yd.onrender.com/analyze/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely wonderful, I loved it!"}'
```

```
{"status": "analyzing..."}
{"sentiment": "positive"}
{"confidence": 0.9999}
```

## Known Limitations

These were found through actual testing against the deployed model, not
theoretical concerns:

- **Negation handling fails.** `"not good"` is predicted **positive** at
  ~58% confidence. Mean pooling has no notion of word order, and the
  embedding for "not" ends up with a weak, near-neutral learned vector
  because it appears constantly in *both* positive contexts ("not bad") and
  negative contexts ("not good") during training — the model never learns
  that "not" flips the sentiment of the word that follows it.

- **Poor confidence calibration.** Predictions on unambiguous text saturate
  near 99.99% confidence — for both correct *and* incorrect predictions.
  Genuinely ambiguous or negated text lands much closer to 50-60%. In
  practice, the confidence score tracks how strongly the pooled vectors
  happen to agree internally, not a properly calibrated probability of
  being right.

- **Mixed-sentiment sentences are resolved by raw averaging, not by
  weighing clauses.** `"great acting but terrible plot"` is decided by
  whichever tokens' embeddings dominate the averaged vector — a function of
  word count and vector magnitude, not which clause actually matters more
  to a human reader. The model can land confidently on the wrong side.

- **`/analyze/stream` calls `predict()` synchronously inside an `async def`
  route, with no internal `await` point during inference.** This is fine at
  this model's millisecond-scale inference speed — it never blocks the
  event loop long enough to matter. But it's a known anti-pattern worth
  flagging: if `predict()` were ever replaced with something slower (e.g. a
  real LLM API call), that synchronous call would block the *entire* event
  loop — every other concurrent request — for its full duration, not just
  the request that triggered it.

## Running Locally

```bash
git clone <repo-url>
cd sentiment-classifier-api

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Optional — retrain from scratch (needs the `datasets` library installed
# separately; it's intentionally not in requirements.txt, see Tech Stack):
# pip install datasets
# python train/train.py

uvicorn api.main:app --reload
```

Then, against `http://127.0.0.1:8000`:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely wonderful, I loved it!"}'

curl -N -X POST http://127.0.0.1:8000/analyze/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely wonderful, I loved it!"}'
```

## Tech Stack

- **Python** / **PyTorch** — model architecture and training loop, written
  from scratch (no `transformers`, no pretrained tokenizer/embeddings)
- **FastAPI** / **Uvicorn** — serving layer
- **Render** — deployment (free tier, CPU-only)
- **HuggingFace `datasets`** — used only on the training side, to download
  the raw IMDB text/label pairs; not a runtime dependency of the API

## What I Learned

- Writing a training loop by hand — `zero_grad()` / `forward()` / `loss.backward()`
  / `optimizer.step()` — makes the mechanics of gradient descent concrete in
  a way that calling a `Trainer.fit()` never would have.
- Why train/serve separation matters: the training code has no business
  knowing about HTTP, and the serving code has no business knowing about
  epochs or optimizers. Splitting `train/vocab_utils.py` out of `train/dataset.py`
  so the API wouldn't transitively import the `datasets` library made this
  concrete, not just theoretical.
- The load-once-at-startup pattern: loading the model and vocab inside the
  request handler, versus once at process startup via FastAPI's `lifespan`,
  is the difference between a server that's fast and one that redoes
  expensive setup work on every single request.
- Async vs. sync in practice: writing the `/analyze/stream` endpoint made
  the event-loop-blocking behavior of a synchronous call inside `async def`
  tangible — and cheap to get away with here, which is exactly why it's
  worth flagging as a latent trap rather than a real bug.
- The biggest lesson came from actually testing the deployed model instead
  of trusting the validation accuracy number: ~87% val accuracy said
  nothing about whether the model understood negation. It doesn't. Testing
  specific, deliberately adversarial inputs ("not good", mixed-sentiment
  sentences) surfaced real failure modes that a single aggregate accuracy
  metric completely hid.
