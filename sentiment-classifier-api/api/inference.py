"""
Phase 4: inference only. Loads the already-trained model/vocab from disk and
serves predictions. Deliberately has zero knowledge of epochs, loss
functions, or optimizers — that's train/train.py's job, not this file's.
"""

import os
import sys

# train/ and api/ are sibling directories with no __init__.py, so neither is
# a regular Python package. Adding the project root to sys.path lets us
# import `train.dataset` / `train.model` as implicit namespace packages
# (a Python 3 feature that doesn't require __init__.py) instead of
# duplicating tokenize/encode/model logic here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from train.dataset import encode, load_vocab
from train.model import PAD_IDX, SentimentClassifier

MAX_LEN = 256  # must match the max_len used in train/dataset.py, or token
# positions the model learned to expect (e.g. "position 255 is usually
# padding") would no longer hold.

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_model.pt")


def load_model():
    """
    Rebuild the exact architecture used in training and load the trained
    weights into it. Returns (model, vocab) so predict() has both.
    """
    # Same vocab file saved in Phase 1 / used in Phase 3 — the embedding
    # table's indices only mean what they mean relative to this specific
    # word->index mapping.
    vocab = load_vocab()

    # vocab_size=len(vocab) (not a hardcoded constant) so this always matches
    # whatever vocab.json actually contains, even if it's rebuilt later with
    # a different min_freq and ends up a different size.
    model = SentimentClassifier(vocab_size=len(vocab), embedding_dim=100, pad_idx=PAD_IDX)

    # map_location="cpu" ensures this loads correctly even if the weights
    # were originally saved from a CUDA GPU tensor and this is running on a
    # CPU-only serving machine — without it, torch.load can raise an error
    # trying to restore tensors onto a GPU device that doesn't exist here.
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    # model.eval() has no visible effect on THIS architecture (no
    # dropout/batchnorm layers to switch behavior), but it's still called
    # here as a deliberate habit: the moment you add dropout or batchnorm to
    # a later version of this model, forgetting eval() at inference time
    # becomes a silent, hard-to-notice bug (dropout would randomly zero
    # activations per request, batchnorm would use per-batch statistics
    # instead of the running statistics learned during training) that
    # produces inconsistent predictions for the exact same input. Calling it
    # unconditionally now means this code is already correct for that future,
    # more complex model.
    model.eval()

    return model, vocab


def predict(text, model, vocab):
    """
    Run one raw text string through the trained model and return
    {"sentiment": "positive"/"negative", "confidence": float}.
    """
    # Reuses dataset.py's encode() (which itself calls tokenize()) instead of
    # reimplementing tokenization here — inference MUST use the identical
    # text-to-ids logic training used, or the model is being fed input in a
    # different "language" than the one it learned.
    token_ids = encode(text, vocab)

    # Same truncate-then-pad logic as IMDBDataset.__getitem__ in dataset.py:
    # the model's embedding/pooling only works correctly on a fixed-length
    # (256,) sequence, so a single inference request has to be shaped
    # exactly like a training example was, not just "however long the real
    # text happens to be."
    token_ids = token_ids[:MAX_LEN]
    token_ids = token_ids + [PAD_IDX] * (MAX_LEN - len(token_ids))

    # unsqueeze(0) turns shape (256,) into (1, 256) — the model's forward()
    # expects a BATCH dimension even for a single review, since nn.Embedding
    # and the masked-mean-pool math are all written in terms of
    # (batch_size, seq_len, ...). A batch of size 1 is still a batch.
    input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    # no_grad() here for the same reason as validation in Phase 3: we will
    # never call .backward() on this output, so there's no reason to pay for
    # building the autograd computation graph. At API-serving scale this
    # matters more than it did for one validation pass — a server handling
    # many requests per second would otherwise build (and have to garbage
    # collect) a full graph of intermediate tensors on EVERY single request,
    # needlessly inflating both per-request latency and peak memory usage.
    # Over sustained traffic that's the difference between a lean, constant
    # memory footprint and one that grows and gets reclaimed in bursts under
    # load — a real risk of added latency or memory pressure for no benefit.
    with torch.no_grad():
        logit = model(input_tensor)  # shape: (1,) — one raw logit for our one review

    probability = torch.sigmoid(logit).item()  # squash to a (0, 1) probability, then
    # pull it out of the tensor as a plain Python float via .item().

    sentiment = "positive" if probability >= 0.5 else "negative"

    # Why report confidence as distance from 0.5, not the raw probability?
    # The raw sigmoid output is "probability the review is POSITIVE." For a
    # negative prediction, that number is naturally small (e.g. 0.06) — but
    # 0.06 is not "6% confident," it's "94% confident the review is
    # NEGATIVE" (since 1 - 0.06 = 0.94). Returning the raw 0.06 next to
    # sentiment="negative" would read as a low-confidence prediction when
    # it's actually a highly confident one. max(p, 1-p) reframes the number
    # around "how far is this from the 50/50 decision boundary," which is
    # the actual quantity a user cares about regardless of which class won.
    confidence = probability if probability >= 0.5 else 1 - probability

    return {"sentiment": sentiment, "confidence": confidence}


if __name__ == "__main__":
    # Load once, predict many times — see the summary on why this matters.
    model, vocab = load_model()

    examples = [
        "This movie was absolutely wonderful, the acting and story were fantastic!",
        "Terrible film, a complete waste of time. I hated every minute of it.",
        "The acting was great but the plot was terrible and made no sense.",
        "It was okay, nothing special but not bad either, pretty average overall.",
    ]

    for text in examples:
        result = predict(text, model, vocab)
        print(f"{result} <- {text!r}")
