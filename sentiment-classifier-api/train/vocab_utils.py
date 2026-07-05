"""
Pure text/vocab utilities: tokenizing, building a vocab, and encoding text to
token ids. Deliberately has NO import of the `datasets` library (or torch) —
this is what api/inference.py imports at serving time, and pulling in
`datasets`/`pyarrow` there would bloat install size and memory for a
dependency that's only ever needed during training (to download IMDB via
IMDBDataset in dataset.py, which imports these from here).
"""

import json
import os
import re
from collections import Counter

# Reserved vocab indices. These two slots always exist regardless of what
# words show up in the data, so every other word's index is shifted by 2.
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# Path where the vocab gets persisted. The API (at inference time) must load
# this exact file rather than rebuilding a vocab from scratch — see the
# save_vocab() docstring for why that consistency is non-negotiable.
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vocab.json")


def tokenize(text):
    """
    A from-scratch word-level tokenizer: lowercase -> strip punctuation -> split on whitespace.

    Why this instead of a real tokenizer (spaCy, BPE, WordPiece, etc.)?
    - Lowercasing: "Great" and "great" should map to the same word. We are
      building a bag-of-words-ish vocab, not something that needs case as a
      signal (e.g. "US" vs "us"), so we trade away that signal for a smaller,
      denser vocab.
    - Stripping punctuation via regex: punctuation tokens like "." or "!"
      would otherwise pollute the vocab as separate "words" and add noise
      without much sentiment signal (this is a simplification — a real
      tokenizer might keep "!" as a signal since "great!!!" often correlates
      with strong sentiment).
    - Splitting on whitespace: the simplest possible way to break a string
      into "word-ish" units.

    Tradeoffs vs. a real tokenizer (e.g. HuggingFace's WordPiece/BPE):
    - No subword handling: an unseen word like "unbelievably" is either a
      known whole word or it's entirely <UNK> — a subword tokenizer could
      break it into "un" + "believ" + "ably" and still extract partial
      meaning. This is the single biggest limitation of word-level tokenizers.
    - Vocabulary size is unbounded by construction: every distinct spelling
      is a new vocab entry (typos, "movie" vs "movies" are different words),
      so vocab size grows with data size, whereas subword tokenizers cap
      vocab size by design.
    - No handling of contractions, hyphens, or unicode edge cases — "don't"
      becomes "dont" here because the apostrophe is punctuation. A real
      tokenizer has explicit rules (or learned merges) for this.
    - Upside: it's fully transparent and dependency-free. Every step is one
      line of code you can trace by hand, which is the whole point of this
      exercise.
    """
    text = text.lower()
    # \w matches [a-zA-Z0-9_]; anything else (punctuation, symbols) is dropped.
    text = re.sub(r"[^\w\s]", "", text)
    # .split() with no args splits on any whitespace run and drops empty strings,
    # which handles double spaces / tabs / newlines left over from HTML in the raw reviews.
    return text.split()


def build_vocab(texts, min_freq=5):
    """
    Build a word -> integer index mapping from a list of raw text strings.

    Why filter by min_freq (default 5) instead of keeping every word seen?
    - Rare words (appearing once or twice in the whole training set) give the
      model almost no signal to learn from — there isn't enough repetition
      for gradient descent to learn a meaningful embedding for them. They are
      mostly typos, names, or one-off phrasing.
    - Every extra vocab entry costs one row in the embedding table, which is
      the model's most memory-heavy component. A larger vocab with lots of
      near-useless rows increases overfitting risk (more trainable parameters
      chasing the same amount of training signal) without meaningfully
      improving accuracy.
    - Mapping all of those rare words to a single <UNK> index instead means
      the model still gets *a* signal ("this is some word I don't know",
      which itself can correlate with sentiment) at a fraction of the
      parameter cost.

    Why reserve index 0 for <PAD> and 1 for <UNK>?
    - <PAD>: sequences in a batch have different lengths (reviews vary in
      length), but tensors in a batch must all be the same shape. <PAD> is a
      dummy "no word here" token appended to the end of shorter sequences so
      every sequence in a batch can be stacked into one rectangular tensor.
      Index 0 is a common convention so it's easy to recognize/mask later
      (e.g. `(token_ids != 0)` gives you a padding mask for free).
    - <UNK>: at inference time (or on the test set), we will see words that
      never appeared in the training vocab. Without a designated fallback,
      converting text to ids would simply crash on unseen words. <UNK> gives
      every possible word *some* valid index.
    - Both are reserved *before* assigning indices to real words, which is
      why every real word's index starts at 2.
    """
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))

    # Sort by frequency (descending) so the most common words get the
    # smallest indices — purely a convention for readability/debugging, it
    # has no effect on model behavior.
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    next_idx = 2
    for word, freq in counts.most_common():
        if freq < min_freq:
            # Counter.most_common() is sorted descending, so once we hit a
            # word below min_freq, every word after it is too — but we don't
            # bother breaking early since building the Counter already cost
            # us the full pass over the data, and the loop itself is cheap.
            continue
        vocab[word] = next_idx
        next_idx += 1

    return vocab


def save_vocab(vocab, path=VOCAB_PATH):
    """
    Persist the vocab to disk as JSON.

    Why this matters: the vocab maps "the word 'terrible'" to, say, index
    847 — but that mapping is arbitrary and depends on exactly which words
    were seen and in what order during training. If the API rebuilt its own
    vocab independently (even from the same dataset), word 847 could mean a
    completely different word. The model's embedding layer has learned
    "index 847 means something negative" — so at inference time we MUST use
    the identical word->index mapping that was used during training, or
    every prediction becomes garbage. Saving this file once and reusing it
    everywhere is what guarantees that.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f)


def load_vocab(path=VOCAB_PATH):
    """Load a previously saved vocab (used by the API at inference time)."""
    with open(path) as f:
        return json.load(f)


def encode(text, vocab):
    """
    Convert a raw text string into a list of integer token ids using vocab.

    Unseen words (not in vocab, whether because they were too rare during
    training or never seen at all) fall back to UNK_IDX via dict.get's
    default — this is the same fallback mechanism <UNK> was reserved for.
    """
    return [vocab.get(token, UNK_IDX) for token in tokenize(text)]
