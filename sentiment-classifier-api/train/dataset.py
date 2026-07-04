"""
Phase 1: data loading and preprocessing for the IMDB sentiment classifier.

Deliberately NOT using HuggingFace transformers/tokenizers here. The point of
this project is to understand what a tokenizer + vocab + Dataset are actually
doing under the hood, not to call a library that hides it. `datasets` is used
purely as a convenient way to *download* the raw (text, label) pairs — it is
not doing any NLP for us.

No model or training code lives in this file (that's Phase 2+).
"""

import json
import os
import re
from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

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


class IMDBDataset(Dataset):
    """
    Wraps the HuggingFace `datasets` IMDB split as a PyTorch Dataset of
    (token_id_sequence, label) pairs, with every sequence forced to a fixed
    length via truncation/padding.

    Why fixed length (max_len) instead of leaving sequences variable-length?
    PyTorch's DataLoader batches samples by stacking them into a single
    tensor of shape (batch_size, seq_len) — that stacking operation requires
    every sample in the tensor to have the same seq_len. Reviews are natural
    language and vary hugely in length (a few sentences to several
    paragraphs), so without truncating/padding to one fixed max_len, you
    literally cannot construct that tensor. max_len=256 is a modeling choice
    (a tradeoff between covering enough of each review and keeping
    compute/memory bounded) — beyond that it's a plain engineering
    requirement of batched tensor ops.
    """

    def __init__(self, split, vocab, max_len=256):
        # split is a string like "train" or "test", passed straight through
        # to load_dataset — this class doesn't know or care about the
        # dataset's internal format beyond "text" and "label" fields.
        # "imdb" (no namespace) is the original canonical dataset repo, but
        # recent huggingface_hub/datasets versions dropped support for
        # resolving bare no-namespace repo ids, so we use the actively
        # maintained namespaced mirror with the identical (text, label) schema.
        self.data = load_dataset("stanfordnlp/imdb", split=split)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        # Required by the Dataset interface — DataLoader uses this to know
        # how many samples exist / how to build batches and shuffle indices.
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        token_ids = encode(example["text"], self.vocab)

        # Truncate first: if the review is longer than max_len, we simply
        # cut it off. This throws away information (whatever's past word
        # 256), but it's the simplest possible strategy and is a reasonable
        # bet since sentiment is often established early/throughout a review
        # rather than concentrated at the very end.
        token_ids = token_ids[: self.max_len]

        # Then pad: if the review is shorter than max_len, append PAD_IDX
        # until it reaches exactly max_len. This is what guarantees every
        # __getitem__ call returns a sequence of the *same* length, which is
        # what makes stacking them into a batch tensor possible without even
        # needing a custom collate_fn (though we still write one below for
        # clarity/practice with variable-length batching).
        pad_len = self.max_len - len(token_ids)
        token_ids = token_ids + [PAD_IDX] * pad_len

        # dtype=torch.long because these are indices into an embedding
        # table (nn.Embedding expects LongTensor inputs), not float features.
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
            example["label"], dtype=torch.long
        )


def collate_batch(batch):
    """
    Collate function for the DataLoader.

    Given our IMDBDataset already pads every sample to a fixed max_len, a
    plain default_collate (torch.stack) would work fine on its own. We write
    an explicit collate_fn anyway for two reasons:
    1. It makes the batching step visible and inspectable, rather than
       relying on an implicit default — the whole point of this project is
       seeing every step.
    2. It's the natural place to handle batching correctly if you ever
       change IMDBDataset to return variable-length sequences (e.g. for
       dynamic padding per-batch instead of a fixed global max_len) — you'd
       pad each batch to the length of its own longest sequence here instead.
    """
    token_id_seqs, labels = zip(*batch)
    # torch.stack requires all input tensors to already be the same shape,
    # which holds here because IMDBDataset guarantees every sequence is
    # exactly max_len long.
    token_ids_batch = torch.stack(token_id_seqs)
    labels_batch = torch.stack(labels)
    return token_ids_batch, labels_batch


if __name__ == "__main__":
    # Sanity-check block: build everything once and print shapes, so a human
    # can eyeball that the pipeline works end-to-end before writing any model
    # code that depends on it.
    from torch.utils.data import DataLoader

    print("Loading IMDB train split...")
    train_data = load_dataset("stanfordnlp/imdb", split="train")

    print("Building vocab (min_freq=5)...")
    # We build the vocab from the training split ONLY, never from
    # validation/test data. If test-set words influenced which words made it
    # into the vocab, that's a (subtle) form of information leakage from
    # data the model is supposed to be evaluated on as if unseen.
    vocab = build_vocab(train_data["text"], min_freq=5)
    print(f"Vocab size (including <PAD>/<UNK>): {len(vocab)}")

    save_vocab(vocab)
    print(f"Saved vocab to {os.path.abspath(VOCAB_PATH)}")

    dataset = IMDBDataset(split="train", vocab=vocab, max_len=256)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

    token_ids_batch, labels_batch = next(iter(loader))
    print(f"Batch token_ids shape: {tuple(token_ids_batch.shape)}")  # (32, 256)
    print(f"Batch labels shape: {tuple(labels_batch.shape)}")  # (32,)
