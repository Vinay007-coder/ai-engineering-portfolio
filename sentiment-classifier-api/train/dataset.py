"""
Phase 1: data loading and preprocessing for the IMDB sentiment classifier.

Deliberately NOT using HuggingFace transformers/tokenizers here. The point of
this project is to understand what a tokenizer + vocab + Dataset are actually
doing under the hood, not to call a library that hides it. `datasets` is used
purely as a convenient way to *download* the raw (text, label) pairs — it is
not doing any NLP for us.

No model or training code lives in this file (that's Phase 2+).

The pure tokenize/vocab logic (tokenize, encode, build_vocab, save_vocab,
load_vocab, PAD_IDX/UNK_IDX) now lives in vocab_utils.py, which has no
`datasets` import. This file imports it from there rather than redefining it,
because IMDBDataset below is the only piece that actually needs `datasets` —
keeping that import isolated here means api/inference.py (which only needs
encode/load_vocab, never IMDBDataset) can avoid pulling in `datasets`/
`pyarrow` transitively at serving time.
"""

import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from vocab_utils import PAD_IDX, build_vocab, encode, load_vocab, save_vocab

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vocab.json")


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
