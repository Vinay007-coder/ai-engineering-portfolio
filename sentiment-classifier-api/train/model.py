"""
Phase 2: model architecture only for the IMDB sentiment classifier.

No training loop, no optimizer, no loss function here — just the
nn.Module definition. Phase 3 will write the training loop that uses this.
"""

import torch
import torch.nn as nn

VOCAB_SIZE = 32060  # size of the vocab saved by train/dataset.py (includes <PAD>/<UNK>)
PAD_IDX = 0


class SentimentClassifier(nn.Module):
    """
    Minimal "embedding bag" style classifier:
    token ids -> embeddings -> masked mean pool -> single linear layer -> logit.

    Why mean pooling instead of using the last token's hidden state (as an
    RNN-style model would)?
    - This model has no recurrence/attention at all — nn.Embedding just looks
      up a vector per token independently, with no notion of order or of one
      token influencing another. There is no "last hidden state" here, only
      a bag of per-token vectors, so pooling (mean) is how we turn "256
      separate word vectors" into "one vector for the whole review."
    - Mean pooling treats every (non-pad) word as equally important and is
      order-invariant, which is a real limitation (it can't tell "not good"
      from "good not" apart) — but it's simple, has no extra parameters, and
      is a reasonable baseline before reaching for an RNN/Transformer later.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=100, pad_idx=PAD_IDX):
        super().__init__()

        # padding_idx=0 tells nn.Embedding two things:
        # 1. The gradient for row 0 of the embedding table is always zeroed
        #    out during backprop, so the <PAD> vector never gets updated by
        #    training and stays at whatever it was initialized to. There is
        #    no "meaning" for padding to learn — it's a placeholder, not a
        #    word — so we don't want the optimizer wasting capacity on it or
        #    letting padding representations drift into meaningful regions.
        # 2. nn.Embedding initializes row `padding_idx` to all zeros at
        #    construction time (rather than the random init every other row
        #    gets). This matters below: our masked-mean-pool math would still
        #    work even if this weren't zeroed (we mask pad positions
        #    explicitly), but having it zeroed is a defensive convention that
        #    matches PyTorch's own expectation of what padding_idx means.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.pad_idx = pad_idx

        # One raw logit out. No sigmoid here — see forward()'s docstring for why.
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, token_ids):
        """
        token_ids: (batch_size, 256) LongTensor of vocab indices, 0 = <PAD>.
        Returns:   (batch_size,) FloatTensor of raw logits (no sigmoid applied).

        Why no sigmoid inside the model?
        - We train with nn.BCEWithLogitsLoss (in Phase 3), which applies
          sigmoid and binary cross-entropy in one fused, numerically stable
          operation (it uses the log-sum-exp trick internally to avoid
          overflow/underflow that computing sigmoid() then log() separately
          can hit for very confident/incorrect predictions). If we applied
          sigmoid here AND used BCEWithLogitsLoss, we'd be sigmoid-ing twice.
          Keeping the model's output as a raw logit keeps it decoupled from
          which loss function is used, and defers the "squash to a
          probability" decision to whoever consumes the logit (training loop
          or inference code), which is standard PyTorch practice.
        """
        # token_ids shape: (batch_size, 256) — integer indices, not floats.

        mask = (token_ids != self.pad_idx).float()
        # mask shape: (batch_size, 256) — 1.0 where a real token sits, 0.0 where it's <PAD>.
        # Computed from token_ids directly (not from the embeddings) because
        # comparing raw ids against pad_idx is unambiguous; after embedding,
        # every position is just a vector of floats and "is this padding?"
        # would no longer be a single obvious check on the tensor itself.

        embedded = self.embedding(token_ids)
        # embedded shape: (batch_size, 256, embedding_dim) — one embedding_dim
        # vector per token position, looked up independently per token id.

        mask_expanded = mask.unsqueeze(-1)
        # mask_expanded shape: (batch_size, 256, 1) — unsqueeze adds a
        # trailing dim of size 1 so this can broadcast against embedded's
        # last dim (embedding_dim) in the multiply below.

        masked_embedded = embedded * mask_expanded
        # masked_embedded shape: (batch_size, 256, embedding_dim) — pad
        # positions are zeroed out; real-token positions are unchanged
        # (mask_expanded broadcasts its single value across embedding_dim).

        summed = masked_embedded.sum(dim=1)
        # summed shape: (batch_size, embedding_dim) — sum over the sequence
        # dimension (dim=1). Padding contributes exactly 0 to this sum, so it
        # cannot dilute the total no matter how many pad tokens exist.

        token_counts = mask.sum(dim=1, keepdim=True)
        # token_counts shape: (batch_size, 1) — the number of REAL (non-pad)
        # tokens in each review. keepdim=True keeps this broadcastable
        # against `summed` in the division below, instead of collapsing to
        # (batch_size,) and needing another unsqueeze.

        mean_pooled = summed / token_counts
        # mean_pooled shape: (batch_size, embedding_dim) — dividing by the
        # REAL token count (not by 256) is what makes this a masked mean:
        # a 20-word review divides its sum by 20, not by 256, so its average
        # magnitude matches a 200-word review's average magnitude. Naive
        # mean pooling (summing embeddings including <PAD> rows, then
        # dividing by the fixed 256) would instead divide every review by
        # the same 256 regardless of its real length — a 20-word review
        # would have its true signal diluted by 236 padding vectors, shrinking
        # its mean toward whatever the pad embedding's value is and making
        # short reviews look artificially similar to each other regardless of
        # their actual sentiment.

        logits = self.fc(mean_pooled)
        # logits shape: (batch_size, 1) — nn.Linear(embedding_dim, 1) applied
        # to each review's pooled vector independently.

        return logits.squeeze(-1)
        # return shape: (batch_size,) — squeeze(-1) drops the trailing size-1
        # dim so the output matches label tensors of shape (batch_size,)
        # exactly, which is what BCEWithLogitsLoss expects for binary targets.


if __name__ == "__main__":
    # Sanity-check block: no real data, just a fake batch of the right shape
    # and dtype, to confirm the forward pass runs and produces the expected
    # output shape before wiring this into a real training loop.
    model = SentimentClassifier(vocab_size=VOCAB_SIZE, embedding_dim=100)

    batch_size = 32
    max_len = 256
    fake_batch = torch.randint(0, VOCAB_SIZE, (batch_size, max_len))
    # fake_batch shape: (32, 256) — matches what train/dataset.py's DataLoader produces.

    logits = model(fake_batch)
    print(f"Output shape: {tuple(logits.shape)}")  # expected: (32,)
