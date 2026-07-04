"""
Phase 3: the training loop for the IMDB sentiment classifier.

Reuses IMDBDataset/collate_batch/load_vocab from dataset.py (Phase 1) and
SentimentClassifier from model.py (Phase 2) as-is. This file is only
responsible for: splitting data, running epochs, tracking train/val
metrics, and saving the trained weights.
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import IMDBDataset, collate_batch, load_vocab
from model import PAD_IDX, SentimentClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_model.pt")

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.1  # 10% of the official train split held out for validation


def main():
    # Reuse the exact vocab saved during Phase 1 rather than rebuilding it.
    # The embedding table's size (and every index's meaning) depends on this
    # specific word->index mapping, so training must use the same one the
    # model's vocab_size was sized against.
    vocab = load_vocab()

    # This is the OFFICIAL IMDB "train" split (25k reviews) — NOT the
    # official "test" split. We deliberately never touch the test set here.
    full_train_dataset = IMDBDataset(split="train", vocab=vocab, max_len=256)

    # Why not validate against the official test set instead of carving out
    # a slice of train?
    # The test set exists to answer "how will this model perform on data it
    # has truly never influenced in any way." If we used it during training
    # to decide things like "which epoch had the best model" or "should I
    # change embedding_dim," we'd be making design/tuning decisions based on
    # test-set performance — at that point the test set has effectively
    # leaked into model selection, and its accuracy number is no longer a
    # trustworthy estimate of real-world performance. A validation set
    # carved out of TRAIN is "spent" for tuning; the test set stays untouched
    # for one final, honest measurement after all decisions are locked in.
    val_size = int(len(full_train_dataset) * VAL_FRACTION)
    train_size = len(full_train_dataset) - val_size
    # A fixed generator seed makes the split reproducible across runs, so
    # re-running this script for comparison isn't also introducing random
    # noise from a different train/val split each time.
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # shuffle=True for training: without shuffling, the model would see
    # batches in the same fixed order every epoch. Since IMDB reviews are
    # stored with all negative reviews first and all positive reviews second
    # (or some other systematic order), an unshuffled DataLoader could hand
    # the model long runs of same-label batches, making each gradient step
    # correlated with recent steps instead of representative of the whole
    # dataset — this slows/destabilizes learning. Reshuffling every epoch
    # also means the model never memorizes a fixed batch composition.
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )

    # shuffle=False for validation: we never call backward()/optimizer.step()
    # during validation, so there's no gradient-correlation risk to avoid.
    # The validation loss/accuracy we print is an aggregate (sum/average)
    # over the whole set, and aggregates don't depend on the order items were
    # visited in — so shuffling would only add pointless randomness that
    # makes it harder to reproduce a specific run's evaluation deterministically.
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
    )

    # cuda if available so this scales to a GPU machine without code changes;
    # falls back to cpu so it still runs anywhere.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentimentClassifier(vocab_size=len(vocab), embedding_dim=100, pad_idx=PAD_IDX)
    model.to(device)

    # BCEWithLogitsLoss, mathematically, does two things in one fused step:
    # 1. sigmoid(logit) -> squashes the model's raw score into a (0, 1)
    #    probability that the review is positive.
    # 2. binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)], where y is the
    #    true 0/1 label and p is that probability. This penalizes the model
    #    more the further its predicted probability is from the true label
    #    (e.g. predicting p=0.01 when the true label is 1 is penalized much
    #    more heavily than predicting p=0.4).
    # It's fused rather than sigmoid() + a separate BCELoss because doing
    # sigmoid and log() as two distinct ops can overflow/underflow for very
    # confident-but-wrong predictions; the fused version uses a numerically
    # stable formulation internally that avoids ever computing log(0).
    criterion = nn.BCEWithLogitsLoss()

    # The loss function and the optimizer answer two different questions:
    # - criterion (loss) answers "how wrong is the model right now?" — it
    #   produces a single scalar number from (model output, true label), and
    #   .backward() on that scalar computes how much each individual weight
    #   in the model contributed to that wrongness (the gradient).
    # - optimizer answers "given those per-weight gradients, how should each
    #   weight actually be changed?" Adam doesn't just do
    #   weight -= gradient * lr like plain SGD — it keeps a running average
    #   of each parameter's past gradients and past squared gradients, and
    #   uses those to give each individual parameter its own effective step
    #   size. The loss never touches the weights directly; it only produces
    #   the gradients that the optimizer then acts on via optimizer.step().
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        # model.train() puts the model in "training mode." This model has no
        # dropout/batchnorm layers so it has no actual behavioral effect
        # right now, but it's still correct practice to call it every epoch
        # in case those layers are added later — forgetting it is a classic,
        # silent bug once such layers exist.
        model.train()
        running_train_loss = 0.0

        for token_ids, labels in train_loader:
            token_ids, labels = token_ids.to(device), labels.to(device)

            # zero_grad() BEFORE backward(): PyTorch accumulates
            # (adds to, not overwrites) gradients into each parameter's
            # .grad every time backward() is called. If we didn't zero them
            # out first, this batch's gradient would be ADDED on top of the
            # previous batch's leftover gradient, so the optimizer would step
            # using a corrupted sum of two unrelated batches' gradients
            # instead of this batch's gradient alone — training would
            # effectively use ever-growing, wrong step sizes and never
            # converge properly.
            optimizer.zero_grad()

            logits = model(token_ids)  # shape: (batch_size,) raw logits

            # BCEWithLogitsLoss expects float targets of the same shape as
            # the logits; labels come out of IMDBDataset as torch.long
            # (0 or 1), so we cast rather than change the Dataset's dtype
            # (long is the conventional dtype for class-index-like labels).
            loss = criterion(logits, labels.float())

            # backward() walks the computation graph built during the
            # forward pass and computes d(loss)/d(weight) for every weight,
            # storing each result in that weight's .grad attribute.
            loss.backward()

            # step() is where the weights actually change, using the
            # gradients .backward() just populated and Adam's per-parameter
            # adaptive update rule.
            optimizer.step()

            # loss.item() pulls the scalar loss out as a plain Python float
            # (detached from the graph); multiplying by batch size lets us
            # correctly average by TOTAL SAMPLES afterward, since the last
            # batch in a loader is often a smaller, uneven size.
            running_train_loss += loss.item() * token_ids.size(0)

        train_loss = running_train_loss / len(train_dataset)

        # model.eval() switches off training-mode-only behavior (again, a
        # no-op for this specific architecture, but required practice).
        model.eval()
        running_val_loss = 0.0
        correct = 0

        # torch.no_grad() tells autograd not to build a computation graph for
        # anything inside this block. During training, PyTorch has to record
        # every operation (and keep references to intermediate tensors) so
        # that backward() can later compute gradients from it — that
        # bookkeeping costs extra memory and compute. Validation never calls
        # backward() (we are not updating weights here, only measuring
        # performance), so that graph would be built and then simply
        # discarded, wasting memory/time for nothing. no_grad() skips
        # building it in the first place.
        with torch.no_grad():
            for token_ids, labels in val_loader:
                token_ids, labels = token_ids.to(device), labels.to(device)

                logits = model(token_ids)
                loss = criterion(logits, labels.float())
                running_val_loss += loss.item() * token_ids.size(0)

                # sigmoid(logit) >= 0.5 is equivalent to logit >= 0 (sigmoid
                # crosses 0.5 exactly at input 0), but writing it via sigmoid
                # keeps the "this is a probability threshold" reasoning
                # explicit rather than relying on that numerical coincidence.
                predicted_labels = (torch.sigmoid(logits) >= 0.5).long()
                correct += (predicted_labels == labels).sum().item()

        val_loss = running_val_loss / len(val_dataset)
        val_accuracy = correct / len(val_dataset)

        # Printed every epoch so you can watch, over time, whether: (a) the
        # model is learning at all (both losses trending down), and (b) it's
        # starting to overfit (train_loss keeps falling while val_loss
        # flattens or rises — the classic overfitting signature).
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_accuracy:.4f}"
        )

    # We save model.state_dict() (an OrderedDict of tensor name -> tensor
    # weights) rather than the whole model object via torch.save(model, ...).
    # Saving the whole object pickles a reference to the exact SentimentClassifier
    # class definition (its module path, source structure, etc.) — if you
    # later rename/move the class or refactor this file, loading that pickle
    # can break entirely. state_dict is just plain tensor data with no
    # dependency on the class code; loading it back requires you to
    # reconstruct a SentimentClassifier instance yourself first and then call
    # model.load_state_dict(...), which is more verbose but far more robust
    # across code changes — this is the standard recommended PyTorch pattern.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model weights to {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
