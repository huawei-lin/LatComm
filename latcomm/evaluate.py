"""Response scoring and evaluation.

The loss is the cross-entropy of the gold response tokens under teacher forcing. For
multiple-choice data the prediction is the valid option letter with the highest logit at
the first response position; the gold answer is never an input to that prediction.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from latcomm.data import MCQ_LETTERS
from latcomm.distributed import DistributedContext
from latcomm.model import LatCommModel, response_logits
from latcomm.packets import round_to_stored


@dataclass
class Score:
    nll: torch.Tensor  # [B, R] negative log-likelihood of each response token, zero at padding
    token_counts: torch.Tensor  # [B] response tokens per example
    correct_tokens: torch.Tensor  # [B] response tokens predicted correctly under teacher forcing
    predictions: torch.Tensor | None  # [B] index of the predicted option, multiple choice only


def score_responses(logits: torch.Tensor, batch: Mapping[str, Any]) -> Score:
    response_ids = batch["response_ids"].to(logits.device)
    mask = batch["response_mask"].to(logits.device)
    labels = response_ids.masked_fill(~mask, -100)
    # Cast to 32-bit floats a few positions at a time; a full float copy of the logits is large.
    nll = torch.cat(
        [
            F.cross_entropy(
                logits[:, start : start + 32].float().flatten(0, 1),
                labels[:, start : start + 32].flatten(),
                reduction="none",
                ignore_index=-100,
            ).view(logits.shape[0], -1)
            for start in range(0, logits.shape[1], 32)
        ],
        dim=1,
    )
    predictions = None
    if "option_ids" in batch:
        option_logits = logits[:, 0].index_select(-1, batch["option_ids"].to(logits.device))
        option_logits = option_logits.masked_fill(~batch["option_mask"].to(logits.device), -torch.inf)
        predictions = option_logits.argmax(dim=-1)
    return Score(
        nll=nll,
        token_counts=mask.sum(dim=1),
        correct_tokens=((logits.argmax(dim=-1) == response_ids) & mask).sum(dim=1),
        predictions=predictions,
    )


def records(batch: Mapping[str, Any], score: Score) -> list[dict[str, Any]]:
    """One JSON-serializable record per example."""
    rows = [
        {"sample_id": sample_id, "dataset": dataset, "nll": nll, "tokens": tokens, "correct_tokens": correct}
        for sample_id, dataset, nll, tokens, correct in zip(
            batch["ids"],
            batch["datasets"],
            score.nll.sum(dim=1).tolist(),
            score.token_counts.tolist(),
            score.correct_tokens.tolist(),
        )
    ]
    if score.predictions is not None:
        for row, predicted, gold in zip(rows, score.predictions.tolist(), batch["gold_options"].tolist()):
            row.update(prediction=MCQ_LETTERS[predicted], gold=MCQ_LETTERS[gold], correct=int(predicted == gold))
    return rows


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Token-level loss and accuracy; for multiple choice also per-dataset accuracy.

    `accuracy` is over all questions. `macro_accuracy` is the unweighted mean over datasets.
    """
    rows = list(rows)
    tokens = sum(row["tokens"] for row in rows)
    metrics = {
        "loss": sum(row["nll"] for row in rows) / tokens,
        "token_accuracy": sum(row["correct_tokens"] for row in rows) / tokens,
    }
    if "correct" in rows[0]:
        by_dataset: dict[str, list[int]] = {}
        for row in rows:
            by_dataset.setdefault(row["dataset"], []).append(row["correct"])
        per_dataset = {name: sum(values) / len(values) for name, values in sorted(by_dataset.items())}
        metrics["accuracy"] = sum(row["correct"] for row in rows) / len(rows)
        metrics["macro_accuracy"] = sum(per_dataset.values()) / len(per_dataset)
        metrics.update({f"accuracy/{name}": value for name, value in per_dataset.items()})
    return metrics


@torch.no_grad()
def evaluate(
    model: LatCommModel,
    loader: Iterable[Mapping[str, Any]],
    distributed: DistributedContext,
    *,
    packet_dtype: str = "float32",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Score every example with its packet. Each process handles its own shard of the data.

    Packets come from storage when the batch carries them. Otherwise the context is encoded
    now and rounded to `packet_dtype`, so the receiver reads what a stored packet would hold.
    """
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        if "packet" in batch:
            packet = batch["packet"]
        else:
            packet = round_to_stored(model.encode_context(batch["context"]), packet_dtype)
        with model.reading(packet, batch["prompt"]):
            logits = response_logits(model.receiver, batch["prompt"])
        rows.extend(records(batch, score_responses(logits, batch)))
    model.train(was_training)
    rows = [row for shard in distributed.all_gather(rows) for row in shard]
    return summarize(rows), rows
