"""Datasets, prompts, batching, and deterministic sampling.

Every data file is JSON Lines. One record is

    {"sample_id": str, "question": str, "choices": [str, ...],
     "evidence": [{"source_record_id": str, "text": str}],
     "response": str, "provenance": {"dataset": str, ...}}

`evidence` holds the context; several passages are joined with a blank line. For
multiple-choice data `response` is the gold option letter. For reconstruction data
`question` is the instruction, `response` is the passage, and there are no `choices`.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import torch
from torch.utils.data import Sampler

from latcomm.config import DataConfig

MCQ_LETTERS = ascii_uppercase


@dataclass(frozen=True)
class Example:
    sample_id: str
    dataset: str
    context: str
    prompt: str
    response: str
    num_choices: int = 0


def render_mcq_prompt(question: str, choices: Sequence[str], template: str) -> str:
    letters = MCQ_LETTERS[: len(choices)]
    return template.format(
        question=question,
        choices="\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices)),
        letters=", ".join(letters[:-1]) + ", or " + letters[-1],
    )


def load_examples(path: str | Path, data: DataConfig) -> list[Example]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if data.response_format == "mcq":
                prompt = render_mcq_prompt(record["question"], record["choices"], data.mcq_template)
                num_choices = len(record["choices"])
            else:
                prompt, num_choices = record["question"], 0
            examples.append(
                Example(
                    sample_id=record["sample_id"],
                    dataset=record["provenance"]["dataset"],
                    context="\n\n".join(item["text"] for item in record["evidence"]),
                    prompt=prompt,
                    response=record["response"],
                    num_choices=num_choices,
                )
            )
    return examples


def load_split(data: DataConfig, split: str) -> list[Example]:
    return load_examples(Path(data.dataset_root) / f"{split}.jsonl", data)


def tokenize(tokenizer: Any, text: str, limit: int, what: str, sample_id: str) -> list[int]:
    """Token ids of a context or prompt, with the tokenizer's own prefix tokens such as BOS."""
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(ids) > limit:
        raise ValueError(f"{sample_id}: {what} has {len(ids)} tokens, limit is {limit}; inputs are never truncated")
    return ids


def pad(sequences: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = 1
    return ids, mask


def _teacher_forced(
    prefixes: Sequence[list[int]], responses: Sequence[list[int]], pad_id: int
) -> dict[str, torch.Tensor]:
    """Pack prefix + response rows and the positions that predict each response token."""
    input_ids, attention_mask = pad([p + r for p, r in zip(prefixes, responses)], pad_id)
    positions, _ = pad(
        [list(range(len(p) - 1, len(p) + len(r) - 1)) for p, r in zip(prefixes, responses)], 0
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prediction_positions": positions,
        "prompt_end_positions": torch.tensor([len(p) - 1 for p in prefixes], dtype=torch.long),
    }


class Collator:
    """Tokenize a list of examples into one batch.

    `context`  encoder tokens of the context (only when `with_context`).
    `packet`   stored packets [B, P, S, d] (only when a packet source is given).
    `prompt`   receiver tokens of the prompt followed by the gold response.
    """

    def __init__(
        self,
        encoder_tokenizer: Any,
        receiver_tokenizer: Any,
        data: DataConfig,
        *,
        with_context: bool = True,
        packets: Callable[[Sequence[Example]], torch.Tensor] | None = None,
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.receiver_tokenizer = receiver_tokenizer
        self.data = data
        self.with_context = with_context
        self.packets = packets

    def _response(self, example: Example) -> list[int]:
        ids = self.receiver_tokenizer(example.response, add_special_tokens=False)["input_ids"]
        if self.data.response_format == "mcq":
            return ids
        ids = ids + [self.receiver_tokenizer.eos_token_id]
        if len(ids) > self.data.max_response_length:
            raise ValueError(f"{example.sample_id}: response has {len(ids)} tokens")
        return ids

    def __call__(self, examples: Sequence[Example]) -> dict[str, Any]:
        receiver, data = self.receiver_tokenizer, self.data
        prompts = [
            tokenize(receiver, e.prompt, data.max_prompt_length, "prompt", e.sample_id) for e in examples
        ]
        responses = [self._response(e) for e in examples]
        response_ids, response_mask = pad(responses, 0)
        batch: dict[str, Any] = {
            "ids": [e.sample_id for e in examples],
            "datasets": [e.dataset for e in examples],
            "prompt": _teacher_forced(prompts, responses, receiver.pad_token_id),
            "response_ids": response_ids,
            "response_mask": response_mask.bool(),
        }
        if data.response_format == "mcq":
            num_choices = torch.tensor([e.num_choices for e in examples])
            letters = MCQ_LETTERS[: int(num_choices.max())]
            # One token per option letter; options beyond an example's own choices are masked.
            batch["option_ids"] = torch.tensor(
                [receiver(letter, add_special_tokens=False)["input_ids"][0] for letter in letters]
            )
            batch["option_mask"] = torch.arange(len(letters))[None, :] < num_choices[:, None]
            batch["gold_options"] = torch.tensor([MCQ_LETTERS.index(e.response) for e in examples])
        if self.with_context:
            contexts = [
                tokenize(self.encoder_tokenizer, e.context, data.max_context_length, "context", e.sample_id)
                for e in examples
            ]
            input_ids, attention_mask = pad(contexts, self.encoder_tokenizer.pad_token_id)
            batch["context"] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.packets is not None:
            batch["packet"] = self.packets(examples)
        return batch


class ShuffleSampler(Sampler[int]):
    """A seeded shuffle per epoch, split into equal shards across training processes.

    A few indices are repeated when the dataset size is not a multiple of the number of
    processes, so that every process runs the same number of steps.
    """

    def __init__(self, size: int, *, seed: int, rank: int = 0, world_size: int = 1):
        self.size = size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.num_samples = math.ceil(size / world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        total = self.num_samples * self.world_size
        indices += (indices * math.ceil(total / self.size))[: total - self.size]
        return iter(indices[self.rank : total : self.world_size])

    def __len__(self) -> int:
        return self.num_samples


class LengthSortedBatchSampler(Sampler[list[int]]):
    """Sort the examples of each optimizer update by length before cutting microbatches.

    An update still holds exactly the examples that `accumulation` consecutive microbatches
    of the wrapped sampler would give; only the padding inside the update shrinks.
    """

    def __init__(self, sampler: ShuffleSampler, lengths: Sequence[int], *, batch_size: int, accumulation: int):
        self.sampler = sampler
        self.lengths = lengths
        self.batch_size = batch_size
        self.group_size = batch_size * accumulation

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(self.sampler)
        for start in range(0, len(indices), self.group_size):
            group = sorted(indices[start : start + self.group_size], key=self.lengths.__getitem__)
            for offset in range(0, len(group), self.batch_size):
                yield group[offset : offset + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)


def length_batches(
    lengths: Sequence[Sequence[int]], budgets: Sequence[int], max_examples: int
) -> list[list[int]]:
    """Group examples of similar length into batches for evaluation.

    `lengths[i]` holds the token counts of example i, one per budget. A batch has at most
    `max_examples` examples, and for each count at most `budget` tokens after padding.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda i: tuple(lengths[i])):
        candidate = current + [index]
        too_large = len(candidate) > max_examples or any(
            len(candidate) * max(lengths[i][k] for i in candidate) > budget for k, budget in enumerate(budgets)
        )
        if current and too_large:
            batches.append(current)
            current = [index]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches
