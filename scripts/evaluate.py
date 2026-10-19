"""Evaluate a trained interface on a multiple-choice test file.

    torchrun --nproc_per_node 8 scripts/evaluate.py \
        --config configs/stage2/qwen4b_to_qwen0.8b.yaml \
        --checkpoint outputs/stage2/qwen4b_to_qwen0.8b/checkpoints/final \
        --data data/stage2/test.jsonl --output results/qwen4b_to_qwen0.8b/task

The encoder compresses each context without the question, and the receiver answers from
the packet. Examples are grouped by length. Contexts and prompts are never truncated,
and the grouping does not change any question, option, or score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from latcomm.checkpoint import load_bridge
from latcomm.config import load_config
from latcomm.data import Collator, length_batches, load_examples
from latcomm.distributed import initialize_distributed
from latcomm.evaluate import evaluate
from latcomm.model import build_model, load_tokenizer

# Padded-token budgets of one batch.
CONTEXT_TOKENS = 65536
PROMPT_TOKENS = 32768
MAX_EXAMPLES = 32


def token_counts(tokenizer, texts: list[str]) -> list[int]:
    counts: list[int] = []
    for start in range(0, len(texts), 256):
        counts += [len(ids) for ids in tokenizer(texts[start : start + 256], add_special_tokens=True)["input_ids"]]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="bridge checkpoint directory")
    parser.add_argument("--data", required=True, help="JSON Lines file with multiple-choice questions")
    parser.add_argument("--output", required=True, help="directory for predictions.jsonl and metrics.json")
    parser.add_argument("--packet-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()

    distributed = initialize_distributed()
    config = load_config(args.config)
    encoder_tokenizer = load_tokenizer(config.encoder)
    receiver_tokenizer = load_tokenizer(config.receiver)
    examples = load_examples(args.data, config.data)[distributed.rank :: distributed.world_size]

    lengths = zip(
        token_counts(encoder_tokenizer, [example.context for example in examples]),
        token_counts(receiver_tokenizer, [example.prompt for example in examples]),
    )
    loader = DataLoader(
        examples,
        batch_sampler=length_batches(list(lengths), [CONTEXT_TOKENS, PROMPT_TOKENS], MAX_EXAMPLES),
        collate_fn=Collator(encoder_tokenizer, receiver_tokenizer, config.data),
        num_workers=config.data.num_workers,
    )
    model = build_model(config, distributed.device)
    load_bridge(args.checkpoint, model.bridge)
    metrics, rows = evaluate(model, loader, distributed, packet_dtype=args.packet_dtype)

    if distributed.is_main:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda row: row["sample_id"])
        (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    distributed.close()


if __name__ == "__main__":
    main()
