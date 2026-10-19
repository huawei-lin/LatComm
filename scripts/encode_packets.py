"""Encode contexts into stored packets with the frozen encoder and the Stage 1 compressors.

    torchrun --nproc_per_node 8 scripts/encode_packets.py \
        --config configs/stage2/qwen4b_to_qwen0.8b.yaml \
        --checkpoint outputs/stage1/qwen4b_to_qwen0.8b/checkpoints/final \
        --output packets/qwen4b_to_qwen0.8b

The encoder sees only the context, never the question. Identical contexts are encoded
once. Contexts are sorted by length, so a batch has little padding, and they are spread
over the processes by token count. Each process writes its own shards.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from latcomm.checkpoint import load_bridge
from latcomm.config import load_config
from latcomm.data import load_split, pad, tokenize
from latcomm.model import build_model, load_tokenizer
from latcomm.packets import PacketWriter, context_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="final Stage 1 checkpoint directory")
    parser.add_argument("--output", required=True, help="packet directory to create")
    parser.add_argument("--splits", nargs="+", default=["train", "fast_dev"])
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    config = load_config(args.config)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else "cpu"

    contexts: dict[str, str] = {}
    for split in args.splits:
        for example in load_split(config.data, split):
            contexts.setdefault(context_key(example.context), example.context)
    tokenizer = load_tokenizer(config.encoder)
    tokens = {
        key: tokenize(tokenizer, text, config.data.max_context_length, "context", key)
        for key, text in contexts.items()
    }

    # Longest contexts first; each one goes to the process with the fewest tokens so far.
    work = [0] * world_size
    own: list[str] = []
    for key in sorted(tokens, key=lambda key: (-len(tokens[key]), key)):
        process = work.index(min(work))
        work[process] += len(tokens[key])
        if process == rank:
            own.append(key)

    model = build_model(config, device, load_receiver=False)
    load_bridge(args.checkpoint, model.bridge)
    model.eval()
    writer = PacketWriter(args.output, rank, args.dtype)
    for start in range(0, len(own), args.batch_size):
        keys = own[start : start + args.batch_size]
        input_ids, attention_mask = pad([tokens[key] for key in keys], tokenizer.pad_token_id)
        with torch.no_grad():
            packets = model.encode_context({"input_ids": input_ids, "attention_mask": attention_mask})
        writer.add(keys, packets)
        if rank == 0 and start % (50 * args.batch_size) == 0:
            print(f"encoded {start + len(keys)} of {len(own)} contexts on process 0", flush=True)
    writer.close()
    if rank == 0:
        meta = {"dtype": args.dtype, "packet_shape": list(model.bridge.packet_shape), "contexts": len(contexts)}
        (Path(args.output) / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
