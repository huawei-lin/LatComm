"""Train the LatComm bridge between two frozen LLMs.

Stage 1 (context reconstruction) trains the compressors, readers and gates.
Stage 2 (task supervision) starts from the final Stage 1 bridge, keeps the compressors
fixed, reads stored packets, and trains the readers and gates only.

    python -m latcomm.train --config configs/stage1/qwen4b_to_qwen0.8b.yaml
    torchrun --nproc_per_node 8 -m latcomm.train --config configs/stage1/qwen4b_to_qwen0.8b.yaml

The loss of one optimizer update is the mean cross-entropy over all response tokens of
the update, across microbatches and processes.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from latcomm.checkpoint import load_bridge, save_bridge
from latcomm.config import LatCommConfig, load_config
from latcomm.data import Collator, Example, LengthSortedBatchSampler, ShuffleSampler, load_split
from latcomm.distributed import DistributedContext, initialize_distributed
from latcomm.evaluate import Score, evaluate, score_responses
from latcomm.model import LatCommModel, build_model, load_tokenizer, response_logits
from latcomm.packets import PacketCache

LOGGER = logging.getLogger("latcomm.train")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    examples: list[Example],
    collator: Collator,
    config: LatCommConfig,
    distributed: DistributedContext,
    *,
    training: bool,
) -> tuple[DataLoader, ShuffleSampler | None]:
    options: dict[str, Any] = dict(
        collate_fn=collator,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    if not training:
        # Every process evaluates its own shard; together the shards cover each example once.
        shard = examples[distributed.rank :: distributed.world_size]
        return DataLoader(shard, batch_size=config.training.per_device_eval_batch_size, **options), None
    sampler = ShuffleSampler(
        len(examples), seed=config.training.seed, rank=distributed.rank, world_size=distributed.world_size
    )
    batch_size = config.training.per_device_batch_size
    if config.data.sort_within_update:
        batch_sampler = LengthSortedBatchSampler(
            sampler,
            [len(example.context) for example in examples],
            batch_size=batch_size,
            accumulation=config.training.gradient_accumulation_steps,
        )
        return DataLoader(examples, batch_sampler=batch_sampler, **options), sampler
    return DataLoader(examples, batch_size=batch_size, sampler=sampler, **options), sampler


def build_scheduler(optimizer: torch.optim.Optimizer, config: LatCommConfig, total_steps: int):
    """Linear warmup, then a constant rate or a half-cosine decay to zero."""
    warmup = config.training.warmup_steps

    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        if config.training.scheduler == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def forward_backward(model: LatCommModel, batch: Mapping[str, Any]) -> tuple[Score, torch.Tensor]:
    """Add the gradient of one microbatch's summed token loss. Returns the score and mean gate."""
    packet = batch["packet"] if "packet" in batch else model.encode_context(batch["context"])
    with model.reading(packet, batch["prompt"]) as reader:
        score = score_responses(response_logits(model.receiver, batch["prompt"]), batch)
        # Backward stays inside the block: checkpointed receiver layers run their readers again.
        score.nll.sum().backward()
    return score, torch.stack(list(reader.gates.values())).mean()


def microbatches(loader: DataLoader, sampler: ShuffleSampler, num_epochs: int) -> Iterator[tuple[Any, bool]]:
    """Yield every microbatch together with a flag that marks the end of an epoch."""
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for index, batch in enumerate(loader, start=1):
            yield batch, index == len(loader)


def train(config: LatCommConfig) -> Path:
    distributed = initialize_distributed()
    device = distributed.device
    training = config.training
    output = Path(training.output_dir)
    if distributed.is_main:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2) + "\n")
    seed_everything(training.seed)

    # With stored packets the context encoder LLM is not needed and is not loaded.
    packets = PacketCache(config.data.packet_cache) if config.data.packet_cache else None
    model = build_model(config, device, load_encoder=packets is None)
    if training.init_checkpoint is not None:
        load_bridge(training.init_checkpoint, model.bridge)
    if training.trainable == "readers_and_gates":
        model.bridge.compressors.requires_grad_(False)
    if training.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainable = [parameter for parameter in model.bridge.parameters() if parameter.requires_grad]

    collator = Collator(
        load_tokenizer(config.encoder),
        load_tokenizer(config.receiver),
        config.data,
        with_context=packets is None,
        packets=packets,
    )
    train_loader, sampler = make_loader(
        load_split(config.data, config.data.train_split), collator, config, distributed, training=True
    )
    eval_loader, _ = make_loader(
        load_split(config.data, config.data.eval_split), collator, config, distributed, training=False
    )

    accumulation = training.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = training.max_steps or training.num_epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(trainable, lr=training.learning_rate, weight_decay=training.weight_decay)
    scheduler = build_scheduler(optimizer, config, total_steps)

    run = None
    if distributed.is_main:
        LOGGER.info("layer pairs (encoder, receiver): %s", list(model.bridge.layer_pairs))
        LOGGER.info("trainable parameters: %d", sum(parameter.numel() for parameter in trainable))
        LOGGER.info(
            "global batch size %d, %d updates",
            training.per_device_batch_size * accumulation * distributed.world_size,
            total_steps,
        )
        if config.wandb.enabled:
            import wandb

            run = wandb.init(
                project=config.wandb.project,
                name=config.wandb.name,
                mode=config.wandb.mode,
                tags=config.wandb.tags,
                dir=str(output),
                config=config.model_dump(mode="json"),
            )

    def log(kind: str, step: int, metrics: Mapping[str, float]) -> None:
        if not distributed.is_main:
            return
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"type": kind, "step": step, **metrics}) + "\n")
        if run is not None:
            run.log({f"{kind}/{name}": value for name, value in metrics.items()}, step=step)
        LOGGER.info("step %d %s %s", step, kind, " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))

    def run_evaluation(step: int) -> None:
        metrics, _ = evaluate(model, eval_loader, distributed)
        log("eval", step, metrics)

    def save(name: str, step: int) -> Path:
        directory = output / "checkpoints" / name
        if distributed.is_main:
            save_bridge(directory, model.bridge, config, step)
        return directory

    model.train()
    optimizer.zero_grad(set_to_none=True)
    run_evaluation(0)

    step = 0
    micro_steps = 0
    update_tokens = 0
    stats = dict.fromkeys(("nll", "tokens", "correct_tokens", "examples", "correct", "gate"), 0.0)
    last_log = time.monotonic()
    for batch, end_of_epoch in microbatches(train_loader, sampler, math.ceil(total_steps / steps_per_epoch)):
        score, gate = forward_backward(model, batch)
        examples = len(batch["ids"])
        tokens = int(score.token_counts.sum())
        update_tokens += tokens
        stats["nll"] += float(score.nll.sum())
        stats["tokens"] += tokens
        stats["correct_tokens"] += int(score.correct_tokens.sum())
        stats["examples"] += examples
        stats["gate"] += float(gate) * examples
        if score.predictions is not None:
            stats["correct"] += int((score.predictions.cpu() == batch["gold_options"]).sum())
        micro_steps += 1
        if micro_steps < accumulation and not end_of_epoch:
            continue

        # Turn the summed token losses of all processes into the mean over the update's tokens.
        total_tokens = distributed.sum(update_tokens)
        distributed.sum_gradients(trainable)
        for parameter in trainable:
            parameter.grad.div_(total_tokens)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, training.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        micro_steps = 0
        update_tokens = 0
        step += 1

        if step % training.log_every == 0:
            merged = {key: sum(shard[key] for shard in distributed.all_gather(stats)) for key in stats}
            now = time.monotonic()
            metrics = {
                "loss": merged["nll"] / merged["tokens"],
                "token_accuracy": merged["correct_tokens"] / merged["tokens"],
                "gate_mean": merged["gate"] / merged["examples"],
                "grad_norm": float(grad_norm),
                "lr": scheduler.get_last_lr()[0],
                "examples_per_second": merged["examples"] / (now - last_log),
            }
            if config.data.response_format == "mcq":
                metrics["accuracy"] = merged["correct"] / merged["examples"]
            log("train", step, metrics)
            stats = dict.fromkeys(stats, 0.0)
            last_log = now
        if step % training.save_every == 0:
            save(f"step-{step}", step)
        if step % training.eval_every == 0 or end_of_epoch or step == total_steps:
            run_evaluation(step)
        if step == total_steps:
            break

    final = save("final", step)
    if run is not None:
        run.finish()
    distributed.close()
    return final


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train the LatComm bridge.")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    train(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
