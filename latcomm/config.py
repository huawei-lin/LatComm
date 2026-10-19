"""Experiment configuration, loaded from a YAML file."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml

MCQ_TEMPLATE = (
    "Accurately answer the following question:\n"
    "\n{question}\n"
    "\nChoices:\n"
    "{choices}\n"
    "\nAnswer with only the letter of the correct choice ({letters}). "
    "Do not explain.\n"
    "\nThe correct answer is "
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelConfig(StrictModel):
    model_name_or_path: str
    revision: str | None = None
    # Qwen3.5 checkpoints are published as multimodal models; only their language model is used.
    model_loader: Literal["causal_lm", "multimodal_lm"] = "causal_lm"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    attn_implementation: str | None = "sdpa"


class BridgeConfig(StrictModel):
    # Slot width d.
    context_dim: int = 256
    # Number of layer pairs P: an integer count, or a fraction of the shallower model's depth.
    num_layer_pairs: int | float = 1.0
    # Slots per layer pair S.
    num_slots: int = 32
    num_attention_heads: int = 8
    gate_pooling_dim: int = 256
    gate_init_bias: float = 2.0


class DataConfig(StrictModel):
    dataset_root: str
    train_split: str = "train"
    eval_split: str = "fast_dev"
    # "text": the response is free text followed by EOS (context reconstruction).
    # "mcq": the response is one option letter (task supervision).
    response_format: Literal["text", "mcq"]
    mcq_template: str = MCQ_TEMPLATE
    max_context_length: int = 32768
    max_prompt_length: int = 4096
    max_response_length: int = 1152
    num_workers: int = 2
    # Sort the examples of each optimizer update by context length before cutting microbatches.
    sort_within_update: bool = False
    # Directory written by scripts/encode_packets.py. When set, packets are read from disk
    # and the context encoder LLM is not loaded.
    packet_cache: str | None = None


class TrainingConfig(StrictModel):
    output_dir: str
    seed: int = 42
    # Bridge checkpoint to start from. Stage 2 starts from the final Stage 1 checkpoint.
    init_checkpoint: str | None = None
    # Stage 1 trains compressors, readers and gates. Stage 2 trains readers and gates only.
    trainable: Literal["all", "readers_and_gates"] = "all"
    per_device_batch_size: int = 24
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    scheduler: Literal["constant", "cosine"] = "constant"
    gradient_checkpointing: bool = True
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500


class WandbConfig(StrictModel):
    enabled: bool = False
    project: str = "latcomm"
    name: str | None = None
    mode: Literal["online", "offline"] = "online"
    tags: list[str] = Field(default_factory=list)


class LatCommConfig(StrictModel):
    encoder: ModelConfig
    receiver: ModelConfig
    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    data: DataConfig
    training: TrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)


def load_config(path: str | Path) -> LatCommConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return LatCommConfig.model_validate(yaml.safe_load(handle))
