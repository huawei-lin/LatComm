"""Frozen LLMs and the LatComm model that connects them through the bridge."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import inspect
from typing import Any

import torch
from torch import nn
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from latcomm.bridge import LatentBridge, select_layer_pairs
from latcomm.config import LatCommConfig, ModelConfig

# Where the transformer layers live: multimodal checkpoints, then plain causal LMs.
LAYER_PATHS = ("model.language_model.layers", "model.layers")


@dataclass
class FrozenLM:
    """A frozen LLM. `model` is None when only its dimensions are needed."""

    model: nn.Module | None
    layers: tuple[nn.Module, ...]
    num_layers: int
    hidden_size: int
    device: torch.device


def load_tokenizer(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, revision=config.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _text_config(model_config: Any) -> Any:
    return getattr(model_config, "text_config", None) or model_config


def load_frozen_lm(
    config: ModelConfig, device: torch.device | str, *, load_weights: bool = True
) -> FrozenLM:
    device = torch.device(device)
    if not load_weights:
        text = _text_config(AutoConfig.from_pretrained(config.model_name_or_path, revision=config.revision))
        return FrozenLM(None, (), text.num_hidden_layers, text.hidden_size, device)
    if config.model_loader == "causal_lm":
        model_class = AutoModelForCausalLM
    else:
        model_class = transformers.AutoModelForImageTextToText
    kwargs: dict[str, Any] = {"revision": config.revision, "dtype": getattr(torch, config.dtype)}
    if config.attn_implementation is not None:
        kwargs["attn_implementation"] = config.attn_implementation
    model = model_class.from_pretrained(config.model_name_or_path, **kwargs).to(device)
    model.requires_grad_(False)
    model.eval()
    return wrap_frozen_lm(model, device)


def wrap_frozen_lm(model: nn.Module, device: torch.device | str) -> FrozenLM:
    for path in LAYER_PATHS:
        layers: Any = model
        for name in path.split("."):
            layers = getattr(layers, name, None)
        if layers is not None:
            break
    else:
        raise ValueError(f"cannot find the transformer layers of {type(model).__name__}")
    hidden_size = _text_config(model.config).hidden_size
    return FrozenLM(model, tuple(layers), len(layers), hidden_size, torch.device(device))


def _hidden_state(output: Any) -> torch.Tensor:
    return output if isinstance(output, torch.Tensor) else output[0]


def _with_hidden_state(output: Any, hidden: torch.Tensor) -> Any:
    return hidden if isinstance(output, torch.Tensor) else (hidden, *output[1:])


@torch.no_grad()
def encode_layer_states(
    lm: FrozenLM, layer_indices: Sequence[int], inputs: Mapping[str, torch.Tensor]
) -> list[torch.Tensor]:
    """Run the frozen LLM once and return the token states after the selected layers."""
    states: dict[int, torch.Tensor] = {}
    handles = [
        lm.layers[index].register_forward_hook(
            lambda _module, _inputs, output, index=index: states.__setitem__(index, _hidden_state(output))
        )
        for index in layer_indices
    ]
    # The language-model head is not needed; keep one position where the model allows it.
    kwargs = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(lm.model.forward).parameters else {}
    try:
        lm.model(
            input_ids=inputs["input_ids"].to(lm.device),
            attention_mask=inputs["attention_mask"].to(lm.device),
            use_cache=False,
            **kwargs,
        )
    finally:
        for handle in handles:
            handle.remove()
    return [states[index] for index in layer_indices]


def response_logits(lm: FrozenLM, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Logits [B, R, V] at the positions that predict each response token."""
    positions = inputs["prediction_positions"].to(lm.device)
    kept = torch.unique(positions)
    logits = lm.model(
        input_ids=inputs["input_ids"].to(lm.device),
        attention_mask=inputs["attention_mask"].to(lm.device),
        logits_to_keep=kept,
        use_cache=False,
        return_dict=True,
    ).logits
    rows = torch.arange(logits.shape[0], device=logits.device)[:, None]
    return logits[rows, torch.searchsorted(kept, positions)]


def prompt_mask(inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Mark the prompt positions [B, W]: no padding and no response tokens."""
    ends = inputs["prompt_end_positions"]
    width = int(ends.max()) + 1
    positions = torch.arange(width, device=ends.device)
    return inputs["attention_mask"][:, :width].bool() & (positions[None, :] <= ends[:, None])


class LatentInjector:
    """Let the paired receiver layers read a packet while the receiver runs.

    A forward hook on each paired layer takes the layer output before the reader's update,
    computes the gates of that layer from the prompt states, reads the slots, and adds the
    result to the layer output. With `cache_gates`, the gates computed during prompt
    prefill are reused for every generated token.

    Under gradient checkpointing the hooks run again during the backward pass, so the
    injector must stay open until `backward()` returns.
    """

    def __init__(
        self,
        bridge: LatentBridge,
        receiver: FrozenLM,
        packet: torch.Tensor,
        prompt_mask: torch.Tensor,
        *,
        cache_gates: bool = False,
    ):
        self.bridge = bridge
        self.receiver = receiver
        self.packet = packet.to(receiver.device)
        self.prompt_mask = prompt_mask.to(receiver.device)
        self.cache_gates = cache_gates
        # Gates of the latest forward pass, one [B, S] tensor per layer pair.
        self.gates: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _hook(self, pair: int):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
            hidden = _hidden_state(output)
            slots = self.packet[:, pair]
            if self.cache_gates and pair in self.gates:
                gate = self.gates[pair]
            else:
                gate = self.bridge.gate(pair, slots, hidden, self.prompt_mask)
                self.gates[pair] = gate.detach()
            update = self.bridge.readers[pair](hidden, slots, gate)
            return _with_hidden_state(output, hidden + update.to(hidden.dtype))

        return hook

    def __enter__(self) -> "LatentInjector":
        for pair, (_, receiver_layer) in enumerate(self.bridge.layer_pairs):
            self._handles.append(self.receiver.layers[receiver_layer].register_forward_hook(self._hook(pair)))
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class LatCommModel(nn.Module):
    """A frozen context encoder LLM, a frozen receiver LLM, and the trainable bridge.

    Only the bridge is registered as a submodule, so `parameters()`, `train()` and
    `state_dict()` cover the interface and never the two LLMs.
    """

    def __init__(self, encoder: FrozenLM, receiver: FrozenLM, bridge: LatentBridge):
        super().__init__()
        self.bridge = bridge
        self.encoder = encoder
        self.receiver = receiver
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        """Recompute receiver layers during the backward pass to save memory."""
        receiver = self.receiver.model
        receiver.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        receiver.enable_input_require_grads()
        receiver.config.use_cache = False
        self._gradient_checkpointing = True

    def train(self, mode: bool = True) -> "LatCommModel":
        super().train(mode)
        # The receiver checkpoints its layers only in training mode. Its weights stay frozen.
        if self.receiver.model is not None:
            self.receiver.model.train(mode and self._gradient_checkpointing)
        return self

    def encode_context(self, context: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode contexts into packets [B, P, S, d]. The question is never an input."""
        encoder_layers = [encoder_layer for encoder_layer, _ in self.bridge.layer_pairs]
        states = encode_layer_states(self.encoder, encoder_layers, context)
        return self.bridge.compress(states, context["attention_mask"].to(self.encoder.device))

    def reading(
        self, packet: torch.Tensor, prompt: Mapping[str, torch.Tensor], *, cache_gates: bool = False
    ) -> LatentInjector:
        """Context manager in which the receiver reads `packet` at every paired layer."""
        return LatentInjector(self.bridge, self.receiver, packet, prompt_mask(prompt), cache_gates=cache_gates)

    @torch.no_grad()
    def generate(self, prompt: Mapping[str, torch.Tensor], packet: torch.Tensor, **generation_kwargs: Any):
        """Generate one continuation per prompt from stored packets.

        `prompt` holds `input_ids` and `attention_mask` only; use left padding for a batch
        of prompts with different lengths. Each layer computes its gates once, on the full
        prompt, and reuses them while decoding.
        """
        input_ids = prompt["input_ids"].to(self.receiver.device)
        attention_mask = prompt["attention_mask"].to(self.receiver.device)
        injector = LatentInjector(self.bridge, self.receiver, packet, attention_mask.bool(), cache_gates=True)
        with injector:
            return self.receiver.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True, **generation_kwargs
            )


def build_model(
    config: LatCommConfig,
    device: torch.device | str,
    *,
    load_encoder: bool = True,
    load_receiver: bool = True,
) -> LatCommModel:
    """Build the two frozen LLMs and a freshly initialized bridge.

    A model that is not loaded still contributes its depth and hidden size, so the bridge
    always has the same shape. Stage 2 training reads stored packets and skips the encoder.
    """
    encoder = load_frozen_lm(config.encoder, device, load_weights=load_encoder)
    receiver = load_frozen_lm(config.receiver, device, load_weights=load_receiver)
    layer_pairs = select_layer_pairs(encoder.num_layers, receiver.num_layers, config.bridge.num_layer_pairs)
    bridge = LatentBridge(layer_pairs, encoder.hidden_size, receiver.hidden_size, config.bridge).to(device)
    return LatCommModel(encoder, receiver, bridge)
