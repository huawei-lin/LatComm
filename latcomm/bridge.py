"""The LatComm interface: context compressors, latent readers, and per-layer, per-slot gates.

A packet is a tensor [P, S, d]: P layer pairs, S slots per pair, slot width d.
All interface modules compute in 32-bit floats.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F

from latcomm.config import BridgeConfig


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def select_layer_pairs(
    num_encoder_layers: int, num_receiver_layers: int, num_layer_pairs: int | float
) -> tuple[tuple[int, int], ...]:
    """Pick P layers uniformly across the depth of each model and pair them in order.

    With P > 1 the first and the last layer of each model are always selected.
    """
    max_pairs = min(num_encoder_layers, num_receiver_layers)
    if isinstance(num_layer_pairs, float):
        count = max(1, _round_half_up(max_pairs * num_layer_pairs))
    else:
        count = num_layer_pairs

    def uniform(num_layers: int) -> list[int]:
        if count == 1:
            return [_round_half_up((num_layers - 1) / 2)]
        return [_round_half_up(i * (num_layers - 1) / (count - 1)) for i in range(count)]

    return tuple(zip(uniform(num_encoder_layers), uniform(num_receiver_layers)))


class ContextCompressor(nn.Module):
    """Compress the encoder's token states at one layer into S slots (Section 3.1)."""

    def __init__(self, encoder_hidden_size: int, context_dim: int, num_slots: int, num_heads: int):
        super().__init__()
        self.source_projection = nn.Linear(encoder_hidden_size, context_dim, bias=False)
        self.latent_queries = nn.Parameter(torch.empty(num_slots, context_dim))
        self.attention = nn.MultiheadAttention(context_dim, num_heads, batch_first=True, bias=False)
        self.attention_norm = nn.LayerNorm(context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2), nn.SiLU(), nn.Linear(context_dim * 2, context_dim)
        )
        self.mlp_norm = nn.LayerNorm(context_dim)
        nn.init.xavier_uniform_(self.source_projection.weight)
        nn.init.normal_(self.latent_queries, std=1.0 / math.sqrt(context_dim))
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, encoder_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.source_projection(encoder_states.to(self.source_projection.weight.dtype))
        queries = self.latent_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(
            queries, tokens, tokens, key_padding_mask=~attention_mask.bool(), need_weights=False
        )
        slots = self.attention_norm(queries + attended)
        return self.mlp_norm(slots + self.mlp(slots))


class LatentReader(nn.Module):
    """Cross-attention from every receiver token to the gated slots (Section 3.2)."""

    def __init__(self, receiver_hidden_size: int, context_dim: int, num_heads: int):
        super().__init__()
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.query_norm = nn.LayerNorm(receiver_hidden_size)
        self.latent_norm = nn.LayerNorm(context_dim)
        self.query_projection = nn.Linear(receiver_hidden_size, context_dim, bias=False)
        self.key_projection = nn.Linear(context_dim, context_dim, bias=False)
        self.value_projection = nn.Linear(context_dim, context_dim, bias=False)
        self.output_projection = nn.Linear(context_dim, receiver_hidden_size, bias=False)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        # The initial residual update is zero.
        nn.init.zeros_(self.output_projection.weight)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, -1).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, slots: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.query_projection.weight.dtype)
        query = self._split_heads(self.query_projection(self.query_norm(hidden)))
        slots = self.latent_norm(slots)
        key = self._split_heads(self.key_projection(slots))
        # The gate scales each slot's value after normalization and value projection.
        # Attention weights are not renormalized.
        value = self._split_heads(self.value_projection(slots) * gate.unsqueeze(-1))
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], self.context_dim)
        return self.output_projection(attended)


class PromptPool(nn.Module):
    """Attention pooling over the receiver's prompt states (Appendix A)."""

    def __init__(self, receiver_hidden_size: int, pooling_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(receiver_hidden_size)
        self.projection = nn.Linear(receiver_hidden_size, pooling_dim)
        self.score = nn.Linear(pooling_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.score.weight)

    def forward(self, hidden: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        # Response and padding positions get zero pooling weight.
        states = hidden[:, : prompt_mask.shape[1]].to(self.projection.weight.dtype)
        states = self.norm(states.masked_fill(~prompt_mask.unsqueeze(-1), 0.0))
        scores = self.score(torch.tanh(self.projection(states))).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~prompt_mask, -torch.inf), dim=1)
        return (states * weights.unsqueeze(-1)).sum(dim=1)


class LatentBridge(nn.Module):
    """One compressor and one reader per layer pair, with gates shared across pairs.

    The pooling module and the gate projections are shared by all layer pairs.
    Every layer-slot pair has its own gate bias.
    """

    def __init__(
        self,
        layer_pairs: Sequence[tuple[int, int]],
        encoder_hidden_size: int,
        receiver_hidden_size: int,
        config: BridgeConfig,
    ):
        super().__init__()
        self.layer_pairs = tuple(layer_pairs)
        self.context_dim = config.context_dim
        self.num_slots = config.num_slots
        self.compressors = nn.ModuleList(
            ContextCompressor(
                encoder_hidden_size, config.context_dim, config.num_slots, config.num_attention_heads
            )
            for _ in self.layer_pairs
        )
        self.readers = nn.ModuleList(
            LatentReader(receiver_hidden_size, config.context_dim, config.num_attention_heads)
            for _ in self.layer_pairs
        )
        self.gate_pool = PromptPool(receiver_hidden_size, config.gate_pooling_dim)
        self.gate_query = nn.Linear(receiver_hidden_size, config.context_dim, bias=False)
        self.gate_key = nn.Linear(config.context_dim, config.context_dim, bias=False)
        self.gate_bias = nn.Parameter(
            torch.full((len(self.layer_pairs), config.num_slots), float(config.gate_init_bias))
        )
        nn.init.xavier_uniform_(self.gate_query.weight)
        nn.init.xavier_uniform_(self.gate_key.weight)

    @property
    def packet_shape(self) -> tuple[int, int, int]:
        return (len(self.layer_pairs), self.num_slots, self.context_dim)

    def compress(
        self, encoder_states: Sequence[torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Map the encoder states of the paired layers to a packet batch [B, P, S, d]."""
        slots = [
            compressor(states, attention_mask)
            for compressor, states in zip(self.compressors, encoder_states, strict=True)
        ]
        return torch.stack(slots, dim=1)

    def gate(
        self, pair: int, slots: torch.Tensor, hidden: torch.Tensor, prompt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Gates in (0, 1) for the slots of one layer pair (Section 3.2), shape [B, S]."""
        query = self.gate_query(self.gate_pool(hidden, prompt_mask))
        keys = self.gate_key(slots)
        logits = (query.unsqueeze(1) * keys).sum(dim=-1) / math.sqrt(self.context_dim)
        return torch.sigmoid(logits + self.gate_bias[pair])

    def reader_parameters(self) -> Iterator[nn.Parameter]:
        """Readers and gates: the parameters trained in Stage 2."""
        yield from self.readers.parameters()
        yield from self.gate_pool.parameters()
        yield from self.gate_query.parameters()
        yield from self.gate_key.parameters()
        yield self.gate_bias
