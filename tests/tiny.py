"""A small causal transformer and a character tokenizer for CPU tests.

Both follow the Hugging Face interfaces that LatComm uses, so the tests run without
downloading a model.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TinyLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size), nn.GELU(), nn.Linear(2 * hidden_size, hidden_size)
        )

    def forward(self, hidden: torch.Tensor, padding: torch.Tensor) -> tuple[torch.Tensor]:
        length = hidden.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=hidden.device), diagonal=1)
        states = self.attention_norm(hidden)
        attended, _ = self.attention(
            states, states, states, attn_mask=causal, key_padding_mask=padding, need_weights=False
        )
        hidden = hidden + attended
        return (hidden + self.mlp(self.mlp_norm(hidden)),)


class TinyLM(nn.Module):
    """Causal language model with `model.layers`, `config`, and `logits_to_keep`."""

    def __init__(self, vocab_size: int = 80, hidden_size: int = 32, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_layers, use_cache=True)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList(TinyLayer(hidden_size, num_heads) for _ in range(num_layers))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.gradient_checkpointing = True

    def enable_input_require_grads(self) -> None:
        pass

    def forward(self, input_ids, attention_mask, logits_to_keep=None, use_cache=False, return_dict=True):
        hidden = self.model.embed_tokens(input_ids)
        padding = ~attention_mask.bool()
        for layer in self.model.layers:
            if self.gradient_checkpointing and self.training:
                # Forward hooks of the layer run inside the checkpoint, as in Hugging Face models.
                (hidden,) = checkpoint(layer.__call__, hidden, padding, use_reentrant=False)
            else:
                (hidden,) = layer(hidden, padding)
        if isinstance(logits_to_keep, int):
            hidden = hidden[:, -logits_to_keep:]
        elif logits_to_keep is not None:
            hidden = hidden[:, logits_to_keep]
        return SimpleNamespace(logits=self.lm_head(hidden))

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens=4, use_cache=True):
        """Greedy decoding without a key-value cache."""
        for _ in range(max_new_tokens):
            logits = self(input_ids, attention_mask, logits_to_keep=1).logits
            input_ids = torch.cat((input_ids, logits.argmax(dim=-1)), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1)
        return input_ids


class CharTokenizer:
    """One token per character. `add_special_tokens` prepends BOS."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [3 + (ord(character) % 77) for character in text]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}
