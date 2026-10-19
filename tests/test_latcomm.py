"""CPU tests on a small random causal transformer. No model download is needed."""
from __future__ import annotations

import json

import pytest
import torch
from tiny import CharTokenizer, TinyLM

from latcomm import train as train_module
from latcomm.bridge import LatentBridge, select_layer_pairs
from latcomm.checkpoint import load_bridge, save_bridge
from latcomm.config import BridgeConfig, LatCommConfig
from latcomm.data import Collator, LengthSortedBatchSampler, ShuffleSampler, length_batches, load_split
from latcomm.evaluate import score_responses
from latcomm.model import LatCommModel, response_logits, wrap_frozen_lm
from latcomm.packets import PacketCache, PacketWriter, context_key, from_stored, round_to_stored, to_stored

BRIDGE = dict(context_dim=16, num_slots=4, num_attention_heads=4, gate_pooling_dim=8)
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()


def build_tiny_model(config: LatCommConfig | None = None, device="cpu", **_kwargs) -> LatCommModel:
    encoder = wrap_frozen_lm(TinyLM(hidden_size=48, num_layers=6).requires_grad_(False).eval(), device)
    receiver = wrap_frozen_lm(TinyLM(hidden_size=32, num_layers=4).requires_grad_(False).eval(), device)
    bridge_config = config.bridge if config is not None else BridgeConfig(**BRIDGE)
    pairs = select_layer_pairs(encoder.num_layers, receiver.num_layers, bridge_config.num_layer_pairs)
    return LatCommModel(encoder, receiver, LatentBridge(pairs, 48, 32, bridge_config))


def write_split(path, kind: str, count: int) -> None:
    generator = torch.Generator().manual_seed(len(str(path)) + count)
    rows = []
    for index in range(count):
        picks = torch.randint(len(WORDS), (8,), generator=generator).tolist()
        context = " ".join(WORDS[pick] for pick in picks[: 3 + index % 5])
        row = {
            "sample_id": f"{path.stem}-{index}",
            "evidence": [{"source_record_id": str(index), "text": context}],
            "provenance": {"dataset": ("first", "second")[index % 2]},
        }
        if kind == "mcq":
            choices = [WORDS[pick] for pick in picks[4 : 4 + (2, 3, 4)[index % 3]]]
            row.update(question="Which word is in the context?", choices=choices, response="ABCD"[index % len(choices)])
        else:
            row.update(question="Repeat the passage.\n\nPassage:\n", response=context)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_config(tmp_path, kind: str, **training) -> LatCommConfig:
    root = tmp_path / kind
    for split, count in (("train", 20), ("fast_dev", 6)):
        write_split(root / f"{split}.jsonl", kind, count)
    return LatCommConfig.model_validate(
        {
            "encoder": {"model_name_or_path": "tiny-encoder"},
            "receiver": {"model_name_or_path": "tiny-receiver"},
            "bridge": BRIDGE,
            "data": {"dataset_root": str(root), "response_format": kind, "num_workers": 0},
            "training": {
                "output_dir": str(tmp_path / f"run-{kind}"),
                "per_device_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "log_every": 1,
                "eval_every": 2,
                "save_every": 2,
                **training,
            },
        }
    )


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(train_module, "build_model", build_tiny_model)
    monkeypatch.setattr(train_module, "load_tokenizer", lambda _config: CharTokenizer())


def test_layer_pairs_cover_both_depths():
    assert select_layer_pairs(24, 24, 1.0) == tuple((i, i) for i in range(24))
    pairs = select_layer_pairs(24, 26, 1.0)
    assert len(pairs) == 24 and pairs[0] == (0, 0) and pairs[-1] == (23, 25)
    assert select_layer_pairs(40, 16, 1.0)[-1] == (39, 15)
    assert len(select_layer_pairs(24, 24, 0.5)) == 12 and len(select_layer_pairs(24, 24, 6)) == 6


def test_samplers_keep_the_examples_of_each_update():
    shards = [list(ShuffleSampler(37, seed=1, rank=rank, world_size=4)) for rank in range(4)]
    assert all(len(shard) == 10 for shard in shards)
    assert set().union(*shards) == set(range(37))
    sampler = ShuffleSampler(37, seed=1)
    lengths = [(index * 7) % 11 for index in range(37)]
    batches = list(LengthSortedBatchSampler(sampler, lengths, batch_size=4, accumulation=3))
    order = list(sampler)
    for update in range(0, len(batches), 3):
        members = [index for batch in batches[update : update + 3] for index in batch]
        assert sorted(members) == sorted(order[update * 4 : update * 4 + 12])
        assert [lengths[index] for index in members] == sorted(lengths[index] for index in members)
    batches = length_batches([[5, 2], [50, 2], [6, 3], [49, 2]], budgets=[100, 100], max_examples=3)
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
    assert all(len(batch) * max((5, 50, 6, 49)[index] for index in batch) <= 100 for batch in batches)


def test_packet_storage_round_trip(tmp_path):
    packets = torch.randn(5, 3, 4, 16) * torch.logspace(-3, 3, 3).view(1, 3, 1, 1)
    assert torch.equal(from_stored(to_stored(packets, "float32"), "float32"), packets)
    stored = from_stored(to_stored(packets, "bfloat16"), "bfloat16")
    assert torch.equal(stored, round_to_stored(packets, "bfloat16"))
    assert ((stored - packets).abs() <= packets.abs() * 2**-8).all()
    assert to_stored(packets, "bfloat16").nbytes * 2 == to_stored(packets, "float32").nbytes


def test_model_reads_packets_and_trains_only_the_bridge(tmp_path):
    torch.manual_seed(0)
    config = make_config(tmp_path, "mcq")
    model = build_tiny_model(config)
    collator = Collator(CharTokenizer(), CharTokenizer(), config.data)
    batch = collator(load_split(config.data, "train")[:6])

    # A new reader adds nothing: the receiver gives the same logits with and without the packet.
    plain = response_logits(model.receiver, batch["prompt"])
    packet = model.encode_context(batch["context"])
    assert packet.shape == (6, *model.bridge.packet_shape)
    with model.reading(packet, batch["prompt"]):
        assert torch.allclose(response_logits(model.receiver, batch["prompt"]), plain, atol=1e-6)

    for reader in model.bridge.readers:
        torch.nn.init.normal_(reader.output_projection.weight, std=0.1)
    with model.reading(packet, batch["prompt"]) as reader:
        logits = response_logits(model.receiver, batch["prompt"])
        score = score_responses(logits, batch)
        score.nll.sum().backward()
    assert not torch.allclose(logits, plain)
    assert all(parameter.grad.abs().sum() > 0 for parameter in model.bridge.parameters())
    assert all(parameter.grad is None for parameter in model.receiver.model.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.model.parameters())
    assert all(0 < float(gate.min()) and float(gate.max()) < 1 for gate in reader.gates.values())
    assert score.predictions.shape == (6,) and bool((score.predictions < batch["option_mask"].sum(1)).all())

    # The gates use prompt states only: another gold answer leaves them unchanged.
    other = {**batch, "prompt": {**batch["prompt"], "input_ids": batch["prompt"]["input_ids"].clone()}}
    rows = torch.arange(6)
    other["prompt"]["input_ids"][rows, batch["prompt"]["prompt_end_positions"] + 1] = 5
    with torch.no_grad(), model.reading(packet, other["prompt"]) as changed:
        response_logits(model.receiver, other["prompt"])
    assert all(torch.allclose(changed.gates[pair], reader.gates[pair], atol=1e-6) for pair in reader.gates)

    # Gradient checkpointing replays the readers and gives the same gradients.
    reference = [parameter.grad.clone() for parameter in model.bridge.parameters()]
    model.zero_grad()
    model.enable_gradient_checkpointing()
    model.train()
    with model.reading(model.encode_context(batch["context"]), batch["prompt"]):
        score_responses(response_logits(model.receiver, batch["prompt"]), batch).nll.sum().backward()
    for parameter, expected in zip(model.bridge.parameters(), reference):
        assert torch.allclose(parameter.grad, expected, atol=1e-5)

    save_bridge(tmp_path / "checkpoint", model.bridge, config, step=3)
    restored = build_tiny_model(config)
    load_bridge(tmp_path / "checkpoint", restored.bridge)
    for name, tensor in model.bridge.state_dict().items():
        assert torch.equal(restored.bridge.state_dict()[name], tensor)


def test_generation_computes_gates_once_per_prompt(tmp_path):
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    for reader in model.bridge.readers:
        torch.nn.init.normal_(reader.output_projection.weight, std=0.1)
    tokenizer = CharTokenizer()
    context = torch.tensor([tokenizer("alpha beta gamma")["input_ids"]])
    prompt_ids = torch.tensor([tokenizer("which word?")["input_ids"]] * 2)
    prompt = {"input_ids": prompt_ids, "attention_mask": torch.ones_like(prompt_ids)}
    with torch.no_grad():
        packet = model.encode_context({"input_ids": context, "attention_mask": torch.ones_like(context)})
    calls = []
    gate = model.bridge.gate
    model.bridge.gate = lambda pair, *args: calls.append(pair) or gate(pair, *args)
    output = model.generate(prompt, packet.repeat(2, 1, 1, 1), max_new_tokens=3)
    assert output.shape == (2, prompt_ids.shape[1] + 3)
    assert calls == list(range(len(model.bridge.layer_pairs)))
    assert torch.equal(output[0], output[1])
    assert not any(layer._forward_hooks for layer in model.receiver.layers)


def test_two_stage_training(tmp_path, tiny):
    # Stage 1: context reconstruction trains compressors, readers and gates.
    stage1 = make_config(tmp_path, "text", num_epochs=2, warmup_steps=1, **{"scheduler": "constant"})
    stage1.data.sort_within_update = True
    final = train_module.train(stage1)
    logs = [json.loads(line) for line in (tmp_path / "run-text" / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in logs if row["type"] == "train"] == list(range(1, 7))
    assert [row["step"] for row in logs if row["type"] == "eval"] == [0, 2, 3, 4, 6]
    initial = build_tiny_model(stage1)
    trained = build_tiny_model(stage1)
    load_bridge(final, trained.bridge)
    assert (tmp_path / "run-text" / "checkpoints" / "step-2" / "bridge.safetensors").exists()

    # Store one packet per distinct context with the Stage 1 compressors.
    stage2 = make_config(tmp_path, "mcq", num_epochs=1, warmup_steps=1, scheduler="cosine")
    writer = PacketWriter(tmp_path / "packets", rank=0, dtype="bfloat16")
    contexts = {example.context for split in ("train", "fast_dev") for example in load_split(stage2.data, split)}
    collator = Collator(CharTokenizer(), CharTokenizer(), stage2.data)
    for context in sorted(contexts):
        ids = torch.tensor([CharTokenizer()(context)["input_ids"]])
        with torch.no_grad():
            packet = trained.encode_context({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        writer.add([context_key(context)], packet)
    writer.close()
    (tmp_path / "packets" / "meta.json").write_text(json.dumps({"dtype": "bfloat16"}))
    cache = PacketCache(tmp_path / "packets")
    examples = load_split(stage2.data, "train")[:3]
    with torch.no_grad():
        online = trained.encode_context(collator(examples)["context"])
    assert torch.allclose(cache(examples), round_to_stored(online, "bfloat16"), atol=1e-6)

    # Stage 2: stored packets, compressors fixed, readers and gates trained.
    stage2.data.packet_cache = str(tmp_path / "packets")
    stage2.training.init_checkpoint = str(final)
    stage2.training.trainable = "readers_and_gates"
    task = build_tiny_model(stage2)
    load_bridge(train_module.train(stage2), task.bridge)
    for name, tensor in task.bridge.state_dict().items():
        unchanged = torch.equal(tensor, trained.bridge.state_dict()[name])
        assert unchanged == name.startswith("compressors."), name
    assert not torch.equal(
        trained.bridge.compressors[0].latent_queries, initial.bridge.compressors[0].latent_queries
    )
    logs = [json.loads(line) for line in (tmp_path / "run-mcq" / "metrics.jsonl").read_text().splitlines()]
    last = [row for row in logs if row["type"] == "eval"][-1]
    assert {"loss", "accuracy", "macro_accuracy", "accuracy/first", "accuracy/second"} <= set(last)
    assert [row for row in logs if row["type"] == "train"][-1]["lr"] == 0.0
