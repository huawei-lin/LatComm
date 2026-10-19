# LatComm

[![Paper](https://img.shields.io/badge/Paper-arXiv%202610.26840-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.26840)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/huawei-lin/LatComm)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/huaweilin/LatComm)

Code for the paper [*LatComm: Reusable Context Compression for Latent Communication
between Large Language Models*](https://arxiv.org/abs/2610.26840).

A frozen **context encoder** LLM reads a context once. Learned **compressors** map its
token states into a fixed number of low-dimensional slots, the **packet**. A frozen
**receiver** LLM reads the packet through cross-attention **readers** while it answers a
question, and **gates** computed from its prompt control each slot's contribution. The
packet does not depend on the question, so it is encoded once, stored, and reused.

## Repository layout

| Path | Content |
| --- | --- |
| `latcomm/bridge.py` | Layer pairing, context compressor, latent reader, prompt pooling, gates |
| `latcomm/model.py` | Frozen LLM loading, packet encoding, packet reading inside the receiver, generation |
| `latcomm/data.py` | Data format, prompts, batching, samplers |
| `latcomm/packets.py` | Stored packets (32-bit float or bfloat16) |
| `latcomm/train.py` | Trainer for both stages, single GPU or multi GPU |
| `latcomm/evaluate.py` | Response scoring and metrics |
| `scripts/download_data.py` | Download the data from the Hugging Face Hub |
| `scripts/encode_packets.py` | Encode and store the packets of a dataset |
| `scripts/evaluate.py` | Evaluate a trained interface on a test file |
| `configs/` | Stage 1 and Stage 2 configurations of the six model pairs |
| `tests/` | CPU tests on a small random transformer |

## Installation

The experiments use Python 3.12, PyTorch 2.11 or later, and Transformers 5.15.1.

```bash
pip install -e .
```

The models are downloaded from the Hugging Face Hub on first use. Gemma and Llama
checkpoints require accepting their licenses on the Hub.

To check the installation, run the tests. They use a small random transformer, need no
download, and finish in a few seconds on a CPU.

```bash
pip install pytest
pytest
```

## Data

The data is on the Hugging Face Hub:
[huaweilin/LatComm](https://huggingface.co/datasets/huaweilin/LatComm). It has two
configurations:

| Configuration | Content | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| `stage-1` | Reconstruction corpus (WikiText-103 and FineWeb-Edu passages) | 2,000,000 | 15,043 | 14,983 |
| `stage-2` | 12-dataset multiple-choice suite | 282,441 | 2,036 | 55,890 |

Download it into `data/stage1` and `data/stage2`, where the configurations read it:

```bash
python scripts/download_data.py
```

The Stage 1 corpus is 2.8 GB compressed. `--stages stage2` downloads only the Stage 2 data.

Each line of a data file is one record:

```json
{"sample_id": "...", "question": "...", "choices": ["...", "..."],
 "evidence": [{"source_record_id": "...", "text": "..."}],
 "response": "B", "provenance": {"dataset": "...", "...": "..."}}
```

`evidence` holds the context, which the context encoder compresses. In Stage 2, `choices`
are the options and `response` is the letter of the gold option. In Stage 1 there are no
`choices`: `question` is the instruction and `response` is the passage, the same text as
the context.

## Usage

The commands below use the main pair, a Qwen3.5-4B encoder with a Qwen3.5-0.8B receiver,
on 8 GPUs. The other pairs only change the configuration file. For a single GPU, replace
`torchrun --nproc_per_node 8` by `python`.

The global batch size is `per_device_batch_size x number of GPUs x
gradient_accumulation_steps`. It is 384 in all configurations. With fewer GPUs, raise
`gradient_accumulation_steps` to keep it.

### 1. Stage 1: context reconstruction

The receiver reproduces the context from its packet. This trains the compressors,
readers, and gates (one epoch, 5,209 updates).

```bash
torchrun --nproc_per_node 8 -m latcomm.train --config configs/stage1/qwen4b_to_qwen0.8b.yaml
```

Checkpoints are written to `outputs/stage1/qwen4b_to_qwen0.8b/checkpoints/`, and metrics
to `metrics.jsonl` in the output directory.

### 2. Store the packets

The compressors are now fixed. Each distinct training context is encoded once, without
the question.

```bash
torchrun --nproc_per_node 8 scripts/encode_packets.py \
    --config configs/stage2/qwen4b_to_qwen0.8b.yaml \
    --checkpoint outputs/stage1/qwen4b_to_qwen0.8b/checkpoints/final \
    --output packets/qwen4b_to_qwen0.8b
```

### 3. Stage 2: task supervision

Only the readers and gates are trained, on the stored packets (three epochs, 2,208
updates). The context encoder LLM is not loaded.

```bash
torchrun --nproc_per_node 8 -m latcomm.train --config configs/stage2/qwen4b_to_qwen0.8b.yaml
```

### 4. Evaluation

The encoder compresses each test context, and the receiver answers from the packet. The
prediction is the valid option letter with the highest logit at the first response
position.

```bash
torchrun --nproc_per_node 8 scripts/evaluate.py \
    --config configs/stage2/qwen4b_to_qwen0.8b.yaml \
    --checkpoint outputs/stage2/qwen4b_to_qwen0.8b/checkpoints/final \
    --data data/stage2/test.jsonl --output results/qwen4b_to_qwen0.8b/task
```

The output directory gets `predictions.jsonl`, with one line per question, and
`metrics.json`, with the accuracy on each dataset, `accuracy` over all questions, and
`macro_accuracy`, the unweighted mean over datasets. Packets are rounded to bfloat16 before the receiver reads them, as
stored packets are; `--packet-dtype float32` keeps full precision.

### Using a trained interface in Python

```python
import torch
from latcomm.checkpoint import load_bridge
from latcomm.config import load_config
from latcomm.model import build_model, load_tokenizer

config = load_config("configs/stage2/qwen4b_to_qwen0.8b.yaml")
model = build_model(config, "cuda").eval()
load_bridge("outputs/stage2/qwen4b_to_qwen0.8b/checkpoints/final", model.bridge)
encoder_tokenizer = load_tokenizer(config.encoder)
receiver_tokenizer = load_tokenizer(config.receiver)

# Encode the context once. The packet has shape [1, P, S, d] and can be stored.
with torch.no_grad():
    packet = model.encode_context(encoder_tokenizer("<the context>", return_tensors="pt"))

# Reuse the packet for any number of questions.
for question in ("<first question>", "<second question>"):
    prompt = receiver_tokenizer(question, return_tensors="pt")
    output = model.generate(prompt, packet, max_new_tokens=32)
    print(receiver_tokenizer.decode(output[0, prompt["input_ids"].shape[1]:]))
```

## Configuration

Each YAML file has the sections `encoder`, `receiver`, `bridge`, `data`, and `training`;
`latcomm/config.py` lists every field with its default. The packet size is set in `bridge`:

| Field | Meaning | Default |
| --- | --- | --- |
| `num_layer_pairs` | number of layer pairs *P*: a count, or a fraction of the shallower model's depth | `1.0` |
| `num_slots` | slots per layer pair *S* | `32` |
| `context_dim` | slot width *d* | `256` |

A packet stores *P x S x d* values, which is 384 KiB in bfloat16 for 24 layer pairs. To
change the packet budget, edit these fields (for example `num_slots: 8`,
`context_dim: 128`, or `num_layer_pairs: 0.5`) and use the same values in Stage 1 and
Stage 2.

`configs/` holds one Stage 1 and one Stage 2 file for each model pair:

| File name | Context encoder | Receiver |
| --- | --- | --- |
| `qwen0.8b_to_qwen0.8b` | Qwen3.5-0.8B | Qwen3.5-0.8B |
| `qwen4b_to_qwen0.8b` | Qwen3.5-4B | Qwen3.5-0.8B |
| `qwen9b_to_qwen0.8b` | Qwen3.5-9B | Qwen3.5-0.8B |
| `qwen0.8b_to_gemma1b` | Qwen3.5-0.8B | Gemma-3-1B-pt |
| `qwen9b_to_gemma1b` | Qwen3.5-9B | Gemma-3-1B-pt |
| `granite8b_to_llama1b` | Granite-4.1-8B | Llama-3.2-1B |

Set `wandb.enabled: true` to also log to Weights & Biases (`pip install wandb`).

## Citation

```bibtex
@misc{lin2026latcomm,
  title         = {LatComm: Reusable Context Compression for Latent Communication between Large Language Models},
  author        = {Huawei Lin and Jinting Liu and Tan Yu and Weijie Zhao},
  year          = {2026},
  eprint        = {2610.26840},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.26840}
}
```
