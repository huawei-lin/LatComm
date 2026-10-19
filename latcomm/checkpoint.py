"""Bridge checkpoints. The two LLMs are frozen and are never written."""
from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from latcomm.bridge import LatentBridge
from latcomm.config import LatCommConfig

BRIDGE_FILE = "bridge.safetensors"


def save_bridge(directory: str | Path, bridge: LatentBridge, config: LatCommConfig, step: int) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in bridge.state_dict().items()}
    save_file(state, str(directory / BRIDGE_FILE))
    info = {"step": step, "layer_pairs": bridge.layer_pairs, "config": config.model_dump(mode="json")}
    (directory / "checkpoint.json").write_text(json.dumps(info, indent=2) + "\n")
    return directory


def load_bridge(directory: str | Path, bridge: LatentBridge) -> None:
    bridge.load_state_dict(load_file(str(Path(directory) / BRIDGE_FILE)))
