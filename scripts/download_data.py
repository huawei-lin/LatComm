"""Download the LatComm data from the Hugging Face Hub (huaweilin/LatComm).

    python scripts/download_data.py                  # both stages
    python scripts/download_data.py --stages stage2  # only the Stage 2 task suite

Writes train.jsonl, dev.jsonl, test.jsonl and fast_dev.jsonl to data/stage1 and
data/stage2, the directories the configurations read. dev.jsonl and fast_dev.jsonl
are the `validation` and `fast_validation` splits on the Hub.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "huaweilin/LatComm"
FOLDERS = {"stage1": "subsets/english_reconstruction_v1/original", "stage2": "original"}
SPLITS = {"train": "train", "dev": "validation", "test": "test", "fast_dev": "fast_validation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=list(FOLDERS), default=list(FOLDERS))
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()

    for stage in args.stages:
        (args.output / stage).mkdir(parents=True, exist_ok=True)
        for split, name in SPLITS.items():
            source = hf_hub_download(REPO, f"{FOLDERS[stage]}/{name}.jsonl.gz", repo_type="dataset")
            target = args.output / stage / f"{split}.jsonl"
            with gzip.open(source, "rb") as compressed, target.open("wb") as out:
                shutil.copyfileobj(compressed, out, 1 << 22)
            print("wrote", target, flush=True)


if __name__ == "__main__":
    main()
