"""Pre-download YOLO weights to local cache. Usage: python scripts/download_models.py"""
from __future__ import annotations

import sys

_DEFAULT_MODELS = ["yolo26n.pt", "yolo26s.pt"]


def main(models: list[str]) -> int:
    from ultralytics import YOLO
    for name in models:
        print(f"Downloading {name} …")
        YOLO(name)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or _DEFAULT_MODELS
    raise SystemExit(main(args))
