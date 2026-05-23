from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AppConfig, build_pipeline, load_config


def _parse_source(value: str):
    """Parse --source 'camera:0' or 'path/to/file.mp4'."""
    if value.startswith("camera:"):
        return {"type": "camera", "device_id": int(value.split(":", 1)[1])}
    return {"type": "file", "path": value}


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.source:
        cfg.source = type(cfg.source).model_validate(_parse_source(args.source))
    if args.tracker:
        cfg.tracker.name = args.tracker
    if args.model:
        cfg.tracker.params["model_name"] = args.model
    if args.device:
        cfg.tracker.params["device"] = args.device
    if args.no_display:
        cfg.sinks.display.enabled = False
    if args.no_video:
        cfg.sinks.video.enabled = False
    if args.no_stats:
        cfg.sinks.stats.enabled = False
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="target-tracking")
    sub = parser.add_subparsers(dest="command")

    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", type=str, help="path/to/video.mp4 or camera:0")
    parser.add_argument("--tracker", type=str, help="tracker name (e.g. yolo)")
    parser.add_argument("--model", type=str, help="model weights (e.g. yolo26n.pt)")
    parser.add_argument("--device", type=str, help="auto|cpu|cuda|mps")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-stats", action="store_true")

    sub.add_parser("ui", help="Launch Gradio Web UI")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ui":
        from .webui import launch
        launch()
        return 0
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    pipeline = build_pipeline(cfg)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
