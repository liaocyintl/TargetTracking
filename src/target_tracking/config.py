from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from ._registry import Registry
from .core.pipeline import TrackingPipeline
from .counters import build as build_counter
from .sinks import build as build_sink
from .sources.base import VideoSource
from .sources.camera import CameraSource
from .sources.file import FileSource
from .trackers import build as build_tracker


SOURCE_REGISTRY: Registry[VideoSource] = Registry("source")
SOURCE_REGISTRY.register("file")(FileSource)
SOURCE_REGISTRY.register("camera")(CameraSource)


class SourceConfig(BaseModel):
    type: str
    path: Optional[str] = None
    device_id: int = 0


class TrackerConfig(BaseModel):
    name: str = "yolo"
    params: dict[str, Any] = Field(default_factory=dict)


class CounterConfig(BaseModel):
    name: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class DisplaySinkConfig(BaseModel):
    enabled: bool = True
    window_name: str = "TargetTracking"


class VideoSinkConfig(BaseModel):
    enabled: bool = False
    path: str = "outputs/annotated.mp4"


class StatsSinkConfig(BaseModel):
    enabled: bool = False
    format: Literal["csv", "json"] = "csv"
    path: str = "outputs/counts.csv"


class SinksConfig(BaseModel):
    display: DisplaySinkConfig = Field(default_factory=DisplaySinkConfig)
    video: VideoSinkConfig = Field(default_factory=VideoSinkConfig)
    stats: StatsSinkConfig = Field(default_factory=StatsSinkConfig)


class AppConfig(BaseModel):
    source: SourceConfig
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    counters: list[CounterConfig] = Field(default_factory=list)
    sinks: SinksConfig = Field(default_factory=SinksConfig)


def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text())
    return AppConfig.model_validate(data)


def _build_source(cfg: SourceConfig) -> VideoSource:
    if cfg.type == "file":
        if not cfg.path:
            raise ValueError("source.path required for file source")
        return SOURCE_REGISTRY.build("file", path=cfg.path)
    if cfg.type == "camera":
        return SOURCE_REGISTRY.build("camera", device_id=cfg.device_id)
    return SOURCE_REGISTRY.build(cfg.type)


def _resolve_point(point, frame_size: tuple[int, int]) -> list[float]:
    """Resolve a [x, y] point. Floats in [0, 1] are fractions of frame size;
    ints (or floats > 1) are absolute pixels."""
    w, h = frame_size
    x, y = point
    rx = x * w if isinstance(x, float) and 0.0 <= x <= 1.0 else x
    ry = y * h if isinstance(y, float) and 0.0 <= y <= 1.0 else y
    return [float(rx), float(ry)]


def build_pipeline(cfg: AppConfig) -> TrackingPipeline:
    source = _build_source(cfg.source)
    tracker = build_tracker(cfg.tracker.name, **cfg.tracker.params)
    frame_size = source.frame_size

    counters = []
    for c in cfg.counters:
        params = dict(c.params)
        if c.type == "line":
            params["start"] = _resolve_point(params["start"], frame_size)
            params["end"] = _resolve_point(params["end"], frame_size)
        counters.append(build_counter(c.type, name=c.name, **params))

    overlay_lines = [
        {
            "name": c.name,
            "start": _resolve_point(c.params["start"], frame_size),
            "end": _resolve_point(c.params["end"], frame_size),
        }
        for c in cfg.counters
        if c.type == "line"
    ]

    sinks = []
    if cfg.sinks.display.enabled:
        sinks.append(build_sink(
            "display",
            window_name=cfg.sinks.display.window_name,
            overlay_lines=overlay_lines,
        ))
    if cfg.sinks.video.enabled:
        sinks.append(build_sink(
            "video",
            path=cfg.sinks.video.path,
            fps=source.fps,
            overlay_lines=overlay_lines,
        ))
    if cfg.sinks.stats.enabled:
        sinks.append(
            build_sink("stats", path=cfg.sinks.stats.path, format=cfg.sinks.stats.format)
        )
    return TrackingPipeline(source=source, tracker=tracker,
                            counters=counters, sinks=sinks)
