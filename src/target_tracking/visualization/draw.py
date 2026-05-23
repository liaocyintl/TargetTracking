from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Detection

_COLOR_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def _color_for_id(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (200, 200, 200)
    return _COLOR_PALETTE[track_id % len(_COLOR_PALETTE)]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = _color_for_id(det.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = det.class_name
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        label = f"{label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def draw_line(
    frame: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
) -> np.ndarray:
    out = frame
    p1 = (int(start[0]), int(start[1]))
    p2 = (int(end[0]), int(end[1]))
    cv2.line(out, p1, p2, (0, 255, 255), 2)
    if label:
        cv2.putText(
            out, label, (p1[0] + 5, p1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
    return out


def draw_counts(frame: np.ndarray, counts: dict) -> np.ndarray:
    out = frame
    y = 25
    for counter_name, snap in counts.items():
        header = f"[{counter_name}] in={snap['total_in']} out={snap['total_out']}"
        cv2.putText(out, header, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 22
        for cls_name, c in snap.get("per_class", {}).items():
            row = f"  {cls_name}: in={c['in']} out={c['out']}"
            cv2.putText(out, row, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y += 18
    return out
