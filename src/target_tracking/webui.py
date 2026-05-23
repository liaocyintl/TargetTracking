from __future__ import annotations

import threading
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from .config import (
    AppConfig, CounterConfig, SinksConfig, SourceConfig, StatsSinkConfig,
    TrackerConfig, VideoSinkConfig, DisplaySinkConfig, build_pipeline,
)
from .core.pipeline import TrackingPipeline


_MODEL_CHOICES = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
_TRACKER_CHOICES = ["bytetrack.yaml", "botsort.yaml"]


class _Runner:
    def __init__(self) -> None:
        self.pipeline: Optional[TrackingPipeline] = None
        self.thread: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_counts: dict = {}
        self.error: Optional[str] = None

    def start(self, cfg: AppConfig) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.pipeline = build_pipeline(cfg)
        self.error = None

        # Wire any DisplaySink's on_frame callback to capture the latest annotated frame
        for s in self.pipeline._sinks:
            if hasattr(s, "_on_frame"):
                def cb(frame, counts, self=self):
                    self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                s._on_frame = cb

        def _run():
            try:
                def on_frame(_result, counts):
                    self.latest_counts = counts

                self.pipeline.run(on_frame=on_frame)
            except Exception as e:  # noqa: BLE001
                self.error = str(e)

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()


def _build_config(
    video_file: str | None,
    use_camera: bool,
    camera_id: int,
    model_name: str,
    tracker_name: str,
    conf: float,
    classes_text: str,
    line_x1: int,
    line_y1: int,
    line_x2: int,
    line_y2: int,
    save_video: bool,
    save_stats: bool,
) -> AppConfig:
    classes = [int(c.strip()) for c in classes_text.split(",") if c.strip()]
    source = (
        SourceConfig(type="camera", device_id=camera_id)
        if use_camera else
        SourceConfig(type="file", path=video_file or "data/videos/demo.mp4")
    )
    return AppConfig(
        source=source,
        tracker=TrackerConfig(
            name="yolo",
            params={
                "model_name": model_name,
                "tracker": tracker_name,
                "conf": conf,
                "classes": classes,
                "device": "auto",
            },
        ),
        counters=[
            CounterConfig(
                name="gate",
                type="line",
                params={
                    "start": [line_x1, line_y1],
                    "end": [line_x2, line_y2],
                    "classes_of_interest": None,
                },
            )
        ],
        sinks=SinksConfig(
            display=DisplaySinkConfig(enabled=True, window_name="webui"),
            video=VideoSinkConfig(enabled=save_video, path="outputs/webui.mp4"),
            stats=StatsSinkConfig(enabled=save_stats, format="csv",
                                  path="outputs/webui.csv"),
        ),
    )


def _format_counts(counts: dict) -> str:
    if not counts:
        return "(no counts yet)"
    lines = []
    for name, snap in counts.items():
        lines.append(f"### {name}  —  in: {snap['total_in']}  out: {snap['total_out']}")
        for cls, c in snap.get("per_class", {}).items():
            lines.append(f"- **{cls}**: in {c['in']} / out {c['out']}")
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    runner = _Runner()

    with gr.Blocks(title="Target Tracking") as demo:
        gr.Markdown("# Target Tracking")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("File"):
                    video_file = gr.File(label="Video file", file_types=["video"])
                with gr.Tab("Camera"):
                    use_camera = gr.Checkbox(value=False, label="Use camera")
                    camera_id = gr.Number(value=0, precision=0, label="Camera id")

                model_name = gr.Dropdown(choices=_MODEL_CHOICES, value="yolo26n.pt",
                                         label="Model")
                tracker_name = gr.Dropdown(choices=_TRACKER_CHOICES,
                                           value="bytetrack.yaml", label="Tracker")
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
                classes_text = gr.Textbox(value="0,2",
                                          label="COCO class ids (comma-separated)")

                gr.Markdown("**Counting line** (pixel coords)")
                with gr.Row():
                    line_x1 = gr.Number(value=0, label="x1")
                    line_y1 = gr.Number(value=540, label="y1")
                with gr.Row():
                    line_x2 = gr.Number(value=1920, label="x2")
                    line_y2 = gr.Number(value=540, label="y2")

                save_video = gr.Checkbox(value=True, label="Save annotated video")
                save_stats = gr.Checkbox(value=True, label="Save stats CSV")

                start_btn = gr.Button("Start", variant="primary")
                stop_btn = gr.Button("Stop")
                status = gr.Markdown("Idle.")

            with gr.Column(scale=2):
                live_img = gr.Image(label="Live", streaming=True, type="numpy")
                counts_md = gr.Markdown("(no counts yet)")

        def on_start(file_obj, use_cam, cam_id, model, tracker, c, cls,
                     x1, y1, x2, y2, sv, ss):
            path = file_obj.name if (file_obj is not None and not use_cam) else None
            cfg = _build_config(
                video_file=path, use_camera=use_cam, camera_id=int(cam_id),
                model_name=model, tracker_name=tracker, conf=float(c),
                classes_text=cls,
                line_x1=int(x1), line_y1=int(y1),
                line_x2=int(x2), line_y2=int(y2),
                save_video=sv, save_stats=ss,
            )
            runner.start(cfg)
            return "Running…"

        def on_stop():
            runner.stop()
            return "Stopped."

        def poll():
            if runner.error:
                return None, f"**Error:** {runner.error}"
            return runner.latest_frame, _format_counts(runner.latest_counts)

        start_btn.click(
            on_start,
            inputs=[video_file, use_camera, camera_id, model_name, tracker_name,
                    conf, classes_text, line_x1, line_y1, line_x2, line_y2,
                    save_video, save_stats],
            outputs=[status],
        )
        stop_btn.click(on_stop, outputs=[status])

        timer = gr.Timer(0.1)
        timer.tick(poll, outputs=[live_img, counts_md])

    return demo


def launch() -> None:
    build_ui().launch()


if __name__ == "__main__":
    launch()
