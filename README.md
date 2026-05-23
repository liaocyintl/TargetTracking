# Target Tracking

Pluggable visual object tracking and counting (cars, people, etc.) for video files and live cameras.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
# CLI with config file
python -m target_tracking.cli --config configs/default.yaml

# CLI shortcuts
python -m target_tracking.cli --source video.mp4
python -m target_tracking.cli --source camera:0

# Web UI
python -m target_tracking.webui
```

## Tests

```bash
pytest
```
