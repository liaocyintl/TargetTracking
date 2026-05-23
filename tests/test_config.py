from pathlib import Path

import pytest

from target_tracking.config import AppConfig, load_config, build_pipeline, _resolve_point


def test_resolve_point_floats_in_unit_interval_are_relative():
    assert _resolve_point([0.5, 0.0], (1920, 1080)) == [960.0, 0.0]
    assert _resolve_point([0.5, 1.0], (1920, 1080)) == [960.0, 1080.0]


def test_resolve_point_ints_are_absolute_pixels():
    assert _resolve_point([960, 540], (1920, 1080)) == [960.0, 540.0]


def test_resolve_point_floats_above_one_treated_as_pixels():
    assert _resolve_point([1.5, 2.0], (1920, 1080)) == [1.5, 2.0]


def test_resolve_point_mixed_axes():
    # x relative, y absolute
    assert _resolve_point([0.25, 540], (1920, 1080)) == [480.0, 540.0]


YAML = """
source:
  type: file
  path: tests/fixtures/short.mp4

tracker:
  name: mock_tracker_for_test
  params: {}

counters:
  - name: gate
    type: line
    params:
      start: [0, 50]
      end: [100, 50]
      classes_of_interest: [person]

sinks:
  stats:
    enabled: true
    format: csv
    path: outputs/x.csv
  video:
    enabled: false
  display:
    enabled: false
"""


def test_load_config_parses_yaml(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(YAML)
    cfg = load_config(cfg_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.source.type == "file"
    assert cfg.tracker.name == "mock_tracker_for_test"
    assert cfg.counters[0].type == "line"
    assert cfg.sinks.stats.enabled is True


def test_load_config_rejects_unknown_source_type(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("source:\n  type: nonsense\n")
    cfg = load_config(cfg_path)
    # load_config accepts any string for source.type;
    # the ValueError is raised by build_pipeline when the registry lookup fails.
    with pytest.raises(Exception):
        build_pipeline(cfg)


def test_build_pipeline_resolves_registry_names(tmp_path: Path):
    # Register a fake tracker so we don't load YOLO weights in this test
    from target_tracking.trackers import register as tracker_register
    from target_tracking.trackers.base import TrackerModel

    @tracker_register("mock_tracker_for_test")
    class _Mock(TrackerModel):
        def update(self, frame):
            return []

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(YAML)
    cfg = load_config(cfg_path)

    # Switch source to a mock so we don't need a real video file
    cfg.source.type = "mock_source_for_test"

    from target_tracking.sources.base import VideoSource
    import target_tracking.config as config_mod

    class _SrcMock(VideoSource):
        def read(self):
            return None
        @property
        def fps(self): return 30.0
        @property
        def frame_size(self): return (10, 10)

    config_mod.SOURCE_REGISTRY.register("mock_source_for_test")(_SrcMock)

    pipe = build_pipeline(cfg)
    assert pipe is not None
