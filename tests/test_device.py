from unittest.mock import patch

from target_tracking.core.device import resolve_device


def test_explicit_device_returned_as_is():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_auto_picks_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == "cuda"


def test_auto_picks_mps_when_no_cuda():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        assert resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert resolve_device("auto") == "cpu"
