from .base import TrackerModel
from .._registry import Registry

REGISTRY: Registry[TrackerModel] = Registry("tracker")
register = REGISTRY.register
build = REGISTRY.build
