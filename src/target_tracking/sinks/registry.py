from .base import Sink
from .._registry import Registry

REGISTRY: Registry[Sink] = Registry("sink")
register = REGISTRY.register
build = REGISTRY.build
