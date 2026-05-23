from .base import Counter
from .._registry import Registry

REGISTRY: Registry[Counter] = Registry("counter")
register = REGISTRY.register
build = REGISTRY.build
