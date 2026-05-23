import pytest

from target_tracking._registry import Registry


def test_register_and_build():
    reg: Registry[object] = Registry("widget")

    @reg.register("foo")
    class Foo:
        def __init__(self, x: int = 0):
            self.x = x

    obj = reg.build("foo", x=5)
    assert isinstance(obj, Foo)
    assert obj.x == 5


def test_build_unknown_raises():
    reg: Registry[object] = Registry("widget")
    with pytest.raises(ValueError, match="Unknown widget: 'bar'"):
        reg.build("bar")


def test_names_listed_in_error():
    reg: Registry[object] = Registry("widget")

    @reg.register("a")
    class A: ...

    @reg.register("b")
    class B: ...

    with pytest.raises(ValueError, match="Available: \\['a', 'b'\\]"):
        reg.build("zzz")
