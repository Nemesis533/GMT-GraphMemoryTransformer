"""Smoke tests for the public GMT package skeleton."""


def test_package_imports() -> None:
    import gmt

    assert gmt.__all__ == []
