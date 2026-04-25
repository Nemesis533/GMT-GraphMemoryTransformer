"""Smoke tests for the GMT package."""


def test_package_imports() -> None:
    import gmt

    assert gmt.__all__ == []
