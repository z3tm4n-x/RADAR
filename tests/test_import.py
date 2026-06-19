from radar import __version__


def test_package_imports() -> None:
    assert isinstance(__version__, str)
    assert __version__