"""Offline by default. Tests marked `live` need `--live`, a key in `.env` or the
environment, and spend a few cents at most; without the flag they are skipped."""

from __future__ import annotations

from pathlib import Path

import pytest

from navjev.env import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", default=False, help="run paid live tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: calls the real APIs")
    if config.getoption("--live"):
        load_dotenv(ROOT / ".env")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    for item in items:
        if "live" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="pass --live to run"))


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
