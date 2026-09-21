"""Offline by default. Live tests need `--live` and a non-zero spend cap in
`configs/live-tests.yaml`; without both they are skipped, never silently run."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
LIVE_CONFIG = ROOT / "configs" / "live-tests.yaml"


def _load_dotenv() -> None:
    """Live tests read keys from `.env`; existing environment values win."""
    import os

    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", default=False, help="run paid live tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: needs an API key and a spend cap")
    if config.getoption("--live"):
        _load_dotenv()


def live_spend_cap() -> float:
    if not LIVE_CONFIG.exists():
        return 0.0
    data = yaml.safe_load(LIVE_CONFIG.read_text()) or {}
    return float(data.get("max_spend_usd", 0.0))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_live = config.getoption("--live")
    cap = live_spend_cap()
    for item in items:
        if "live" not in item.keywords:
            continue
        if not run_live:
            item.add_marker(pytest.mark.skip(reason="pass --live to run"))
        elif cap <= 0.0:
            item.add_marker(
                pytest.mark.skip(reason=f"max_spend_usd is {cap} in {LIVE_CONFIG.name}")
            )


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
