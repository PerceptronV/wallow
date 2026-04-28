"""Shared pytest fixtures for wallow tests."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover
    import tomli as _toml

from wallow import Schema, Store, load_schema
from wallow.schema import _parse


FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXAMPLE_TOML = FIXTURES_DIR / "example_wallow.toml"


def schema_from_toml(text: str) -> Schema:
    """Build a Schema from an inline TOML string (no disk I/O)."""
    return _parse(_toml.loads(textwrap.dedent(text)))


@pytest.fixture
def schema_from_string():
    return schema_from_toml


@pytest.fixture
def example_schema() -> Schema:
    return load_schema(EXAMPLE_TOML)


@pytest.fixture
def memory_store(example_schema: Schema) -> Store:
    return Store(":memory:", schema=example_schema, check_schema=False)


@pytest.fixture
def file_store(tmp_path: Path, example_schema: Schema) -> Store:
    return Store(tmp_path / "runs.db", schema=example_schema, check_schema=False)


def make_identifying(**overrides: Any) -> dict[str, Any]:
    base = {
        "cell_k": 4,
        "cell_sigma": 0.1,
        "generation": 0,
        "candidate_id": 1,
        "seed": 0,
    }
    base.update(overrides)
    return base
