"""Tests for wallow._paths (sanitise + substitute_layout)."""

from __future__ import annotations

import pytest

from wallow import SchemaValidationError
from wallow._paths import sanitise_for_path, substitute_layout


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("simple", "simple"),
        ("Hello World", "Hello_World"),
        ("café", "cafe"),
        ("résumé", "resume"),
        ("with/slashes", "with_slashes"),
        ("multi   spaces", "multi_spaces"),
        ("uPpEr_LoWeR-1.2", "uPpEr_LoWeR-1.2"),  # safe chars preserved as-is
        ("trailing.dots...", "trailing.dots"),
        ("__leading_underscores", "leading_underscores"),
        ("a_b!c@d#e%f", "a_b_c_d_e_f"),
        (123, "123"),  # non-strings are stringified
        (1.5, "1.5"),
        (True, "True"),
    ],
)
def test_sanitise_for_path(raw, expected):
    assert sanitise_for_path(raw) == expected


def test_sanitise_for_path_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        sanitise_for_path("...")
    with pytest.raises(ValueError, match="empty"):
        sanitise_for_path("!!!")


def test_substitute_layout_simple():
    out = substitute_layout("{kind}/{uuid}", {"kind": "speed", "uuid": "abc123"})
    assert out == "speed/abc123"


def test_substitute_layout_sanitises():
    out = substitute_layout(
        "{label}/{uuid}", {"label": "Hello World", "uuid": "deadbeef"}
    )
    assert out == "Hello_World/deadbeef"


def test_substitute_layout_unknown_placeholder_raises():
    with pytest.raises(SchemaValidationError, match="no value"):
        substitute_layout("{missing}/{uuid}", {"uuid": "abc"})


def test_substitute_layout_empty_substitution_raises():
    with pytest.raises(SchemaValidationError, match="empty"):
        substitute_layout("{label}", {"label": "..."})


def test_substitute_layout_no_placeholders_passes_through():
    assert substitute_layout("static/path", {}) == "static/path"
