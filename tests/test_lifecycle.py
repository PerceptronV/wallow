"""Tests for wallow.contrib.lifecycle.run_lifecycle."""

from __future__ import annotations

import pytest

from wallow import F, Schema, Store
from wallow.contrib.lifecycle import AlreadyCompleted, run_lifecycle

from conftest import schema_from_toml


# A dedicated schema with the annotation fields the lifecycle helper writes to.
LIFECYCLE_TOML = """
[project]
name = "lifecycle_test"
artefacts_root = "tmp/data"

[identifying.task]
type = "string"

[identifying.seed]
type = "int"
default = 0

[annotating.status]
type = "string"
indexed = true

[annotating.started_at]
type = "datetime"

[annotating.completed_at]
type = "datetime"

[annotating.wallclock_seconds]
type = "float"

[annotating.error_excerpt]
type = "string"

[annotating.host]
type = "string"

[annotating.final_acc]
type = "float"
"""


@pytest.fixture
def lc_store() -> Store:
    schema = schema_from_toml(LIFECYCLE_TOML)
    return Store(":memory:", schema=schema, check_schema=False)


def test_happy_path_writes_completed(lc_store: Store):
    with run_lifecycle(lc_store, identifying={"task": "alpha"}) as h:
        assert h.uuid is not None
        assert h.run.status == "running"
        h.finalise(annotating={"final_acc": 0.95})

    row = lc_store.find_by_uuid(h.uuid)
    assert row.status == "completed"
    assert row.final_acc == 0.95
    assert row.completed_at is not None
    assert row.wallclock_seconds is not None and row.wallclock_seconds >= 0


def test_body_omits_finalise_still_marks_completed(lc_store: Store):
    with run_lifecycle(lc_store, identifying={"task": "beta"}) as h:
        pass  # caller forgot finalise — lifecycle should write it itself
    row = lc_store.find_by_uuid(h.uuid)
    assert row.status == "completed"


def test_body_exception_writes_failed(lc_store: Store):
    with pytest.raises(RuntimeError, match="boom"):
        with run_lifecycle(lc_store, identifying={"task": "gamma"}) as h:
            raise RuntimeError("boom")
    row = lc_store.find_by_uuid(h.uuid)
    assert row.status == "failed"
    assert "RuntimeError" in row.error_excerpt
    assert "boom" in row.error_excerpt


def test_already_completed_raises(lc_store: Store):
    with run_lifecycle(lc_store, identifying={"task": "delta"}) as h:
        h.finalise(annotating={"final_acc": 0.9})
    with pytest.raises(AlreadyCompleted) as ei:
        with run_lifecycle(lc_store, identifying={"task": "delta"}):
            pytest.fail("body should not run")
    assert ei.value.run.uuid == h.uuid


def test_force_re_runs_completed_row(lc_store: Store):
    with run_lifecycle(lc_store, identifying={"task": "epsilon"}) as h1:
        h1.finalise(annotating={"final_acc": 0.5})
    original_uuid = h1.uuid

    with run_lifecycle(
        lc_store, identifying={"task": "epsilon"}, force=True
    ) as h2:
        h2.finalise(annotating={"final_acc": 0.99})

    # Same identifying tuple → same row → same uuid; result overwritten.
    assert h2.uuid == original_uuid
    row = lc_store.find_by_uuid(original_uuid)
    assert row.final_acc == 0.99


def test_failed_row_can_be_reclaimed_without_force(lc_store: Store):
    with pytest.raises(RuntimeError):
        with run_lifecycle(lc_store, identifying={"task": "zeta"}) as h:
            raise RuntimeError("first attempt")
    failed_uuid = h.uuid

    # No force needed — only `completed` blocks reclaim.
    with run_lifecycle(lc_store, identifying={"task": "zeta"}) as h2:
        h2.finalise(annotating={"final_acc": 0.7})
    assert h2.uuid == failed_uuid
    row = lc_store.find_by_uuid(failed_uuid)
    assert row.status == "completed"


def test_start_annotating_propagates(lc_store: Store):
    with run_lifecycle(
        lc_store,
        identifying={"task": "eta"},
        start_annotating={"host": "test-host-1"},
    ) as h:
        assert h.run.host == "test-host-1"
        h.finalise()
    row = lc_store.find_by_uuid(h.uuid)
    assert row.host == "test-host-1"


def test_artefacts_dir_off_handle_run(lc_store: Store):
    with run_lifecycle(lc_store, identifying={"task": "theta"}) as h:
        d = lc_store.artefacts_dir(h.run)
        # Default layout {uuid}; uuid is on the run row.
        assert d.name == h.uuid
        h.finalise()
