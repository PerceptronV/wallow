"""Tests for the v0.2 ergonomics pass: defaults at register, RegisterResult,
eager F validation / schema.f, claim_if_stale + heartbeat, float normalisation.
"""

from __future__ import annotations

import datetime as _dt
import math
import textwrap
import time

import pytest
from sqlalchemy import select

from wallow import (
    F,
    Field,
    RegisterResult,
    SchemaValidationError,
    Store,
    find,
    heartbeat,
    load_schema,
    register,
)
from wallow.schema import _normalise_float, _parse

from conftest import make_identifying, schema_from_toml


# ---- Fix 1: default-fill at register / find --------------------------------


def test_register_fills_omitted_identifying_default(memory_store: Store):
    keys = make_identifying()
    keys.pop("seed")  # `seed` declares default = 0 in the fixture
    result = register(memory_store, identifying=keys, on_duplicate="raise")
    assert result.was_inserted is True
    assert result.run.seed == 0


def test_find_fills_omitted_identifying_default(memory_store: Store):
    register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    keys = make_identifying()
    keys.pop("seed")
    hit = find(memory_store, **keys)
    assert hit is not None
    assert hit.seed == 0


def test_register_dedupes_with_default_filled(memory_store: Store):
    # First call passes seed explicitly, second omits it; both should hit the same row.
    register(
        memory_store,
        identifying=make_identifying(seed=0),
        on_duplicate="raise",
    )
    keys = make_identifying()
    keys.pop("seed")
    result = register(memory_store, identifying=keys, on_duplicate="return_existing")
    assert result.was_inserted is False


def test_register_missing_field_without_default_still_errors(memory_store: Store):
    # `cell_k` has no default; omitting it must still raise.
    keys = make_identifying()
    keys.pop("cell_k")
    with pytest.raises(SchemaValidationError) as ei:
        register(memory_store, identifying=keys, on_duplicate="raise")
    assert "cell_k" in ei.value.missing_keys


# ---- Fix 2: RegisterResult flags --------------------------------------------


def test_register_result_is_returned_on_insert(memory_store: Store):
    result = register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    assert isinstance(result, RegisterResult)
    assert result.was_inserted is True
    assert result.was_updated is False
    assert result.was_skipped is False
    assert result.run is not None


def test_register_result_overwrite_on_duplicate(memory_store: Store):
    register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    result = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "done"},
        on_duplicate="overwrite",
    )
    assert result.was_inserted is False
    assert result.was_updated is True
    assert result.was_skipped is False


def test_register_result_skip_on_duplicate(memory_store: Store):
    register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    result = register(
        memory_store, identifying=make_identifying(), on_duplicate="skip"
    )
    assert result.run is None
    assert result.was_skipped is True


def test_register_result_return_existing_flags_all_false(memory_store: Store):
    register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    result = register(
        memory_store,
        identifying=make_identifying(),
        on_duplicate="return_existing",
    )
    assert result.was_inserted is False
    assert result.was_updated is False
    assert result.was_skipped is False
    assert result.run is not None


# ---- Fix 3: eager F validation + schema.f -----------------------------------


def test_F_with_schema_validates_eagerly(example_schema):
    with pytest.raises(SchemaValidationError) as ei:
        F("typo_name", schema=example_schema)
    assert "typo_name" in str(ei.value)
    assert ei.value.field == "typo_name"


def test_F_with_schema_known_name_succeeds(example_schema):
    f = F("cell_k", schema=example_schema)
    assert isinstance(f, Field)
    assert f.name == "cell_k"


def test_F_without_schema_defers_validation(example_schema):
    # No schema= → no eager check; the AST is built and validation happens at compile.
    f = F("typo_name")
    expr = f == 4
    # Resolution at compile time should raise.
    with pytest.raises(SchemaValidationError):
        expr.compile(example_schema.Run)


def test_schema_f_attribute_access(example_schema):
    f = example_schema.f.cell_k
    assert isinstance(f, Field)
    assert f.name == "cell_k"


def test_schema_f_unknown_name_raises_attribute_error(example_schema):
    with pytest.raises(AttributeError) as ei:
        example_schema.f.typo_name
    assert "typo_name" in str(ei.value)


def test_schema_f_dir_lists_fields(example_schema):
    names = dir(example_schema.f)
    assert "cell_k" in names
    assert "val_loss" in names
    assert names == sorted(names)


def test_F_eager_propagates_through_json_path(example_schema):
    # structural_traj is a json field in the fixture
    f = F("structural_traj", schema=example_schema).json_path("a.b")
    assert isinstance(f, Field)
    assert f._json_path == ("a", "b")


def test_schema_f_runs_in_real_query(memory_store: Store, example_schema):
    register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "completed"},
        on_duplicate="raise",
    )
    rows = memory_store.where(example_schema.f.status == "completed").all()
    assert len(rows) == 1


# ---- Fix 4: claim_if_stale + heartbeat -------------------------------------


def test_claim_if_stale_inserts_when_no_row(memory_store: Store):
    result = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "running"},
        on_duplicate="claim_if_stale",
        stale_after=_dt.timedelta(minutes=5),
    )
    assert result.was_inserted is True
    assert result.run.status == "running"


def test_claim_if_stale_skips_fresh_row(memory_store: Store):
    register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "running"},
        on_duplicate="raise",
    )
    result = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "claimed_by_other"},
        on_duplicate="claim_if_stale",
        stale_after=_dt.timedelta(hours=1),
    )
    assert result.was_inserted is False
    assert result.was_skipped is True
    assert result.run.status == "running"  # untouched


def _backdate_updated_at(store: Store, identifying: dict, delta: _dt.timedelta) -> None:
    """Helper: rewind a row's updated_at to make it look stale."""
    Run = store.schema.Run
    past = _dt.datetime.now(_dt.timezone.utc) - delta
    with store.session() as s:
        existing = s.scalar(select(Run).filter_by(**identifying))
        assert existing is not None
        existing.updated_at = past


def test_claim_if_stale_claims_stale_row(memory_store: Store):
    register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "running"},
        on_duplicate="raise",
    )
    _backdate_updated_at(memory_store, make_identifying(), _dt.timedelta(hours=2))

    result = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "reclaimed", "host": "worker-2"},
        on_duplicate="claim_if_stale",
        stale_after=_dt.timedelta(minutes=5),
    )
    assert result.was_inserted is False
    assert result.was_updated is True
    assert result.run.status == "reclaimed"
    assert result.run.host == "worker-2"


def test_claim_if_stale_requires_stale_after(memory_store: Store):
    with pytest.raises(ValueError, match="stale_after"):
        register(
            memory_store,
            identifying=make_identifying(),
            on_duplicate="claim_if_stale",
        )


def test_heartbeat_bumps_updated_at(memory_store: Store):
    register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    _backdate_updated_at(memory_store, make_identifying(), _dt.timedelta(hours=1))

    new_ts = heartbeat(memory_store, identifying=make_identifying())
    Run = memory_store.schema.Run
    with memory_store.session() as s:
        existing = s.scalar(select(Run).filter_by(**make_identifying()))
    # updated_at should be roughly the heartbeat timestamp (within a second).
    stored = existing.updated_at
    if stored.tzinfo is None:
        stored = stored.replace(tzinfo=_dt.timezone.utc)
    assert abs((stored - new_ts).total_seconds()) < 1


def test_heartbeat_raises_for_missing_row(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="no run"):
        heartbeat(memory_store, identifying=make_identifying())


def test_heartbeat_then_claim_keeps_row_fresh(memory_store: Store):
    # End-to-end of the live multi-worker pattern: register, heartbeat,
    # second worker calls claim_if_stale and skips because we're alive.
    register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "running"},
        on_duplicate="raise",
    )
    _backdate_updated_at(memory_store, make_identifying(), _dt.timedelta(hours=1))
    heartbeat(memory_store, identifying=make_identifying())  # we're alive
    result = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "claimed"},
        on_duplicate="claim_if_stale",
        stale_after=_dt.timedelta(minutes=5),
    )
    assert result.was_skipped is True
    assert result.run.status == "running"


# ---- Fix 5: float normalisation --------------------------------------------


def test_normalise_float_helper_rounds_mantissa_noise():
    assert _normalise_float(0.1 + 0.2, 12) == 0.3
    assert _normalise_float(0.3, 12) == 0.3


def test_normalise_float_skips_zero_and_special():
    assert _normalise_float(0.0, 12) == 0.0
    assert math.isnan(_normalise_float(float("nan"), 12))
    assert math.isinf(_normalise_float(float("inf"), 12))


def test_register_normalises_identifying_float(memory_store: Store):
    a = register(
        memory_store,
        identifying=make_identifying(cell_sigma=0.1 + 0.2),
        on_duplicate="raise",
    ).run
    # Second register at the canonical 0.3 should hit the same row.
    result = register(
        memory_store,
        identifying=make_identifying(cell_sigma=0.3),
        on_duplicate="return_existing",
    )
    assert result.was_inserted is False
    assert result.run.id == a.id


def test_find_normalises_identifying_float_lookups(memory_store: Store):
    register(
        memory_store,
        identifying=make_identifying(cell_sigma=0.3),
        on_duplicate="raise",
    )
    hit = find(memory_store, **make_identifying(cell_sigma=0.1 + 0.2))
    assert hit is not None


def test_dsl_normalises_identifying_float_eq(memory_store: Store):
    register(
        memory_store,
        identifying=make_identifying(cell_sigma=0.3),
        on_duplicate="raise",
    )
    rows = memory_store.where(F("cell_sigma") == 0.1 + 0.2).all()
    assert len(rows) == 1


def test_dsl_normalises_identifying_float_in(memory_store: Store):
    register(
        memory_store,
        identifying=make_identifying(cell_sigma=0.3),
        on_duplicate="raise",
    )
    rows = memory_store.where(F("cell_sigma").in_([0.1 + 0.2, 999.0])).all()
    assert len(rows) == 1


def test_annotating_floats_are_not_normalised(memory_store: Store):
    # val_loss is annotating, so the IEEE bits are preserved.
    raw = 0.1 + 0.2
    register(
        memory_store,
        identifying=make_identifying(),
        annotating={"val_loss": raw},
        on_duplicate="raise",
    )
    hit = find(memory_store, **make_identifying())
    assert hit.val_loss == raw  # NOT rounded


def test_custom_float_precision(schema_from_string):
    schema = schema_from_string(
        """
        [project]
        name = "p"
        float_precision = 3

        [identifying.x]
        type = "float"
        """
    )
    store = Store(":memory:", schema=schema, check_schema=False)
    # 0.1234 → rounded to 3 sig figs → 0.123
    register(store, identifying={"x": 0.1234}, on_duplicate="raise")
    # 0.1239 → also rounds to 0.124, NOT the same row.
    result_diff = register(
        store, identifying={"x": 0.1239}, on_duplicate="return_existing"
    )
    assert result_diff.was_inserted is True
    # 0.1235 rounds (banker's rounding) to 0.124 too — distinct from 0.123.
    # Verify 0.123 → same row as the first.
    result_same = register(
        store, identifying={"x": 0.123}, on_duplicate="return_existing"
    )
    assert result_same.was_inserted is False
