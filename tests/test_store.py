"""Tests for wallow.store: register, find, session, pragmas, concurrency."""

from __future__ import annotations

import datetime as _dt
import multiprocessing as mp
import os
from pathlib import Path

import pytest
from sqlalchemy import text

from wallow import (
    DuplicateRunError,
    SchemaValidationError,
    Store,
    find,
    register,
)

from conftest import make_identifying


# ---- happy path -------------------------------------------------------------


def test_register_inserts_new_run(memory_store: Store):
    result = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "running", "val_accuracy": 0.8},
        on_duplicate="raise",
    )
    assert result.was_inserted is True
    assert result.was_updated is False
    assert result.was_skipped is False
    run = result.run
    assert run.id is not None
    assert run.cell_k == 4
    assert run.status == "running"


def test_register_omitted_annotating_defaults_to_null(memory_store: Store):
    run = register(
        memory_store,
        identifying=make_identifying(),
        on_duplicate="raise",
    ).run
    assert run.status is None
    assert run.val_accuracy is None


def test_find_returns_existing(memory_store: Store):
    register(
        memory_store, identifying=make_identifying(), on_duplicate="raise"
    )
    hit = find(memory_store, **make_identifying())
    assert hit is not None
    assert hit.cell_k == 4


def test_find_returns_none_for_missing(memory_store: Store):
    assert find(memory_store, **make_identifying()) is None


# ---- on_duplicate variants --------------------------------------------------


def test_on_duplicate_raise(memory_store: Store):
    register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    with pytest.raises(DuplicateRunError) as ei:
        register(
            memory_store, identifying=make_identifying(), on_duplicate="raise"
        )
    assert ei.value.existing.cell_k == 4


def test_on_duplicate_return_existing(memory_store: Store):
    first = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "running"},
        on_duplicate="raise",
    ).run
    result = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "different"},
        on_duplicate="return_existing",
    )
    assert result.was_inserted is False
    assert result.was_updated is False
    assert result.was_skipped is False
    second = result.run
    assert second.id == first.id
    # return_existing does NOT update annotating fields
    assert second.status == "running"


def test_on_duplicate_skip(memory_store: Store):
    first = register(
        memory_store, identifying=make_identifying(), on_duplicate="raise"
    ).run
    result = register(
        memory_store, identifying=make_identifying(), on_duplicate="skip"
    )
    assert result.run is None
    assert result.was_inserted is False
    assert result.was_skipped is True
    # Confirm the original is still there.
    assert find(memory_store, **make_identifying()).id == first.id


def test_on_duplicate_overwrite(memory_store: Store):
    first = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "running", "val_accuracy": 0.5},
        on_duplicate="raise",
    ).run
    result = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "completed", "val_accuracy": 0.9},
        on_duplicate="overwrite",
    )
    assert result.was_inserted is False
    assert result.was_updated is True
    updated = result.run
    assert updated.id == first.id
    assert updated.status == "completed"
    assert updated.val_accuracy == 0.9


def test_invalid_on_duplicate_raises(memory_store: Store):
    with pytest.raises(ValueError, match="on_duplicate"):
        register(
            memory_store,
            identifying=make_identifying(),
            on_duplicate="bogus",  # type: ignore[arg-type]
        )


# ---- schema / type validation ----------------------------------------------


def test_register_missing_identifying_key(memory_store: Store):
    # cell_k has no `default` in the fixture, so omitting it is a hard error.
    keys = make_identifying()
    keys.pop("cell_k")
    with pytest.raises(SchemaValidationError) as ei:
        register(memory_store, identifying=keys, on_duplicate="raise")
    assert "cell_k" in ei.value.missing_keys


def test_register_extra_identifying_key(memory_store: Store):
    keys = make_identifying()
    keys["bogus"] = 1
    with pytest.raises(SchemaValidationError) as ei:
        register(memory_store, identifying=keys, on_duplicate="raise")
    assert "bogus" in ei.value.extra_keys


def test_register_unknown_annotating_key(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="unknown annotating"):
        register(
            memory_store,
            identifying=make_identifying(),
            annotating={"bogus": 1},
            on_duplicate="raise",
        )


def test_register_wrong_type(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="expected int"):
        register(
            memory_store,
            identifying=make_identifying(cell_k="four"),
            on_duplicate="raise",
        )


def test_register_bool_for_int_rejected(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="expected int"):
        register(
            memory_store,
            identifying=make_identifying(cell_k=True),
            on_duplicate="raise",
        )


def test_register_int_for_float_accepted(memory_store: Store):
    # Spec allows int for float field.
    run = register(
        memory_store,
        identifying=make_identifying(cell_sigma=2),
        on_duplicate="raise",
    ).run
    assert run.cell_sigma == 2


def test_register_nan_in_identifying_rejected(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="NaN"):
        register(
            memory_store,
            identifying=make_identifying(cell_sigma=float("nan")),
            on_duplicate="raise",
        )


def test_register_naive_datetime_rejected(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="naive datetime"):
        register(
            memory_store,
            identifying=make_identifying(),
            annotating={"started_at": _dt.datetime(2026, 1, 1, 12, 0, 0)},
            on_duplicate="raise",
        )


def test_register_aware_datetime_accepted(memory_store: Store):
    when = _dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)
    run = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"started_at": when},
        on_duplicate="raise",
    ).run
    assert run.started_at is not None


def test_register_unjsonable_rejected(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="not JSON-serialisable"):
        register(
            memory_store,
            identifying=make_identifying(),
            annotating={"discovered_T": {1, 2, 3}},  # set isn't JSON
            on_duplicate="raise",
        )


# ---- pragma checks ----------------------------------------------------------


def test_wal_journal_mode_on_file_db(file_store: Store):
    with file_store.session() as s:
        mode = s.execute(text("PRAGMA journal_mode")).scalar()
    assert mode == "wal"


def test_foreign_keys_pragma_on(memory_store: Store):
    with memory_store.session() as s:
        fk = s.execute(text("PRAGMA foreign_keys")).scalar()
    assert fk == 1


# ---- check_schema / migrate -----------------------------------------------


def test_migrate_no_alembic_ini_raises_clear_error(memory_store: Store):
    from wallow import WallowError

    with pytest.raises(WallowError, match="alembic.ini"):
        memory_store.migrate()


def test_check_schema_no_alembic_ini_returns_silently(memory_store: Store):
    # When Alembic isn't configured for this project, opening a Store with
    # check_schema=True must not raise — preserves the Phase 1/2 quick-start.
    assert memory_store.check_schema() is None


# ---- concurrent register ---------------------------------------------------


def _worker_register(db_path: str, schema_path: str) -> str:
    """Helper run in a separate process; returns 'inserted' or 'duplicate'."""
    from wallow import Store, load_schema, register, DuplicateRunError

    schema = load_schema(schema_path)
    store = Store(db_path, schema=schema, check_schema=False)
    try:
        register(
            store,
            identifying={
                "cell_k": 4,
                "cell_sigma": 0.1,
                "generation": 0,
                "candidate_id": 1,
                "seed": 0,
            },
            on_duplicate="raise",
        )
        return "inserted"
    except DuplicateRunError:
        return "duplicate"


def test_concurrent_register_one_winner(tmp_path: Path):
    db_path = tmp_path / "concurrent.db"
    schema_path = Path(__file__).parent / "fixtures" / "example_wallow.toml"

    # Pre-create the DB so the schema exists; then both workers race.
    from wallow import load_schema

    schema = load_schema(schema_path)
    Store(db_path, schema=schema, check_schema=False)

    # spawn so children get a fresh interpreter — avoids fork+SQLite quirks
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        results = pool.starmap(
            _worker_register,
            [(str(db_path), str(schema_path)), (str(db_path), str(schema_path))],
        )
    assert sorted(results) == ["duplicate", "inserted"]


# ---- uuid column ------------------------------------------------------------


def test_register_inserts_uuid(memory_store: Store):
    run = register(
        memory_store, identifying=make_identifying(), on_duplicate="raise"
    ).run
    assert run.uuid is not None
    assert isinstance(run.uuid, str)
    assert len(run.uuid) == 12
    int(run.uuid, 16)  # valid hex


def test_uuid_stable_across_overwrite(memory_store: Store):
    first = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "running"},
        on_duplicate="raise",
    ).run
    original_uuid = first.uuid
    result = register(
        memory_store,
        identifying=make_identifying(),
        annotating={"status": "completed", "val_accuracy": 0.9},
        on_duplicate="overwrite",
    )
    assert result.was_updated
    assert result.run.uuid == original_uuid
    assert result.run.status == "completed"


def test_uuid_unique_across_inserts(memory_store: Store):
    uuids = set()
    for i in range(10):
        run = register(
            memory_store,
            identifying=make_identifying(candidate_id=i),
            on_duplicate="raise",
        ).run
        uuids.add(run.uuid)
    assert len(uuids) == 10


def test_uuid_cannot_be_passed_as_annotation(memory_store: Store):
    # `uuid` is reserved (not in schema.annotating); register should reject it.
    with pytest.raises(SchemaValidationError, match="unknown annotating"):
        register(
            memory_store,
            identifying=make_identifying(),
            annotating={"uuid": "deadbeefcafe"},
            on_duplicate="raise",
        )


def test_find_by_uuid(memory_store: Store):
    run = register(
        memory_store, identifying=make_identifying(), on_duplicate="raise"
    ).run
    hit = memory_store.find_by_uuid(run.uuid)
    assert hit is not None
    assert hit.id == run.id


def test_find_by_uuid_returns_none_when_missing(memory_store: Store):
    assert memory_store.find_by_uuid("nonexistent1") is None


# ---- Store.artefacts_dir ----------------------------------------------------


def test_artefacts_dir_requires_root(memory_store: Store):
    """example_schema doesn't set artefacts_root; the helper should refuse."""
    from wallow import WallowError

    run = register(
        memory_store, identifying=make_identifying(), on_duplicate="raise"
    ).run
    with pytest.raises(WallowError, match="artefacts_root"):
        memory_store.artefacts_dir(run)


def test_artefacts_dir_default_layout_is_uuid(schema_from_string, tmp_path):
    s = schema_from_string(
        f"""
        [project]
        name = "p"
        artefacts_root = "{tmp_path}"
        [identifying.k]
        type = "int"
        """
    )
    store = Store(":memory:", schema=s, check_schema=False)
    run = register(store, identifying={"k": 1}, on_duplicate="raise").run
    assert store.artefacts_dir(run) == tmp_path / run.uuid


def test_artefacts_dir_substitutes_layout(schema_from_string, tmp_path):
    s = schema_from_string(
        f"""
        [project]
        name = "p"
        artefacts_root = "{tmp_path}"
        artefacts_layout = "{{architecture}}/{{uuid}}"
        [identifying.architecture]
        type = "string"
        """
    )
    store = Store(":memory:", schema=s, check_schema=False)
    run = register(
        store, identifying={"architecture": "resnet18"}, on_duplicate="raise"
    ).run
    expected = tmp_path / "resnet18" / run.uuid
    assert store.artefacts_dir(run) == expected


def test_artefacts_dir_sanitises_substitutions(schema_from_string, tmp_path):
    s = schema_from_string(
        f"""
        [project]
        name = "p"
        artefacts_root = "{tmp_path}"
        artefacts_layout = "{{label}}/{{uuid}}"
        [identifying.label]
        type = "string"
        """
    )
    store = Store(":memory:", schema=s, check_schema=False)
    # Spaces, accents, slashes should all be normalised away.
    run = register(
        store, identifying={"label": "Hello World / café"}, on_duplicate="raise"
    ).run
    out = store.artefacts_dir(run)
    # The label component should be lowercase ASCII with no spaces/slashes.
    assert "Hello World" not in str(out)
    assert "café" not in str(out)
    assert "Hello_World_cafe" in str(out)


def test_artefacts_dir_appends_parts_and_mkdir(schema_from_string, tmp_path):
    s = schema_from_string(
        f"""
        [project]
        name = "p"
        artefacts_root = "{tmp_path}"
        [identifying.k]
        type = "int"
        """
    )
    store = Store(":memory:", schema=s, check_schema=False)
    run = register(store, identifying={"k": 1}, on_duplicate="raise").run
    out = store.artefacts_dir(run, "checkpoints", "best.pt", mkdir=False)
    assert out == tmp_path / run.uuid / "checkpoints" / "best.pt"
    # mkdir=True creates the parent dir (since the path includes a file part,
    # this creates `<root>/<uuid>/checkpoints/best.pt` as a directory — that
    # is the documented behaviour: mkdir always operates on the full path).
    out2 = store.artefacts_dir(run, "logs", mkdir=True)
    assert out2.is_dir()
