"""Tests for Phase 3 — Alembic migrations + snapshot mechanism + collision detection."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from wallow import (
    CollisionGroup,
    PendingMigrationError,
    Store,
    WallowError,
    find_collisions_after_drop,
    load_schema,
    register,
)
from wallow import migrations
from wallow.cli import main as cli_main


# ---------------------------------------------------------------------------
# Bootstrap fixture: a freshly-initialized wallow project under tmp_path.
# Calling cli.main(['init', ...]) instead of replicating the file copies so
# the tests exercise the same code path users will hit.
# ---------------------------------------------------------------------------


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    rc = cli_main(["init", "--dir", str(tmp_path), "--db", "runs.db"])
    assert rc == 0
    # Replace the minimal init template with the example schema so we can
    # reuse make_identifying() and exercise the full identifying set.
    (tmp_path / "wallow.toml").write_text(
        (Path(__file__).parent / "fixtures" / "example_wallow.toml").read_text()
    )
    return tmp_path


def _config(project_dir: Path):
    return migrations._make_config(project_dir / "alembic.ini")


def _db_url(project_dir: Path) -> str:
    return f"sqlite:///{project_dir / 'runs.db'}"


def _generate(project_dir: Path, message: str) -> Path:
    return migrations.generate(
        _config(project_dir),
        message=message,
        schema_path=project_dir / "wallow.toml",
    )


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_creates_expected_files(tmp_path: Path):
    rc = cli_main(["init", "--dir", str(tmp_path), "--db", "runs.db"])
    assert rc == 0
    assert (tmp_path / "wallow.toml").is_file()
    assert (tmp_path / "alembic.ini").is_file()
    assert (tmp_path / "alembic" / "env.py").is_file()
    assert (tmp_path / "alembic" / "script.py.mako").is_file()
    assert (tmp_path / "alembic" / "versions").is_dir()
    assert (tmp_path / "alembic" / "snapshots").is_dir()


def test_init_refuses_if_files_exist(tmp_path: Path):
    cli_main(["init", "--dir", str(tmp_path)])
    rc = cli_main(["init", "--dir", str(tmp_path)])
    assert rc != 0


def test_init_force_overwrites(tmp_path: Path):
    cli_main(["init", "--dir", str(tmp_path)])
    (tmp_path / "wallow.toml").write_text("# user edits")
    rc = cli_main(["init", "--dir", str(tmp_path), "--force"])
    assert rc == 0
    # The template re-rendered, dropping the user's hand edits.
    assert "user edits" not in (tmp_path / "wallow.toml").read_text()


# ---------------------------------------------------------------------------
# generate / apply / snapshot
# ---------------------------------------------------------------------------


def test_generate_writes_revision_and_snapshot(project_dir: Path):
    rev_path = _generate(project_dir, "initial schema")
    assert rev_path.is_file()
    # Snapshot named by revision id.
    snapshots = list((project_dir / "alembic" / "snapshots").glob("*.toml"))
    assert len(snapshots) == 1
    snap = snapshots[0].read_text()
    assert "wallow migration snapshot" in snap
    assert "DO NOT EDIT" in snap
    assert "[project]" in snap  # the schema body is appended


def test_apply_brings_db_to_head(project_dir: Path):
    _generate(project_dir, "initial")
    cfg = _config(project_dir)
    migrations.apply(cfg)
    engine = create_engine(_db_url(project_dir), future=True)
    assert migrations.current_revision(engine) == migrations.head_revision(cfg)
    # `runs` table created.
    assert "runs" in inspect(engine).get_table_names()


def test_generate_after_adding_annotating_field(project_dir: Path):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    toml = (project_dir / "wallow.toml").read_text()
    toml += '\n[annotating.notes]\ntype = "string"\n'
    (project_dir / "wallow.toml").write_text(toml)

    rev_path = _generate(project_dir, "add notes")
    body = rev_path.read_text()
    assert "add_column" in body
    assert "notes" in body


# ---------------------------------------------------------------------------
# pre-flight: missing default + identifying drop
# ---------------------------------------------------------------------------


def test_add_identifying_field_without_default_aborts(project_dir: Path):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    toml = (project_dir / "wallow.toml").read_text()
    toml += '\n[identifying.warmup_steps]\ntype = "int"\n'
    (project_dir / "wallow.toml").write_text(toml)

    with pytest.raises(WallowError, match="no `default`"):
        _generate(project_dir, "add warmup")


def test_add_identifying_field_with_default_applies_cleanly(
    project_dir: Path,
):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    schema = load_schema(project_dir / "wallow.toml")
    store = Store(project_dir / "runs.db", schema=schema, check_schema=False)
    register(
        store,
        identifying={
            "cell_k": 4, "cell_sigma": 0.1, "generation": 0,
            "candidate_id": 1, "seed": 0,
        },
        on_duplicate="raise",
    )
    store.engine.dispose()

    toml = (project_dir / "wallow.toml").read_text()
    toml += '\n[identifying.warmup_steps]\ntype = "int"\ndefault = 0\n'
    (project_dir / "wallow.toml").write_text(toml)

    _generate(project_dir, "add warmup")
    migrations.apply(_config(project_dir))

    # The pre-existing row was backfilled with warmup_steps=0.
    engine = create_engine(_db_url(project_dir), future=True)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT warmup_steps FROM runs")).all()
    assert rows == [(0,)]


def test_drop_identifying_field_aborts(project_dir: Path):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    # Drop `seed` from identifying.
    toml = (project_dir / "wallow.toml").read_text()
    toml = toml.replace(
        '[identifying.seed]\ntype = "int"\ndefault = 0\n',
        "",
    )
    (project_dir / "wallow.toml").write_text(toml)

    with pytest.raises(WallowError, match="dropped"):
        _generate(project_dir, "drop seed")


# ---------------------------------------------------------------------------
# find_collisions_after_drop
# ---------------------------------------------------------------------------


def test_find_collisions_after_drop_safe(project_dir: Path):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    schema = load_schema(project_dir / "wallow.toml")
    store = Store(project_dir / "runs.db", schema=schema, check_schema=False)
    register(
        store,
        identifying={
            "cell_k": 4, "cell_sigma": 0.1, "generation": 0,
            "candidate_id": 1, "seed": 0,
        },
        on_duplicate="raise",
    )
    # Single row, no collisions possible regardless of which field is dropped.
    assert find_collisions_after_drop(store, "seed") == []


def test_find_collisions_after_drop_returns_groups(project_dir: Path):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    schema = load_schema(project_dir / "wallow.toml")
    store = Store(project_dir / "runs.db", schema=schema, check_schema=False)
    base = dict(cell_k=4, cell_sigma=0.1, generation=0, candidate_id=1)
    # Two rows that differ only in `seed` — would collapse if `seed` is dropped.
    register(store, identifying={**base, "seed": 0}, on_duplicate="raise")
    register(store, identifying={**base, "seed": 1}, on_duplicate="raise")
    # An unrelated single row.
    register(
        store,
        identifying={**base, "candidate_id": 2, "seed": 0},
        on_duplicate="raise",
    )

    groups = find_collisions_after_drop(store, "seed")
    assert len(groups) == 1
    g = groups[0]
    assert isinstance(g, CollisionGroup)
    assert g.field_values == {
        "cell_k": 4, "cell_sigma": 0.1, "generation": 0, "candidate_id": 1,
    }
    assert len(g.row_ids) == 2


def test_find_collisions_rejects_non_identifying_field(project_dir: Path):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))
    schema = load_schema(project_dir / "wallow.toml")
    store = Store(project_dir / "runs.db", schema=schema, check_schema=False)
    with pytest.raises(WallowError, match="not an identifying"):
        find_collisions_after_drop(store, "status")


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------


def test_downgrade_reverses_change(project_dir: Path):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    toml = (project_dir / "wallow.toml").read_text()
    toml += '\n[annotating.notes]\ntype = "string"\n'
    (project_dir / "wallow.toml").write_text(toml)
    _generate(project_dir, "add notes")
    migrations.apply(_config(project_dir))

    # Downgrade by one step.
    migrations.downgrade(_config(project_dir), target="-1")
    engine = create_engine(_db_url(project_dir), future=True)
    cols = {c["name"] for c in inspect(engine).get_columns("runs")}
    assert "notes" not in cols


# ---------------------------------------------------------------------------
# stamp + check_schema integration
# ---------------------------------------------------------------------------


def test_pending_migration_error_from_check_schema(project_dir: Path):
    # Generate + apply once to install runs + alembic_version.
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    # Add an annotating field, generate but do NOT apply.
    toml = (project_dir / "wallow.toml").read_text()
    toml += '\n[annotating.notes]\ntype = "string"\n'
    (project_dir / "wallow.toml").write_text(toml)
    _generate(project_dir, "add notes")

    schema = load_schema(project_dir / "wallow.toml")
    with pytest.raises(PendingMigrationError) as ei:
        Store(project_dir / "runs.db", schema=schema, check_schema=True)
    assert ei.value.current_rev is not None
    assert ei.value.head_rev is not None
    assert ei.value.current_rev != ei.value.head_rev


def test_store_migrate_applies_pending(project_dir: Path):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    toml = (project_dir / "wallow.toml").read_text()
    toml += '\n[annotating.notes]\ntype = "string"\n'
    (project_dir / "wallow.toml").write_text(toml)
    _generate(project_dir, "add notes")

    schema = load_schema(project_dir / "wallow.toml")
    store = Store(project_dir / "runs.db", schema=schema, check_schema=False)
    store.migrate()
    # Now check_schema should pass.
    store2 = Store(project_dir / "runs.db", schema=schema, check_schema=True)
    assert store2.check_schema() is None


# ---------------------------------------------------------------------------
# concurrent register still works after the gated-create_all change
# ---------------------------------------------------------------------------


def _worker_concurrent(db_path: str, schema_path: str) -> str:
    from wallow import DuplicateRunError, Store, load_schema, register

    schema = load_schema(schema_path)
    store = Store(db_path, schema=schema, check_schema=False)
    try:
        register(
            store,
            identifying={
                "cell_k": 4, "cell_sigma": 0.1, "generation": 0,
                "candidate_id": 1, "seed": 0,
            },
            on_duplicate="raise",
        )
        return "inserted"
    except DuplicateRunError:
        return "duplicate"


def test_concurrent_register_one_winner_after_migrations(project_dir: Path):
    _generate(project_dir, "initial")
    migrations.apply(_config(project_dir))

    db_path = str(project_dir / "runs.db")
    schema_path = str(project_dir / "wallow.toml")

    # Open a Store once to set journal_mode=WAL (and dispose) — matches the
    # Phase 1/2 concurrency-test bootstrap. Without WAL, two SQLite writers
    # in the rollback-journal mode can deadlock under contention.
    schema = load_schema(schema_path)
    bootstrap = Store(db_path, schema=schema, check_schema=False)
    bootstrap.engine.dispose()

    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        results = pool.starmap(
            _worker_concurrent,
            [(db_path, schema_path), (db_path, schema_path)],
        )
    assert sorted(results) == ["duplicate", "inserted"]
