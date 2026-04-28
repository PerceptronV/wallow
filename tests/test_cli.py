"""Tests for Phase 4 — the `wallow` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from wallow import Store, load_schema, register
from wallow import migrations
from wallow.cli import main as cli_main


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Initialized wallow project with the example schema applied."""
    cli_main(["init", "--dir", str(tmp_path), "--db", "runs.db"])
    (tmp_path / "wallow.toml").write_text(
        (Path(__file__).parent / "fixtures" / "example_wallow.toml").read_text()
    )
    cli_main(
        [
            "migrate", "generate", "initial",
            "--alembic-ini", str(tmp_path / "alembic.ini"),
        ]
    )
    cli_main(
        ["migrate", "apply", "--alembic-ini", str(tmp_path / "alembic.ini")]
    )
    return tmp_path


# ---- init -----------------------------------------------------------------


def test_cli_init_happy_path(tmp_path: Path):
    rc = cli_main(["init", "--dir", str(tmp_path), "--db", "runs.db"])
    assert rc == 0
    assert (tmp_path / "wallow.toml").is_file()
    assert (tmp_path / "alembic.ini").is_file()


def test_cli_init_refuses_existing(tmp_path: Path, capsys):
    cli_main(["init", "--dir", str(tmp_path)])
    rc = cli_main(["init", "--dir", str(tmp_path)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "refusing" in err


def test_cli_init_force(tmp_path: Path):
    cli_main(["init", "--dir", str(tmp_path)])
    (tmp_path / "wallow.toml").write_text("# user edits")
    rc = cli_main(["init", "--dir", str(tmp_path), "--force"])
    assert rc == 0
    assert "user edits" not in (tmp_path / "wallow.toml").read_text()


# ---- status ---------------------------------------------------------------


def test_cli_status_no_alembic_ini(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = cli_main(["status"])
    assert rc != 0
    assert "alembic.ini" in capsys.readouterr().err


def test_cli_status_pending_after_init_before_generate(tmp_path: Path, capsys):
    """Right after init, no migrations exist yet — status should print
    `<none>` for both head and current and exit 0 (head==current==None)."""
    cli_main(["init", "--dir", str(tmp_path), "--db", "runs.db"])
    rc = cli_main(
        ["status", "--alembic-ini", str(tmp_path / "alembic.ini")]
    )
    out = capsys.readouterr().out
    assert "head rev:      <none>" in out
    assert "current rev:   <none>" in out
    assert rc == 0  # nothing to apply


def test_cli_status_in_sync(project_dir: Path, capsys):
    rc = cli_main(
        ["status", "--alembic-ini", str(project_dir / "alembic.ini")]
    )
    out = capsys.readouterr().out
    assert "pending:       no" in out
    assert rc == 0


def test_cli_status_pending_after_schema_edit(project_dir: Path, capsys):
    toml = (project_dir / "wallow.toml").read_text()
    toml += '\n[annotating.notes]\ntype = "string"\n'
    (project_dir / "wallow.toml").write_text(toml)
    cli_main(
        [
            "migrate", "generate", "add notes",
            "--alembic-ini", str(project_dir / "alembic.ini"),
        ]
    )
    rc = cli_main(
        ["status", "--alembic-ini", str(project_dir / "alembic.ini")]
    )
    out = capsys.readouterr().out
    assert "pending:       yes" in out
    assert rc != 0


def test_cli_status_run_count(project_dir: Path, capsys):
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

    cli_main(["status", "--alembic-ini", str(project_dir / "alembic.ini")])
    out = capsys.readouterr().out
    assert "run count:     1" in out


# ---- inspect --------------------------------------------------------------


def test_cli_inspect_existing_run(project_dir: Path, capsys):
    schema = load_schema(project_dir / "wallow.toml")
    store = Store(project_dir / "runs.db", schema=schema, check_schema=False)
    run = register(
        store,
        identifying={
            "cell_k": 4, "cell_sigma": 0.1, "generation": 0,
            "candidate_id": 1, "seed": 0,
        },
        annotating={"status": "running"},
        on_duplicate="raise",
    )
    run_id = run.id
    store.engine.dispose()

    rc = cli_main(
        [
            "inspect", str(run_id),
            "--alembic-ini", str(project_dir / "alembic.ini"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert f"id:           {run_id}" in out
    assert "cell_k: 4" in out
    assert "status: 'running'" in out


def test_cli_inspect_missing_id(project_dir: Path, capsys):
    rc = cli_main(
        [
            "inspect", "9999",
            "--alembic-ini", str(project_dir / "alembic.ini"),
        ]
    )
    assert rc != 0
    assert "no run with id=9999" in capsys.readouterr().err


# ---- migrate {history, stamp, downgrade} ----------------------------------


def test_cli_migrate_history_lists_revisions(project_dir: Path, capsys):
    rc = cli_main(
        ["migrate", "history", "--alembic-ini", str(project_dir / "alembic.ini")]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "initial" in out


def test_cli_migrate_stamp(tmp_path: Path):
    cli_main(["init", "--dir", str(tmp_path), "--db", "runs.db"])
    (tmp_path / "wallow.toml").write_text(
        (Path(__file__).parent / "fixtures" / "example_wallow.toml").read_text()
    )
    cli_main(
        ["migrate", "generate", "initial",
         "--alembic-ini", str(tmp_path / "alembic.ini")]
    )
    # Stamp instead of apply: records the revision without running DDL.
    rc = cli_main(
        ["migrate", "stamp", "head",
         "--alembic-ini", str(tmp_path / "alembic.ini")]
    )
    assert rc == 0

    cfg = migrations._make_config(tmp_path / "alembic.ini")
    from sqlalchemy import create_engine
    engine = create_engine(
        cfg.get_main_option("sqlalchemy.url"), future=True
    )
    assert migrations.current_revision(engine) == migrations.head_revision(cfg)


def test_cli_migrate_downgrade_base_requires_yes(project_dir: Path, capsys):
    rc = cli_main(
        ["migrate", "downgrade", "base",
         "--alembic-ini", str(project_dir / "alembic.ini")]
    )
    assert rc != 0
    assert "--yes" in capsys.readouterr().err


def test_cli_migrate_downgrade_one_step(project_dir: Path):
    toml = (project_dir / "wallow.toml").read_text()
    toml += '\n[annotating.notes]\ntype = "string"\n'
    (project_dir / "wallow.toml").write_text(toml)
    cli_main(
        ["migrate", "generate", "add notes",
         "--alembic-ini", str(project_dir / "alembic.ini")]
    )
    cli_main(
        ["migrate", "apply",
         "--alembic-ini", str(project_dir / "alembic.ini")]
    )
    rc = cli_main(
        ["migrate", "downgrade", "-1",
         "--alembic-ini", str(project_dir / "alembic.ini")]
    )
    assert rc == 0
