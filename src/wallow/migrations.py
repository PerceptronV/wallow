"""Alembic wrappers + snapshot mechanism + collision detection.

This module is a thin façade over `alembic.command` and `alembic.script` so
the CLI and `Store` don't shell out to `alembic`. It also implements two
wallow-specific pieces:

  * **Snapshot mechanism** — after each `migrate generate`, the current
    `wallow.toml` is copied to `alembic/snapshots/{rev}.toml`. Reviewers
    and the pre-flight diff use these to see schema state per revision.

  * **Pre-flight diff** — before invoking Alembic autogenerate, we compare
    the current TOML's identifying set against the head snapshot. If a
    field is being dropped, abort (spec §8.2 requires this); if a new
    identifying field has no `default`, abort with instruction.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from alembic import command as _alembic_command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from .errors import WallowError
from .schema import Schema, load_schema

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover - exercised only on 3.10
    import tomli as _toml


# ---- config + discovery ----------------------------------------------------


def discover_alembic_ini(start: Path | None = None) -> Path | None:
    """Walk up from `start` (defaulting to cwd) looking for `alembic.ini`.

    Returns the first match or None if none is found before the filesystem
    root. Callers that have additional candidate directories (e.g. the
    parent of a db_path) should call this multiple times.
    """
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    if here.is_file():
        here = here.parent
    for d in (here, *here.parents):
        candidate = d / "alembic.ini"
        if candidate.is_file():
            return candidate
    return None


def _resolve_sqlite_url(url: str, ini_dir: Path) -> str:
    """Resolve a relative `sqlite:///` URL against `ini_dir`.

    Keeps `alembic.ini` portable: the URL `sqlite:///runs.db` means
    "runs.db next to the ini" rather than "runs.db in cwd". Non-sqlite
    URLs and absolute paths pass through unchanged.
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    rest = url[len(prefix):]
    if rest == ":memory:" or Path(rest).is_absolute():
        return url
    return f"{prefix}{(ini_dir / rest).resolve()}"


def _make_config(
    alembic_ini_path: Path, db_url: str | None = None
) -> Config:
    """Build an Alembic Config rooted at the given `alembic.ini`.

    `script_location` and a relative `sqlite:///` URL are both anchored to
    the ini file's directory so callers can run from any cwd.
    """
    ini = Path(alembic_ini_path).resolve()
    cfg = Config(str(ini))
    script_location = cfg.get_main_option("script_location") or "alembic"
    if not Path(script_location).is_absolute():
        cfg.set_main_option(
            "script_location", str(ini.parent / script_location)
        )
    if db_url is None:
        db_url = cfg.get_main_option("sqlalchemy.url") or ""
    cfg.set_main_option("sqlalchemy.url", _resolve_sqlite_url(db_url, ini.parent))
    return cfg


# ---- revision queries ------------------------------------------------------


def current_revision(engine: Engine) -> str | None:
    """Read the current revision from the database's `alembic_version` table.

    Returns None if the table is missing (no migrations ever applied) or
    empty. Multiple-head scenarios collapse to the first row — we don't
    support branching in v1.
    """
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        return ctx.get_current_revision()


def head_revision(config: Config) -> str | None:
    """Resolve the head revision from the script directory."""
    script = ScriptDirectory.from_config(config)
    head = script.get_current_head()
    return head


def is_pending(engine: Engine, config: Config) -> bool:
    return current_revision(engine) != head_revision(config)


# ---- alembic command wrappers ---------------------------------------------


def apply(config: Config, *, target: str = "head") -> None:
    _alembic_command.upgrade(config, target)


def downgrade(config: Config, *, target: str) -> None:
    _alembic_command.downgrade(config, target)


def stamp(config: Config, *, revision: str = "head") -> None:
    _alembic_command.stamp(config, revision)


def history(config: Config) -> list[Any]:
    """Return ScriptDirectory revisions as a list (newest first)."""
    script = ScriptDirectory.from_config(config)
    return list(script.walk_revisions())


# ---- snapshot mechanism ---------------------------------------------------


def _snapshots_dir(config: Config) -> Path:
    """`{script_location}/snapshots/`. Created on first write."""
    script = ScriptDirectory.from_config(config)
    return Path(script.dir) / "snapshots"


def write_snapshot(
    revision_id: str, schema_path: Path, snapshots_dir: Path
) -> Path:
    """Copy `schema_path` to `{snapshots_dir}/{revision_id}.toml`.

    Prepends a three-line header marking the file as auto-generated. TOML
    treats `#` lines as comments so the file remains a valid wallow schema.
    """
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    dest = snapshots_dir / f"{revision_id}.toml"
    body = Path(schema_path).read_text()
    header = (
        "# wallow migration snapshot — DO NOT EDIT\n"
        f"# revision = {revision_id}\n"
        f"# generated = {_dt.datetime.now(_dt.timezone.utc).isoformat()}\n\n"
    )
    dest.write_text(header + body)
    return dest


def _load_snapshot(snapshots_dir: Path, revision_id: str) -> Schema | None:
    """Return the parsed Schema for `revision_id`, or None if missing."""
    p = snapshots_dir / f"{revision_id}.toml"
    if not p.is_file():
        return None
    return load_schema(p)


# ---- pre-flight diff -------------------------------------------------------


def _preflight_identifying_drop(
    new_schema: Schema, head_schema: Schema
) -> None:
    """Abort if the new schema drops any identifying field from the head.

    Spec §8.2: removing an identifying field can produce duplicate keys.
    The user must resolve manually first via `find_collisions_after_drop`.
    """
    dropped = head_schema.identifying - new_schema.identifying
    if dropped:
        names = sorted(dropped)
        raise WallowError(
            f"refusing to generate migration: identifying field(s) {names} "
            f"are being dropped. Removing an identifying field can produce "
            f"duplicate keys. Inspect collisions first:\n"
            f"    from wallow.migrations import find_collisions_after_drop\n"
            f"    find_collisions_after_drop(store, '{names[0]}')\n"
            f"Then either delete duplicates or merge their annotating data."
        )


def _preflight_new_identifying_default(
    new_schema: Schema, head_schema: Schema
) -> None:
    """Abort if a new identifying field is added without a default.

    Spec §9: NOT NULL columns without a default can't backfill existing
    rows. We detect this here so the user gets a clear error before
    Alembic generates an unapplyable migration.
    """
    added = new_schema.identifying - head_schema.identifying
    missing_default = [
        name for name in sorted(added) if new_schema.field(name).default is None
    ]
    if missing_default:
        raise WallowError(
            f"refusing to generate migration: new identifying field(s) "
            f"{missing_default} have no `default` in wallow.toml. "
            f"NOT NULL columns can't be added to a non-empty table without "
            f"a default. Add `default = ...` to each field, or use a "
            f"manual migration if the table is empty."
        )


# ---- generate --------------------------------------------------------------


def generate(
    config: Config,
    *,
    message: str,
    schema_path: Path,
    snapshots_dir: Path | None = None,
) -> Path:
    """Run autogenerate and write a snapshot. Returns the new revision file path.

    Pre-flight checks:
      1. Identifying drop — abort with collision-detection guidance.
      2. New identifying field without `default` — abort with instruction.

    Both checks require a head snapshot; if none exists (first migration)
    they're skipped.
    """
    schema_path = Path(schema_path).resolve()
    new_schema = load_schema(schema_path)

    if snapshots_dir is None:
        snapshots_dir = _snapshots_dir(config)

    head = head_revision(config)
    if head is not None:
        head_schema = _load_snapshot(snapshots_dir, head)
        if head_schema is not None:
            _preflight_identifying_drop(new_schema, head_schema)
            _preflight_new_identifying_default(new_schema, head_schema)

    # `alembic.command.revision` returns a Script (or list of them when
    # branches are involved). v1 doesn't branch, so we expect a single
    # Script — but tolerate both shapes defensively.
    result = _alembic_command.revision(
        config, message=message, autogenerate=True
    )
    script = result[0] if isinstance(result, list) else result
    if script is None:
        raise WallowError(
            "alembic produced no revision (no changes detected?)"
        )
    revision_id = script.revision
    revision_path = Path(script.path)

    write_snapshot(revision_id, schema_path, snapshots_dir)
    return revision_path


# ---- collision detection --------------------------------------------------


@dataclass(frozen=True)
class CollisionGroup:
    """Rows that would collide if `field_name` were dropped from identifying.

    `field_values` carries the values of the *remaining* identifying fields
    that were equal across the colliding rows; `row_ids` lists the `runs.id`
    primary keys of the rows in this group (always >=2 for a collision).
    """

    field_values: dict[str, Any]
    row_ids: list[int]


def find_collisions_after_drop(
    store: Any, field_name: str
) -> list[CollisionGroup]:
    """Find groups of rows that would collide if `field_name` were dropped.

    Empty list = safe to drop. Otherwise each `CollisionGroup` describes a
    set of rows whose remaining identifying fields are identical, so they'd
    collapse into a single row violating UNIQUE.

    Implementation: GROUP BY (identifying \\ {field_name}) HAVING count > 1,
    then SELECT id for each group.
    """
    schema = store.schema
    if field_name not in schema.identifying:
        raise WallowError(
            f"{field_name!r} is not an identifying field; "
            f"identifying = {sorted(schema.identifying)}"
        )

    Run = schema.Run
    remaining_names = sorted(schema.identifying - {field_name})
    if not remaining_names:
        # Dropping the only identifying field collapses every row.
        with store.session() as s:
            ids = [r.id for r in s.scalars(select(Run))]
        if len(ids) > 1:
            return [CollisionGroup(field_values={}, row_ids=ids)]
        return []

    remaining_cols = [getattr(Run, name) for name in remaining_names]

    with store.session() as s:
        # Stage 1: find tuples with >1 rows.
        group_q = (
            select(*remaining_cols, func.count().label("n"))
            .group_by(*remaining_cols)
            .having(func.count() > 1)
        )
        groups = list(s.execute(group_q))

        results: list[CollisionGroup] = []
        for row in groups:
            values = {name: row[i] for i, name in enumerate(remaining_names)}
            # Stage 2: fetch the colliding row ids for this tuple.
            id_q = select(Run.id).where(
                *(getattr(Run, name) == values[name] for name in remaining_names)
            )
            ids = sorted(s.scalars(id_q).all())
            results.append(CollisionGroup(field_values=values, row_ids=ids))
        return results


__all__ = [
    "CollisionGroup",
    "apply",
    "current_revision",
    "discover_alembic_ini",
    "downgrade",
    "find_collisions_after_drop",
    "generate",
    "head_revision",
    "history",
    "is_pending",
    "stamp",
    "write_snapshot",
]
