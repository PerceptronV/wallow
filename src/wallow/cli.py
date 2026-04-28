"""`wallow` CLI: init, migrate {generate, apply, downgrade, history, stamp}, status, inspect.

The CLI is thin: it parses args, delegates to `migrations` / `store`, and
formats output. Errors print to stderr with a non-zero exit code; argparse
errors exit with code 2 (its default).

Templates ship under `wallow.templates` and are read via `importlib.resources`
so the CLI works under editable installs, wheels, and zipped wheels alike.
"""

from __future__ import annotations

import argparse
import configparser
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

from . import migrations
from .errors import WallowError
from .schema import load_schema


# ---- exit codes -----------------------------------------------------------

EXIT_OK = 0
EXIT_FAIL = 1


# ---- helpers --------------------------------------------------------------


def _err(msg: str) -> None:
    sys.stderr.write(f"wallow: error: {msg}\n")


def _resolve_alembic_ini(explicit: str | None) -> Path:
    """Discover the alembic.ini, raising a friendly error if missing."""
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_file():
            raise WallowError(f"alembic.ini not found at {p}")
        return p
    discovered = migrations.discover_alembic_ini()
    if discovered is None:
        raise WallowError(
            "no alembic.ini found; pass --alembic-ini or run `wallow init`"
        )
    return discovered


def _db_url_from_ini(ini_path: Path) -> str:
    """Read sqlalchemy.url from the ini, resolving relative SQLite paths.

    Used by `inspect` and `status` when they need to open a Store but don't
    need an Alembic Config. Mirrors `migrations._resolve_sqlite_url` so the
    ini stays portable.
    """
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(ini_path)
    raw = parser.get("alembic", "sqlalchemy.url")
    return migrations._resolve_sqlite_url(raw, ini_path.parent)


def _schema_path_from_ini(ini_path: Path) -> Path:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(ini_path)
    rel = parser.get("alembic", "wallow_schema", fallback="wallow.toml")
    return (ini_path.parent / rel).resolve()


def _db_path_from_url(url: str) -> Path | None:
    """Best-effort extraction of a filesystem path from a sqlite URL."""
    if url.startswith("sqlite:///"):
        rest = url[len("sqlite:///"):]
        if rest == ":memory:":
            return None
        return Path(rest)
    return None


# ---- init -----------------------------------------------------------------


def _read_template(*parts: str) -> str:
    """Read a template file shipped under `wallow.templates`."""
    pkg = resources.files("wallow.templates")
    for p in parts:
        pkg = pkg.joinpath(p)
    return pkg.read_text()


def _cmd_init(args: argparse.Namespace) -> int:
    target_dir = Path(args.dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    schema_path = target_dir / args.schema
    ini_path = target_dir / "alembic.ini"
    alembic_dir = target_dir / "alembic"
    env_path = alembic_dir / "env.py"
    mako_path = alembic_dir / "script.py.mako"
    versions_dir = alembic_dir / "versions"
    snapshots_dir = alembic_dir / "snapshots"

    existing = [
        p for p in (schema_path, ini_path, env_path, mako_path) if p.exists()
    ]
    if existing and not args.force:
        _err(
            "refusing to overwrite existing files: "
            + ", ".join(str(p.relative_to(target_dir)) for p in existing)
            + " (pass --force to overwrite)"
        )
        return EXIT_FAIL

    alembic_dir.mkdir(parents=True, exist_ok=True)
    versions_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    project_name = target_dir.name or "my_project"
    schema_path.write_text(
        _read_template("wallow.toml.template").format(project_name=project_name)
    )
    ini_path.write_text(
        _read_template("alembic.ini.template").format(db_path=args.db)
    )
    env_path.write_text(_read_template("alembic", "env.py.template"))
    mako_path.write_text(_read_template("alembic", "script.py.mako"))

    rel = target_dir
    print(f"initialised wallow project in {rel}/")
    print(f"  schema: {schema_path.relative_to(rel)}")
    print(f"  config: {ini_path.relative_to(rel)}")
    print(f"  alembic dir: {alembic_dir.relative_to(rel)}/")
    print()
    print("next steps:")
    print("  1. edit wallow.toml to declare your fields")
    print('  2. wallow migrate generate "initial schema"')
    print("  3. wallow migrate apply")
    return EXIT_OK


# ---- migrate --------------------------------------------------------------


def _cmd_migrate_generate(args: argparse.Namespace) -> int:
    ini = _resolve_alembic_ini(args.alembic_ini)
    schema_path = (
        Path(args.schema).resolve() if args.schema else _schema_path_from_ini(ini)
    )
    cfg = migrations._make_config(ini)
    rev_path = migrations.generate(
        cfg, message=args.message, schema_path=schema_path
    )
    print(f"wrote revision: {rev_path}")
    print("review the file before applying with `wallow migrate apply`.")
    return EXIT_OK


def _cmd_migrate_apply(args: argparse.Namespace) -> int:
    ini = _resolve_alembic_ini(args.alembic_ini)
    cfg = migrations._make_config(ini)
    target = args.target or "head"
    migrations.apply(cfg, target=target)
    print(f"applied migrations to {target}")
    return EXIT_OK


def _cmd_migrate_downgrade(args: argparse.Namespace) -> int:
    if args.target == "base" and not args.yes:
        _err("downgrading past `base` requires --yes")
        return EXIT_FAIL
    ini = _resolve_alembic_ini(args.alembic_ini)
    cfg = migrations._make_config(ini)
    migrations.downgrade(cfg, target=args.target)
    print(f"downgraded to {args.target}")
    return EXIT_OK


def _cmd_migrate_history(args: argparse.Namespace) -> int:
    from sqlalchemy import create_engine

    ini = _resolve_alembic_ini(args.alembic_ini)
    cfg = migrations._make_config(ini)
    revisions = migrations.history(cfg)

    db_url = _db_url_from_ini(ini)
    cur: str | None = None
    try:
        engine = create_engine(db_url, future=True)
        cur = migrations.current_revision(engine)
    except Exception:
        cur = None

    if not revisions:
        print("(no revisions)")
        return EXIT_OK
    for r in revisions:
        marker = "*" if r.revision == cur else " "
        msg = (r.doc or "").strip().splitlines()[0] if r.doc else ""
        print(f"{marker} {r.revision}  {msg}")
    print()
    print("(* = currently applied)")
    return EXIT_OK


def _cmd_migrate_stamp(args: argparse.Namespace) -> int:
    ini = _resolve_alembic_ini(args.alembic_ini)
    cfg = migrations._make_config(ini)
    migrations.stamp(cfg, revision=args.revision)
    print(f"stamped database at revision {args.revision}")
    return EXIT_OK


# ---- status ---------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    from sqlalchemy import create_engine

    try:
        ini = _resolve_alembic_ini(args.alembic_ini)
    except WallowError as e:
        _err(str(e))
        return EXIT_FAIL

    schema_path = (
        Path(args.schema).resolve() if args.schema else _schema_path_from_ini(ini)
    )
    schema = load_schema(schema_path)
    db_url = args.db_url or _db_url_from_ini(ini)
    engine = create_engine(db_url, future=True)

    cfg = migrations._make_config(ini, db_url=db_url)
    head = migrations.head_revision(cfg)
    cur = migrations.current_revision(engine)
    pending = cur != head

    # Run count: only meaningful if the runs table exists.
    run_count: int | str
    try:
        with engine.connect() as conn:
            from sqlalchemy import text

            run_count = conn.execute(text("SELECT COUNT(*) FROM runs")).scalar()
    except Exception:
        run_count = "n/a"

    print(f"schema:        {schema_path}")
    print(f"project:       {schema.project_name}")
    print(f"database:      {db_url}")
    print(f"current rev:   {cur or '<none>'}")
    print(f"head rev:      {head or '<none>'}")
    print(f"pending:       {'yes' if pending else 'no'}")
    print(f"run count:     {run_count}")

    return EXIT_FAIL if pending else EXIT_OK


# ---- inspect --------------------------------------------------------------


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .store import Store

    try:
        ini = _resolve_alembic_ini(args.alembic_ini) if hasattr(args, "alembic_ini") else None
    except WallowError:
        ini = None

    if args.schema:
        schema_path = Path(args.schema).resolve()
    elif ini is not None:
        schema_path = _schema_path_from_ini(ini)
    else:
        schema_path = Path("wallow.toml").resolve()

    if args.db:
        db_path: Any = args.db
    elif ini is not None:
        db_url = _db_url_from_ini(ini)
        candidate = _db_path_from_url(db_url)
        if candidate is None:
            _err("cannot inspect: db is :memory:")
            return EXIT_FAIL
        db_path = candidate
    else:
        _err("--db is required when no alembic.ini is discoverable")
        return EXIT_FAIL

    schema = load_schema(schema_path)
    store = Store(db_path, schema=schema, check_schema=False)

    Run = schema.Run
    with store.session() as s:
        run = s.get(Run, args.id)
    if run is None:
        _err(f"no run with id={args.id}")
        return EXIT_FAIL

    print(f"id:           {run.id}")
    print(f"created_at:   {run.created_at}")
    print(f"updated_at:   {run.updated_at}")
    print()
    print("identifying:")
    for name in sorted(schema.identifying):
        print(f"  {name}: {getattr(run, name)!r}")
    print()
    print("annotating:")
    for name in sorted(schema.annotating):
        print(f"  {name}: {getattr(run, name)!r}")
    return EXIT_OK


# ---- argparse wiring ------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wallow",
        description="Deduplicating run registry for ML research.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="initialise a new wallow project")
    p_init.add_argument("--force", action="store_true",
                        help="overwrite existing files")
    p_init.add_argument("--db", default="runs.db",
                        help="path to the SQLite db file (default: runs.db)")
    p_init.add_argument("--schema", default="wallow.toml",
                        help="path to the wallow.toml file (default: wallow.toml)")
    p_init.add_argument("--dir", default=".",
                        help="target directory (default: cwd)")
    p_init.set_defaults(func=_cmd_init)

    # migrate
    p_migrate = sub.add_parser("migrate", help="schema migration commands")
    msub = p_migrate.add_subparsers(dest="migrate_command", required=True)

    p_gen = msub.add_parser("generate", help="autogenerate a revision")
    p_gen.add_argument("message")
    p_gen.add_argument("--schema", default=None)
    p_gen.add_argument("--alembic-ini", default=None)
    p_gen.set_defaults(func=_cmd_migrate_generate)

    p_app = msub.add_parser("apply", help="apply pending migrations")
    p_app.add_argument("--target", default=None,
                       help="revision to apply to (default: head)")
    p_app.add_argument("--alembic-ini", default=None)
    p_app.set_defaults(func=_cmd_migrate_apply)

    p_dn = msub.add_parser("downgrade", help="downgrade to a revision")
    p_dn.add_argument("target")
    p_dn.add_argument("--yes", action="store_true",
                      help="confirm downgrade past `base`")
    p_dn.add_argument("--alembic-ini", default=None)
    p_dn.set_defaults(func=_cmd_migrate_downgrade)

    p_hist = msub.add_parser("history", help="list revisions")
    p_hist.add_argument("--alembic-ini", default=None)
    p_hist.set_defaults(func=_cmd_migrate_history)

    p_stamp = msub.add_parser(
        "stamp", help="record a revision without applying any DDL"
    )
    p_stamp.add_argument("revision",
                         help="revision id, or 'head'")
    p_stamp.add_argument("--alembic-ini", default=None)
    p_stamp.set_defaults(func=_cmd_migrate_stamp)

    # status
    p_status = sub.add_parser("status", help="report schema/db sync state")
    p_status.add_argument("--schema", default=None)
    p_status.add_argument("--db", dest="db_url", default=None,
                          help="SQLAlchemy URL (default: from alembic.ini)")
    p_status.add_argument("--alembic-ini", default=None)
    p_status.set_defaults(func=_cmd_status)

    # inspect
    p_ins = sub.add_parser("inspect", help="pretty-print a single run")
    p_ins.add_argument("id", type=int)
    p_ins.add_argument("--db", default=None,
                       help="path to the SQLite db file (default: from alembic.ini)")
    p_ins.add_argument("--schema", default=None)
    p_ins.add_argument("--alembic-ini", default=None)
    p_ins.set_defaults(func=_cmd_inspect)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WallowError as e:
        _err(str(e))
        return EXIT_FAIL


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
