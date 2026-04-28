# wallow

A deduplicating run registry for ML research, with TOML schemas, an expression DSL, and Alembic migrations.

`wallow` solves one problem: deduplicating ML experiment runs by their identifying hyperparameters, with explicit support for schema evolution. It's built on SQLAlchemy 2.x and Alembic; the default backend is a single SQLite file.

The full specification lives in [`specs/wallow_spec.md`](specs/wallow_spec.md).

## Install

```bash
pip install -e .            # editable install from a clone
pip install -e .[test]      # with pytest + pytest-cov for the test suite
```

Requires Python 3.10+.

## Quickstart

```bash
mkdir my_project && cd my_project
wallow init --db runs.db
```

This creates `wallow.toml` (schema), `alembic.ini` (migration config), `alembic/env.py` (Alembic environment wired to your TOML), and the `alembic/versions/` and `alembic/snapshots/` directories.

Edit `wallow.toml` to declare your fields:

```toml
[project]
name = "matching_feedback"

[identifying.cell_k]
type = "int"

[identifying.cell_sigma]
type = "float"

[identifying.seed]
type = "int"
default = 0

[annotating.status]
type = "string"
indexed = true

[annotating.val_loss]
type = "float"
indexed = true
```

Then generate and apply the first migration:

```bash
wallow migrate generate "initial schema"   # writes alembic/versions/<rev>_initial_schema.py
wallow migrate apply                       # creates the runs table + alembic_version
wallow status                              # exits 0 when in sync
```

## Registering runs

```python
from wallow import Store, load_schema, register

schema = load_schema("wallow.toml")
store = Store("runs.db", schema=schema)

run = register(
    store,
    identifying={"cell_k": 4, "cell_sigma": 0.1, "seed": 0},
    annotating={"status": "running"},
    on_duplicate="return_existing",   # required: "raise" | "return_existing" | "overwrite" | "skip"
)
```

`on_duplicate` has no default — every caller picks the dedup policy explicitly.

## Querying with the DSL

```python
from wallow import F

best = (
    store.where((F("cell_k") == 4) & (F("status") == "completed"))
         .order_by(F("val_loss").asc())
         .limit(10)
         .all()
)
```

Composition operators `&`, `|`, `~` need parentheses around comparisons:
`(F("k") == 4) & (F("v") > 0.85)`. Field names resolve at compile time;
unknown names raise `SchemaValidationError` with a list of valid fields.

For escape-hatch queries, `store.execute(stmt)` runs raw SQLAlchemy and `store.engine` exposes the engine.

## Evolving the schema

Edit `wallow.toml`, then:

```bash
wallow migrate generate "add warmup_steps"   # autogenerate + snapshot
# (review the generated file in alembic/versions/)
wallow migrate apply
```

`wallow` aborts the generate step before invoking Alembic if it detects:

- An identifying field being **dropped** (would cause silent dedup collisions). Use `wallow.find_collisions_after_drop(store, "<field>")` to inspect, then resolve manually.
- A new identifying field added without `default` (NOT NULL columns can't be added to a non-empty table without a default).

The `default` for an identifying field is rendered as both Python-side `default` and DDL-level `server_default` so existing rows are backfilled when the field is added in a later migration.

## Adopting `wallow` on an existing database

If your project pre-dates the migration setup, your `runs` table was likely created via SQLAlchemy's `create_all` with no `alembic_version`. Adoption:

```bash
wallow init                         # writes alembic.ini + templates
wallow migrate generate "baseline"  # autogen against the existing DB → empty migration
wallow migrate stamp head           # records the revision without DDL
```

After this, edits to `wallow.toml` flow through the normal `generate` + `apply` cycle.

## CLI reference

| Command | Description |
|---|---|
| `wallow init [--force] [--dir DIR] [--db DB] [--schema PATH]` | Scaffold a new project. |
| `wallow migrate generate <message>` | Autogenerate a revision + snapshot. |
| `wallow migrate apply [--target REV]` | Apply pending migrations. |
| `wallow migrate downgrade <target> [--yes]` | Downgrade to a revision. `--yes` required for `base`. |
| `wallow migrate history` | List revisions; the applied one is marked `*`. |
| `wallow migrate stamp <revision>` | Record a revision in `alembic_version` without running DDL. |
| `wallow status` | Print sync state. Exit 0 when in sync, 1 when pending or no `alembic.ini` found. |
| `wallow inspect <id>` | Pretty-print one run's fields. |

Every `migrate`/`status`/`inspect` command accepts `--alembic-ini PATH` for explicit config; otherwise the CLI walks up from cwd looking for `alembic.ini`.

## Tests

```bash
pytest -q     # 129 passing as of the last commit
```

## Layout

```
src/wallow/
  schema.py        # TOML parser + dynamic SQLAlchemy model generation
  store.py         # Store, register, find, session management
  dsl.py           # F, Field, Expr, Query — operator-overloaded query builder
  migrations.py    # Alembic wrappers + snapshot mechanism + collision detection
  cli.py           # `wallow` command (argparse)
  errors.py        # WallowError hierarchy
  templates/       # files copied by `wallow init`
tests/             # ~129 tests covering all phases
specs/wallow_spec.md  # authoritative specification
```
