# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`wallow` is a deduplicating run registry for ML experiments. The authoritative spec is `specs/wallow_spec.md` (~1140 lines) — consult it when adding features or resolving ambiguity. §15 ("Implementation notes (as-built)") records every decision resolved during build; treat it as the diff between original spec and shipped code. The library targets a single SQLite file by default; the abstraction is meant to support Postgres but isn't tested there.

**Status**: Phases 1–4 are feature-complete and tests are green. Phase 5 (README polish, packaging, `examples/matching_feedback/` directory) is still pending per spec §15 — don't treat their absence as missing-by-oversight.

## Commands

```bash
pip install -e .[test]               # editable install with test deps
pytest -q                            # full suite (~129 tests)
pytest tests/test_dsl.py -q          # one file
pytest tests/test_store.py::test_register_basic -q   # one test
pytest --cov=wallow --cov-report=term-missing        # coverage
```

CLI smoke-test in a scratch dir: `wallow init --db runs.db && wallow migrate generate "init" && wallow migrate apply && wallow status`.

## Architecture

The five source modules form a layered pipeline. Read them in this order:

1. **`schema.py`** — parses `wallow.toml` into a `Schema`, then dynamically synthesises a fresh `DeclarativeBase` + `Run` SQLAlchemy class per Schema. Each Schema gets its own Base so multiple schemas don't collide on `__tablename__ = "runs"`. The `Run` class carries a `__wallow_schema__` back-reference that the DSL uses to resolve field names at compile time. Identifying fields with a `default` get a DDL-level `server_default` so Alembic can backfill existing rows when the field is added later.

2. **`store.py`** — `Store` owns the engine/session and exposes `register()` / `find()` / DSL entry points. Two operating modes determined at init: if `alembic_version` table exists the Store defers DDL to migrations and `check_schema()` compares current revision to head; otherwise it falls back to `Base.metadata.create_all` and skips the check. SQLite pragmas (`WAL`, `synchronous=NORMAL`, `foreign_keys=ON`) are installed via a `connect` event listener.

3. **`dsl.py`** — `F("name") == 4` builds a frozen AST (`_Compare`, `_And`, etc.) which compiles to SQLAlchemy `ColumnElement` only when the `Query` materialises. Field name resolution is deferred to compile time so the same expression can be reused across schemas. `Field` overrides `__eq__`/`__ne__` to return `Expr`, so `__hash__ = None` is set to prevent accidental dict/set use.

4. **`migrations.py`** — thin wrappers around `alembic.command` plus two wallow-specific pieces:
   - **Snapshot mechanism**: every `migrate generate` copies the current `wallow.toml` to `alembic/snapshots/{rev}.toml` (with a header marking it auto-generated; `#`-prefixed lines are TOML comments).
   - **Pre-flight diff**: before invoking Alembic autogenerate, compares new schema vs head snapshot and aborts if (a) an identifying field is being dropped, or (b) a new identifying field has no `default`. The first case would break dedup; the second produces a NOT NULL column that can't be added to a non-empty table.
   - `find_collisions_after_drop()` is the recommended manual escape: GROUP BY remaining identifying columns HAVING count > 1, then return `(field_values, row_ids)` for each colliding group.

5. **`cli.py`** — `argparse`-based, thin. Commands locate config by walking up from cwd looking for `alembic.ini` (or `--alembic-ini`). The ini stores `wallow_schema = wallow.toml` and a `sqlalchemy.url` that gets resolved relative to the ini file's directory (so projects are portable across cwds). `_resolve_sqlite_url` in `migrations.py` is the single source of truth for that resolution; `cli.py` and the env.py template both call it.

## Critical conventions

- **`on_duplicate` has no default.** `register()` requires the caller to choose explicitly: `"raise" | "return_existing" | "overwrite" | "skip"`. This is intentional per spec.
- **Identifying fields are restricted to `int`/`float`/`string`/`bool`** (the `_IDENTIFYING_ALLOWED` set). All other types (`json`, `datetime`, `path`) are annotating-only.
- **Strict bool/int distinction.** Python's `bool` is a subclass of `int`, but the schema treats them as distinct — uses `type(x) is int` rather than `isinstance(x, int)`. Mirror this when adding type checks in `schema.py` and `dsl.py`.
- **NaN identifying floats are rejected** (NaN ≠ NaN breaks dedup). Naive datetimes are rejected too (ambiguous across machines).
- **Float identity is by IEEE 754, not normalised.** A run with `x=0.1+0.2` and one with `x=0.3` dedupe as distinct. Spec §9 says don't try to fix this — document it for users instead.
- **Reserved field names**: `id`, `created_at`, `updated_at`, plus anything matching `^_wallow_` (case-insensitive).
- **Schema isolation.** Each `Schema` instance creates its own `DeclarativeBase` subclass — never share metadata across Schemas. Two `load_schema()` calls produce two `Run` classes that aren't `is`-equal (spec §13.1); downstream callers should hold a single `Schema` reference.
- **Alembic uses `render_as_batch=True`** in the env.py template because SQLite can't ALTER TABLE to drop/modify constraints; Alembic emulates via copy-and-rename.
- **`doc`-only TOML changes don't generate a migration** (spec §8.1). Alembic doesn't see `doc` because it's not a column attribute. This is expected — don't try to "fix" it.
- **JSON blobs >1 MB**: spec §9 recommends warning at insert and steering users to `path` fields. Not currently enforced; if you add the warning, keep it a warning, not a rejection.

## Templates

`src/wallow/templates/` ships files that `wallow init` materialises via `importlib.resources` (so editable installs, wheels, and zipped wheels all work). When changing template content, also update the `pyproject.toml` `[tool.setuptools.package-data]` glob if you add new file types.

## Testing gotchas

- **Concurrent-register tests need a WAL bootstrap** (spec §15 known limitations). If the parent test process forks workers without first opening a `Store`, the SQLite DB is still in rollback-journal mode and two writer subprocesses can deadlock. The fix in `tests/test_migrations.py` is to construct a parent `Store` once before forking — `Store._install_pragmas` flips `journal_mode=WAL` on its first connection. Mirror this pattern in any new multi-process test.
- **In-memory stores** (`Store(":memory:", ...)`) skip the WAL pragma (no-op on `:memory:`) and bypass the Alembic check by passing `check_schema=False` in fixtures (`tests/conftest.py`). Use `memory_store` for fast unit tests, `file_store` (uses `tmp_path`) when you need persistence or migration interaction.
