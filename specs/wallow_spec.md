# `wallow` — implementation specification

*A deduplicating run registry for ML research, with TOML schemas, an expression DSL, and Alembic migrations.*

This document is the complete specification for an implementer (human or coding agent). It supersedes prior drafts. The implementer should be able to build the project from this alone; ambiguities should be resolved in favour of the simpler interpretation, and resolved decisions noted in code comments where they affect future maintenance.

---

## 1. Overview

`wallow` is a small Python library that solves one problem: deduplicating ML experiment runs by their identifying hyperparameters, with explicit support for schema evolution. It is built on SQLAlchemy 2.x and Alembic.

### Core ideas

**Two-tier field model.** Every run has *identifying* fields (which collectively define dedup) and *annotating* fields (which are queryable but don't affect dedup). This split is declared in TOML and enforced by a composite UNIQUE constraint on the identifying columns at the database level.

**TOML as schema source of truth.** The schema lives in `wallow.toml`. The library reads it at runtime and dynamically generates the SQLAlchemy model class. Users do not write SQLAlchemy models by hand.

**Composite unique constraint, no content hash.** Dedup is enforced by `UniqueConstraint(*identifying_field_names)` on the runs table. There is no separate hash column. The database's B-tree index on the constraint provides O(log n) dedup checks, which is effectively O(1) at the scales `wallow` targets.

**Alembic for migrations.** Schema changes go through Alembic. `wallow` provides thin CLI wrappers around `alembic.command` plus a TOML-snapshot mechanism so each migration is self-describing.

**Expression DSL for queries.** A small `F`-based DSL with operator overloading, used by downstream consumers (plotting utilities, analysis notebooks). Compiles to SQLAlchemy expressions under the hood. Raw SQLAlchemy remains available as an escape hatch.

### Goals

- Per-call dedup of runs with explicit duplicate-handling policy (`raise` | `return_existing` | `overwrite` | `skip`).
- TOML-declared schema with field types, indexing, defaults, and documentation.
- Expression DSL usable by downstream tools, with field references resolvable from string names.
- CLI for project initialization and migration management (`wallow init`, `wallow migrate generate/apply`, `wallow status`).
- Single SQLite file by default, no infrastructure.
- Programmatic API for downstream tools that's stable across schema migrations.

### Non-goals

- Multi-writer / distributed coordination beyond what SQLite WAL provides.
- Streaming or time-series metrics (use a JSON-typed annotating field if needed).
- Artefact storage (paths only; users manage their own files).
- Web UI.
- Postgres backend in v1 (the abstraction supports it; just not tested).
- Custom user-defined types.

---

## 2. Package layout

```
wallow/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── wallow/
│       ├── __init__.py            # public API re-exports
│       ├── schema.py              # TOML parser + dynamic model generation
│       ├── store.py               # Store class, register, find
│       ├── dsl.py                 # F, Expr, operator overloading, compilation
│       ├── migrations.py          # Alembic wrapper, snapshot mechanism
│       ├── cli.py                 # argparse entry points (`wallow` command)
│       ├── errors.py              # exception hierarchy
│       └── templates/             # files copied by `wallow init`
│           ├── wallow.toml.template
│           ├── alembic.ini.template
│           └── alembic/
│               ├── env.py.template
│               └── script.py.mako
└── tests/
    ├── test_schema.py
    ├── test_store.py
    ├── test_dsl.py
    ├── test_migrations.py
    ├── test_cli.py
    └── fixtures/
        └── example_wallow.toml
```

### Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "tomli>=2.0; python_version < '3.11'",
]

[project.scripts]
wallow = "wallow.cli:main"
```

No other runtime deps. Test deps: `pytest`, `pytest-cov`.

---

## 3. TOML schema format

The schema file is `wallow.toml` at the project root by default (location configurable). Format:

```toml
[project]
name = "matching_feedback"
description = "Meta-learning feedback plasticity rules across structural classes."
float_precision = 12   # optional; sig-figs used to normalise identifying-float values (default 12)

[identifying.cell_k]
type = "int"
doc = "Feature dimension of low-rank component"

[identifying.cell_sigma]
type = "float"
doc = "Residual standard deviation"

[identifying.generation]
type = "int"

[identifying.candidate_id]
type = "int"

[identifying.seed]
type = "int"
default = 0

[annotating.git_commit]
type = "string"
indexed = true

[annotating.host]
type = "string"
indexed = true

[annotating.status]
type = "string"
indexed = true
doc = "'running' | 'completed' | 'failed'"

[annotating.started_at]
type = "datetime"
indexed = true

[annotating.wall_clock_sec]
type = "float"

[annotating.val_loss]
type = "float"
indexed = true

[annotating.val_accuracy]
type = "float"
indexed = true

[annotating.artefacts_dir]
type = "path"

[annotating.structural_traj]
type = "json"

[annotating.discovered_T]
type = "json"
```

### Field declaration grammar

Each `[identifying.<name>]` or `[annotating.<name>]` table accepts:

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `type` | string | yes | — | One of: `int`, `float`, `string`, `bool`, `json`, `path`, `datetime` |
| `doc` | string | no | `None` | Surfaced in CLI help and migration generation |
| `indexed` | bool | no | `true` for identifying, `false` for annotating | Creates a B-tree index on the column |
| `default` | scalar | no | `None` | Default value at insert; for identifying fields, also used by autogen as `server_default` for migration backfill |
| `nullable` | bool | no | `false` for identifying, `true` for annotating | Whether the column can be NULL |

### Type catalogue

| Type | SQLAlchemy | Python | Allowed identifying? |
|---|---|---|---|
| `int` | `Integer` | `int` | yes |
| `float` | `Float` | `float` | yes |
| `string` | `String` | `str` | yes |
| `bool` | `Boolean` | `bool` | yes |
| `json` | `JSON` | any JSON-serialisable | **no** |
| `path` | `String` | `str` (filesystem path) | **no** |
| `datetime` | `DateTime` | `datetime.datetime` | **no** |

Identifying fields must be hashable scalars; the four allowed types reflect this. Attempting to declare a `json`, `path`, or `datetime` field as identifying must raise `SchemaParseError` at TOML load time.

### Reserved field names

The implementer must reject any user-declared field with name `id`, `created_at`, `updated_at`, or `_wallow_*`. These are managed by `wallow` and appear automatically on every model.

### Validation rules

The TOML loader must:

1. Parse the file using `tomllib` (Python 3.11+) or `tomli` (3.10).
2. Require exactly one `[project]` table with at least `name` (string). Accept optional `description` (string) and `float_precision` (positive int, default 12). Reject any other key inside `[project]`.
3. Require at least one identifying field. Schemas with zero identifying fields raise `SchemaParseError`.
4. Validate every field's `type` is in the catalogue.
5. Reject identifying fields with disallowed types.
6. Reject duplicate field names across `[identifying]` and `[annotating]`.
7. Reject reserved names.
8. Validate that `default` values, when present, are coercible to the declared type.

---

## 4. Public API

All types listed below are re-exported from `wallow.__init__`.

### 4.1 Schema loading

```python
def load_schema(path: str | Path) -> Schema:
    """Parse a wallow.toml file and return a Schema with a generated model."""

class Schema:
    project_name: str
    fields: dict[str, FieldDecl]            # all fields, name -> FieldDecl
    identifying: frozenset[str]
    annotating: frozenset[str]
    Run: type                                # the dynamically generated model class
    Base: type                               # the SQLAlchemy declarative base

    def field(self, name: str) -> FieldDecl: ...
    def __iter__(self) -> Iterator[FieldDecl]: ...

@dataclass(frozen=True)
class FieldDecl:
    name: str
    type: str                                # one of the type catalogue strings
    is_identifying: bool
    indexed: bool
    nullable: bool
    default: Any | None
    doc: str | None
```

The `Schema.Run` class is generated dynamically — see §6 for the model-generation algorithm. Downstream code may use `Run` directly with SQLAlchemy:

```python
schema = load_schema("wallow.toml")
Run = schema.Run
session.scalars(select(Run).where(Run.cell_k == 4))
```

### 4.2 Store

```python
class Store:
    def __init__(
        self,
        db_path: str | Path,
        *,
        schema: Schema,
        check_schema: bool = True,
    ) -> None:
        """Open or create a SQLite store backed by `db_path`.

        If `check_schema=True` (default), raises PendingMigrationError when
        the database revision is behind the schema's head revision.
        WAL journal mode is enabled automatically.
        """

    @property
    def schema(self) -> Schema: ...
    @property
    def engine(self) -> Engine: ...

    def session(self) -> ContextManager[Session]:
        """Context manager yielding a SQLAlchemy session.

        Commits on clean exit, rolls back on exception.
        """

    def execute(self, statement: Executable) -> Result:
        """Execute a raw SQLAlchemy statement. Escape hatch for power users."""

    def where(self, *exprs: Expr) -> Query:
        """Start a DSL query. Multiple exprs are AND-combined."""

    def all(self) -> list[Run]:
        """Equivalent to store.where().all() — returns all runs."""

    def count(self) -> int:
        """Equivalent to store.where().count()."""

    def check_schema(self) -> None:
        """Raise PendingMigrationError if DB schema is behind code schema."""

    def migrate(self) -> None:
        """Apply all pending migrations. Equivalent to `wallow migrate apply`."""
```

### 4.3 register()

```python
@dataclass(frozen=True)
class RegisterResult:
    run: Run | None        # the row (None only for "skip" on duplicate)
    was_inserted: bool     # True iff this call inserted a new row
    was_updated: bool      # True iff this call wrote annotating fields to an existing row
    was_skipped: bool      # True iff an existing row was returned without modification

def register(
    store: Store,
    *,
    identifying: dict[str, Any],
    annotating: dict[str, Any] | None = None,
    on_duplicate: Literal[
        "raise", "return_existing", "overwrite", "skip", "claim_if_stale"
    ],
    stale_after: datetime.timedelta | None = None,
) -> RegisterResult:
    """Register a run.

    Args:
        store: a Store.
        identifying: keys must match store.schema.identifying, except that any
            field with a TOML `default` may be omitted (the default is filled
            in before validation and dedup).
        annotating: subset of store.schema.annotating; missing fields are NULL.
        on_duplicate: required (no default).
        stale_after: required when on_duplicate='claim_if_stale'; ignored otherwise.

    Returns:
        RegisterResult. Exactly one of was_inserted / was_updated / was_skipped is
        True for every outcome except `return_existing` on a duplicate, where all
        three are False.

        - 'raise':           was_inserted=True; raises DuplicateRunError on duplicate.
        - 'return_existing': was_inserted=True (new) or all-False (existing returned).
        - 'overwrite':       was_inserted=True (new) or was_updated=True (existing modified).
        - 'skip':            was_inserted=True (new) or run=None + was_skipped=True (duplicate).
        - 'claim_if_stale':  was_inserted=True (new); was_updated=True (existing was stale
                             — annotating fields written and updated_at bumped); was_skipped=
                             True (existing was fresh — returned untouched).

    Identifying float values are normalised to schema.float_precision sig figs
    before insertion / lookup so IEEE-754 mantissa noise (0.1+0.2 vs 0.3) doesn't
    split dedup groups.

    Raises:
        SchemaValidationError: required identifying keys missing (no default), or
            unknown / wrong-typed values supplied.
        ValueError: on_duplicate='claim_if_stale' without a timedelta `stale_after`.
        DuplicateRunError: only when on_duplicate='raise'.
    """
```

The `on_duplicate` parameter is required (no default). This forces every caller to think about the semantics.

**`claim_if_stale` algorithm.** On a UNIQUE collision, read the existing row's `updated_at` (auto-populated on insert and bumped on every UPDATE). If `now() - updated_at > stale_after`, treat as `overwrite` (write the annotating fields and bump `updated_at` unconditionally so the heartbeat clock restarts) and return `was_updated=True`. Otherwise return the existing row untouched with `was_skipped=True`. SQLAlchemy's default DateTime column on SQLite drops tzinfo on read; the implementation must coerce a naive `updated_at` back to UTC before subtracting from a tz-aware `now()`.

### 4.4 find()

```python
def find(store: Store, **identifying: Any) -> Run | None:
    """Direct identifying-fields lookup.

    Args:
        store: a Store.
        **identifying: keys must match store.schema.identifying, except that any
            field with a TOML `default` may be omitted. Float values are normalised
            to schema.float_precision sig figs (same as register).

    Returns:
        The Run with matching identifying fields, or None.

    Raises:
        SchemaValidationError: required identifying keys missing (no default), or
            unknown / wrong-typed values supplied.
    """
```

### 4.4b heartbeat()

```python
def heartbeat(store: Store, *, identifying: dict[str, Any]) -> datetime.datetime:
    """Bump `updated_at` for the run with this identifying tuple.

    Pairs with on_duplicate='claim_if_stale' for live multi-worker dispatch:
    a worker calls heartbeat periodically while training so other workers see
    the row as fresh and don't claim it. Returns the new updated_at (tz-aware UTC).

    Identifying defaults are filled and floats normalised the same way as
    register / find.

    Raises:
        SchemaValidationError: no row matches the identifying tuple.
    """
```

### 4.5 DSL: `F`, `Field`, `Expr`

The DSL is the recommended way to build queries from downstream code, especially when field names come from configuration rather than as Python identifiers.

```python
def F(name: str, schema: Schema | None = None) -> Field:
    """Create a field reference.

    By default, name resolution is deferred until compile time so the same
    expression can be reused across schemas. Pass `schema=...` to validate
    the name eagerly; an unknown name raises SchemaValidationError at the
    F() callsite. The schema-bound shortcut `schema.f.<name>` performs the
    same eager check via attribute access (raises AttributeError on typo)
    and supports `dir(schema.f)` for IDE autocomplete.
    """

class Field:
    """A named field reference. Supports comparison and JSON path operators."""

    name: str

    # Comparison operators — return Expr.
    def __eq__(self, other: Any) -> Expr: ...
    def __ne__(self, other: Any) -> Expr: ...
    def __lt__(self, other: Any) -> Expr: ...
    def __le__(self, other: Any) -> Expr: ...
    def __gt__(self, other: Any) -> Expr: ...
    def __ge__(self, other: Any) -> Expr: ...

    # Set membership.
    def in_(self, values: Iterable[Any]) -> Expr: ...
    def not_in(self, values: Iterable[Any]) -> Expr: ...

    # String operators (only valid on string-typed fields; checked at compile).
    def contains(self, substr: str) -> Expr: ...
    def startswith(self, prefix: str) -> Expr: ...
    def endswith(self, suffix: str) -> Expr: ...

    # Null checks.
    def is_null(self) -> Expr: ...
    def is_not_null(self) -> Expr: ...

    # JSON path (only valid on json-typed fields; checked at compile).
    def json_path(self, path: str) -> Field:
        """Returns a new Field representing the JSON path inside this field.
        The returned Field supports comparison operators normally."""

    # Ordering.
    def asc(self) -> OrderClause: ...
    def desc(self) -> OrderClause: ...

class Expr:
    """A boolean expression. Composable via &, |, ~."""

    def __and__(self, other: Expr) -> Expr: ...
    def __or__(self, other: Expr) -> Expr: ...
    def __invert__(self) -> Expr: ...

    def compile(self, model_class: type) -> ColumnElement:
        """Compile to a SQLAlchemy expression. For internal use by Query."""

class Query:
    """A pending query. Lazily executed."""

    def where(self, *exprs: Expr) -> Query: ...      # AND-combined
    def order_by(self, *clauses: OrderClause | Field) -> Query: ...
    def limit(self, n: int) -> Query: ...
    def offset(self, n: int) -> Query: ...

    def all(self) -> list[Run]: ...
    def first(self) -> Run | None: ...
    def one(self) -> Run: ...                        # raises if 0 or >1 results
    def count(self) -> int: ...
    def exists(self) -> bool: ...
    def __iter__(self) -> Iterator[Run]: ...         # streaming
```

#### DSL semantics

**Operator precedence.** Python's `&`/`|` have lower precedence than comparison operators only when the comparisons are inside parentheses. Users must write `(F("k") == 4) & (F("v") > 0.85)`. This is the standard pandas/SQLAlchemy convention; document it.

**Field resolution.** `F("name")` (no schema) does not validate the name immediately. Resolution happens when an `Expr` is compiled against a model class (during query execution). If `name` is not a field, `compile()` raises `SchemaValidationError` with a helpful message listing valid field names. `F("name", schema=...)` and `schema.f.name` both validate eagerly at the callsite — preferred for code that already has a Schema in hand.

**Type coercion.** When the right-hand side of a comparison can't be coerced to the field's type, raise `SchemaValidationError` at compile time. Special cases: comparing a `bool` field to `0` or `1` is allowed (coerce); comparing a `string` field to `None` is allowed and treated as `IS NULL` / `IS NOT NULL`. Comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`, `in_`, `not_in`) against an *identifying* float field have their RHS normalised to `schema.float_precision` sig figs (same rule as `register` / `find`), so `F("lr") == 0.1 + 0.2` matches a row stored at `lr = 0.3`. JSON-path comparisons are left untouched (the underlying type is opaque). Range queries against annotating floats are not normalised; they preserve full precision.

**JSON paths.** `F("discovered_T").json_path("T_1100") > 0.05` compiles to `func.json_extract(Run.discovered_T, '$.T_1100') > 0.05`. Nested paths use dot notation: `F("x").json_path("a.b.c")` → `'$.a.b.c'`. JSON paths are only valid on `json`-typed fields; checked at compile.

**String operators on non-string fields.** Compile-time error.

### 4.6 Errors

```python
class WallowError(Exception): ...

class SchemaError(WallowError): ...
class SchemaParseError(SchemaError): ...
class SchemaValidationError(SchemaError): ...
class PendingMigrationError(SchemaError):
    current_rev: str | None
    head_rev: str | None

class DuplicateRunError(WallowError):
    existing: Run
```

All errors carry structured context (field names, expected types, missing keys) as attributes; users may parse the exception programmatically rather than parsing the message.

---

## 5. CLI

The `wallow` command is the entry point. Implemented with `argparse`.

### 5.1 `wallow init`

```
wallow init [--force] [--db DB_PATH] [--schema SCHEMA_PATH] [--dir DIR]
```

Initializes a new project. Creates:

- `wallow.toml` — copied from `templates/wallow.toml.template`. Contains a `[project]` block and minimal `[identifying]` / `[annotating]` examples. The template substitutes the target directory's basename as `[project].name`.
- `alembic.ini` — copied from template, with `sqlalchemy.url` pointing at `--db` (default `runs.db`, written as a relative path), `script_location = alembic`, and `wallow_schema = wallow.toml` (the path env.py reads).
- `alembic/env.py` — copied from template. Imports the schema via `wallow.load_schema()` and exposes `Base.metadata` to Alembic. Sets `render_as_batch=True` (required for SQLite to drop/rebuild constraints).
- `alembic/script.py.mako` — copied verbatim.
- `alembic/versions/` — empty directory.
- `alembic/snapshots/` — empty directory (for TOML snapshots; see §8).

`--force` overwrites existing files (refuses by default if any exist). `--dir` writes into a directory other than cwd (defaults to `.`); useful for scripting and tests.

### 5.2 `wallow migrate generate <message>`

```
wallow migrate generate "add warmup steps" [--schema PATH] [--alembic-ini PATH]
```

1. Load the current `wallow.toml`.
2. Pre-flight: load the head revision's snapshot (`alembic/snapshots/{head}.toml`) and compare to the new schema. Abort with `WallowError` if (a) any identifying field is being dropped (see §8.2), or (b) any newly-added identifying field has no `default` in the TOML. The first migration (no head snapshot) skips this check.
3. Invoke `alembic.command.revision(config, message=<message>, autogenerate=True)`. Alembic compares the model (generated from current TOML) against the database state and produces a revision file in `alembic/versions/`.
4. After Alembic writes the revision file, copy the current `wallow.toml` to `alembic/snapshots/{revision_id}.toml` with a three-line auto-generated header (TOML treats `#` lines as comments).
5. Print the path to the generated revision file and remind the user to review it before applying.

The revision file is always reviewed by hand. `wallow` never auto-applies generated migrations.

### 5.3 `wallow migrate apply [--target REV]`

```
wallow migrate apply [--target <revision_id>] [--alembic-ini PATH]
```

Calls `alembic.command.upgrade(config, revision="head" or REV)`. Wraps Alembic exceptions in `WallowError` subclasses where appropriate; passes through unchanged otherwise.

### 5.4 `wallow migrate downgrade <revision>`

```
wallow migrate downgrade <revision> [--yes] [--alembic-ini PATH]
```

Calls `alembic.command.downgrade(config, revision)`. Required argument; refuses to downgrade to `base` without explicit confirmation flag `--yes`.

### 5.5 `wallow migrate history`

```
wallow migrate history [--alembic-ini PATH]
```

Lists revisions newest-first via `ScriptDirectory.walk_revisions()`, marking the currently-applied revision with `*`. Each line: `* <rev_id>  <slug>`.

### 5.5b `wallow migrate stamp <revision>`

```
wallow migrate stamp <revision> [--alembic-ini PATH]
```

*Added beyond the original spec.* Wraps `alembic.command.stamp`. Records a revision in `alembic_version` without running any DDL. Used when adopting Alembic on an existing Phase 1/2 database (run `wallow migrate stamp head` after the first `wallow init` so subsequent `migrate generate` calls see a populated history).

### 5.6 `wallow status`

```
wallow status [--schema PATH] [--db URL] [--alembic-ini PATH]
```

Reports:

- Schema file path and `[project] name`.
- Database URL (resolved against `alembic.ini`'s directory if relative).
- Current applied revision (or `<none>` if database is empty / no revisions).
- Schema head revision (or `<none>` if no migrations exist yet).
- Whether migrations are pending (`yes`/`no`).
- Run count in the database (`n/a` if the `runs` table doesn't exist yet).

Exit codes: **0** when in sync (current == head), **1** when pending OR when no `alembic.ini` is discoverable, **2** for argparse errors. Right after `wallow init` and before any `migrate generate`, both current and head are `<none>`, which counts as "in sync" and exits 0.

### 5.7 `wallow inspect <run_id>`

```
wallow inspect <id> [--db PATH] [--schema PATH] [--alembic-ini PATH]
```

Pretty-prints a single run's identifying and annotating fields. Opens a `Store(check_schema=False)` so it works on databases that are pending or stamped at an old revision. Exits 0 on success, 1 if the id is not found.

### 5.8 `wallow query` (deferred to v1.1)

Out of scope for v1; users use the Python API for ad-hoc queries.

### 5.9 `--alembic-ini` and config discovery

Every `migrate`/`status`/`inspect` subcommand accepts an optional `--alembic-ini PATH`. When omitted, the CLI walks up from cwd looking for `alembic.ini`. If still not found, commands that need it print a helpful error and exit 1. `Store.check_schema()` and `Store.migrate()` apply the same discovery rule, with an additional fallback to walking up from `db_path`'s parent.

---

## 6. Internal modules

### 6.1 `schema.py`

Responsibilities:

- Parse TOML files into `Schema` objects.
- Validate field declarations against the type catalogue and reserved-name list.
- Generate the SQLAlchemy declarative `Base` and `Run` classes dynamically.
- Provide `Schema.fill_identifying_defaults`, `Schema.normalise_identifying_value`, and the `Schema.f` attribute-access namespace used by the DSL for eager validation.

#### Dynamic model generation

The implementer constructs the `Run` class using `type()` (the three-argument form) or, equivalently, by calling `DeclarativeBase.__init_subclass__` machinery directly. Sketch:

```python
from sqlalchemy import Column, UniqueConstraint, Integer, DateTime
from sqlalchemy.orm import DeclarativeBase

def _build_model(schema: Schema) -> tuple[type, type]:
    class Base(DeclarativeBase):
        pass

    attrs: dict[str, Any] = {
        "__tablename__": "runs",
        "__wallow_schema__": schema,            # back-reference, not a column
        "_wallow_identifying": schema.identifying,
        "id": Column(Integer, primary_key=True, autoincrement=True),
        "created_at": Column(DateTime, nullable=False, default=_utcnow),
        "updated_at": Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow),
    }

    for f in schema.fields.values():
        col = Column(
            f.sa_type(),
            nullable=f.nullable,
            index=f.indexed,
            default=f.default,
        )
        attrs[f.name] = col

    attrs["__table_args__"] = (
        UniqueConstraint(*sorted(schema.identifying), name="uq_runs_identifying"),
    )

    Run = type("Run", (Base,), attrs)
    return Base, Run
```

The implementer must ensure the generated `Run` class registers correctly with the `Base.metadata` so Alembic's autogenerate sees it.

**Important:** The `UniqueConstraint` argument order is sorted alphabetically. This means changing identifying fields in the TOML doesn't change the generated constraint as long as the *set* of identifying fields is the same — sort order shouldn't be a source of spurious migration regenerations.

#### `server_default` for identifying fields with a default

When an identifying field declares a `default`, `_build_model` ALSO renders that default as `server_default` (a DDL-level `DEFAULT` clause). Without this, Alembic autogenerate would emit an `add_column` with `nullable=False` but no `DEFAULT`, which fails to apply on a non-empty table when the field is added in a later migration (spec §8.2).

A small helper `_server_default_literal(decl)` converts the default into a SQL literal string. Identifying fields are restricted to int/float/string/bool, so the conversion is well-defined:

| Field type | Conversion | Notes |
|---|---|---|
| `int` | `str(int(default))` | |
| `float` | `repr(default)` | `repr` preserves precision better than `str` |
| `string` | the literal string | Alembic quotes it when rendering DDL |
| `bool` | `"1"` for `True`, `"0"` for `False` | SQLite stores booleans as integers; `str(True)` would emit `"True"` which SQLite doesn't understand |

Annotating fields are not affected: they're nullable, so adding one to an existing table doesn't need backfill.

#### Type validation at register time

When `register()` is called, the implementer must validate values:

- For each identifying field, check that the supplied value is `isinstance` of the declared Python type. Raise `SchemaValidationError` on mismatch. Special case: `int` is acceptable for `float` fields; `bool` is *not* acceptable for `int` fields (Python's `bool <: int` would otherwise let `True` slip through).
- For annotating fields, the same rules apply but with `None` always permitted.
- For `json` fields, check that the value is JSON-serialisable (`json.dumps` succeeds); reject `set`, `bytes`, `datetime` etc. unless coerced upstream.
- For `datetime` fields, accept only `datetime.datetime`. Reject naive datetimes optionally — recommend rejecting them and requiring tz-aware values for cross-machine consistency.
- For `path` fields, accept any string; do not check existence (paths may refer to files on other machines).

NaN in identifying float fields must be rejected (raises `SchemaValidationError`). NaN's identity semantics break dedup.

#### Default-fill, float normalisation, and the field namespace

Three helpers on `Schema` support the v0.2 ergonomics:

- `fill_identifying_defaults(d)` returns a new dict with declared TOML defaults filled in for any missing identifying key. Called by `register`, `find`, and `heartbeat` before `validate_identifying_keys`. Fields without a `default` stay missing and the validator still raises.
- `_normalise_float(value, precision)` (module-level helper) rounds a float to `precision` significant figures via `round(value, -floor(log10(|value|)) + precision - 1)`. Zero, NaN, and infinity pass through unchanged (`math.isfinite` gate). `Schema.normalise_identifying_value(name, v)` calls this for identifying-float fields and returns `v` unchanged for everything else.
- `Schema.f` returns a `_FieldNamespace` that resolves attribute access to a DSL `Field` (lazily importing `wallow.dsl.Field` to avoid a module-load cycle). Unknown names raise `AttributeError` at the callsite. `dir(schema.f)` returns the sorted list of declared field names so IDE autocomplete works.

### 6.2 `store.py`

Responsibilities:

- Wrap engine creation, session management, and pragma settings (WAL, foreign keys).
- Implement `register()`, `find()`, and `heartbeat()`.
- Provide the `where()` entry point that returns a `Query`.
- Implement `check_schema()` by comparing the database's `alembic_version` row to the head revision in `alembic/versions/`.
- Define the `RegisterResult` dataclass returned by `register`.

#### Coexistence with Alembic

`Store.__init__` calls `schema.Base.metadata.create_all(engine)` only when no `alembic_version` table is present in the database. Once Alembic has stamped a revision, `create_all` is locked out — otherwise edits to `wallow.toml` could silently add columns the DB hasn't recorded a revision for, masking drift.

`Store.check_schema()` and `Store._maybe_check_schema()` (the constructor-time check when `check_schema=True`) follow this rule:

- **No `alembic.ini` discoverable + no `alembic_version` table** → no-op. Phase 1/2 quick-start, in-memory tests, etc.
- **`alembic.ini` discoverable + `alembic_version` table exists** → compare current rev to head; raise `PendingMigrationError` if they differ.
- **No `alembic.ini` + `alembic_version` table exists** → `WallowError` from `check_schema()` (the project is partially set up).

`Store.migrate()` raises `WallowError` if no `alembic.ini` is discoverable.

#### Session management

`Store.session()` is a context manager that yields a SQLAlchemy `Session`. On clean exit, `session.commit()`. On exception, `session.rollback()` and re-raise. This is the only sanctioned way to get a session; downstream code that wants finer control uses `store.engine` and SQLAlchemy directly.

#### `register()` implementation sketch

The shared identifying-prep step (default-fill → validate-keys → validate-types → normalise floats) is factored into `_prepare_identifying(schema, identifying)` and reused by `register`, `find`, and `heartbeat`.

```python
def _prepare_identifying(schema, identifying):
    out = schema.fill_identifying_defaults(identifying)
    schema.validate_identifying_keys(out)
    for k, v in out.items():
        schema.validate_value(schema.field(k), v, allow_none=False)
    return {k: schema.normalise_identifying_value(k, v) for k, v in out.items()}

def register(store, *, identifying, annotating=None,
             on_duplicate, stale_after=None):
    if on_duplicate == "claim_if_stale" and not isinstance(stale_after, timedelta):
        raise ValueError("on_duplicate='claim_if_stale' requires `stale_after`")

    schema = store.schema
    annotating = dict(annotating or {})
    schema.validate_annotating_keys(annotating)
    for k, v in annotating.items():
        schema.validate_value(schema.field(k), v, allow_none=True)
    identifying = _prepare_identifying(schema, identifying)

    Run = schema.Run
    with store.session() as s:
        new_run = Run(**identifying, **annotating)
        s.add(new_run)
        try:
            s.flush()
            return RegisterResult(run=new_run, was_inserted=True)
        except IntegrityError:
            s.rollback()
            existing = s.scalar(select(Run).filter_by(**identifying))

            if on_duplicate == "raise":
                s.expunge(existing)
                raise DuplicateRunError(existing)
            if on_duplicate == "return_existing":
                return RegisterResult(run=existing, was_inserted=False)
            if on_duplicate == "skip":
                return RegisterResult(run=None, was_inserted=False, was_skipped=True)
            if on_duplicate == "overwrite":
                for k, v in annotating.items():
                    setattr(existing, k, v)
                s.flush()
                return RegisterResult(run=existing, was_inserted=False, was_updated=True)
            if on_duplicate == "claim_if_stale":
                now = _utcnow()
                last = _make_naive_aware(existing.updated_at)  # SQLite drops tz
                if last is None or (now - last) > stale_after:
                    for k, v in annotating.items():
                        setattr(existing, k, v)
                    existing.updated_at = now    # bump unconditionally
                    s.flush()
                    return RegisterResult(run=existing, was_inserted=False, was_updated=True)
                return RegisterResult(run=existing, was_inserted=False, was_skipped=True)
```

The implementer must ensure the rolled-back `new_run` doesn't pollute the session. `_make_naive_aware` reattaches `tzinfo=UTC` to a naive `updated_at` read back from SQLite (SQLAlchemy's default DateTime column is timezone-naive; tz-aware values written by `_utcnow` come back without tzinfo).

`heartbeat(store, identifying)` is the symmetric primitive: prepare the identifying tuple (defaults filled, floats normalised), look up the row, set `updated_at = _utcnow()`, raise `SchemaValidationError` if the row doesn't exist. It exists so a long-running worker can signal liveness without spuriously rewriting annotating fields.

### 6.3 `dsl.py`

Responsibilities:

- Implement `F`, `Field`, `Expr`, `Query`, `OrderClause`.
- Compile expressions to SQLAlchemy.

#### AST nodes

Internally, `Field` and `Expr` are AST nodes:

```python
@dataclass(frozen=True)
class FieldRef:
    name: str
    json_path: tuple[str, ...] = ()  # empty if not a JSON path

@dataclass(frozen=True)
class Compare:
    op: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    field: FieldRef
    value: Any

@dataclass(frozen=True)
class In:
    field: FieldRef
    values: tuple[Any, ...]
    negate: bool

@dataclass(frozen=True)
class StringOp:
    op: Literal["contains", "startswith", "endswith"]
    field: FieldRef
    value: str

@dataclass(frozen=True)
class IsNull:
    field: FieldRef
    negate: bool

@dataclass(frozen=True)
class And: ...
@dataclass(frozen=True)
class Or: ...
@dataclass(frozen=True)
class Not: ...
```

The dataclasses are private; users only see `Field` and `Expr` (which wrap these).

#### Compilation

`Expr.compile(model_class)` walks the AST and produces a SQLAlchemy `ColumnElement`. For each AST node, look up the column on `model_class` via `getattr`, validate types, and emit the corresponding SQLAlchemy expression. For JSON paths, emit `func.json_extract(column, f"$.{'.'.join(path)}")`.

#### Field validation

When compiling, the implementer must:

- Look up `model_class.__wallow_schema__` to access the `Schema`.
- Validate that each `FieldRef.name` exists in the schema. Raise `SchemaValidationError` with a list of valid names if not.
- Validate that string operators are only used on string-typed fields, JSON paths only on JSON-typed fields, etc.
- Coerce comparison values to the field's type where reasonable.
- After coercion, normalise the RHS of comparisons against an *identifying* float field (skipping JSON-path comparisons) using `schema.normalise_identifying_value`, mirroring the rule applied in `register` / `find`. This applies to both `_Compare` and `_In` nodes.

`F(name, schema=...)` performs the eager-validation check before constructing the AST node — it queries `schema.fields` membership and raises `SchemaValidationError` immediately. The schema reference is stored on the resulting `Field` and propagated through `Field.json_path()` so chained ops keep the eager binding.

### 6.4 `migrations.py`

Responsibilities:

- Programmatic wrappers around `alembic.command` so the CLI doesn't shell out.
- Snapshot management: copy `wallow.toml` next to each generated revision.
- Schema state queries: current revision, head revision, pending migrations.
- Pre-flight diff for unsafe schema changes (identifying-drop, missing-default).
- Collision detection for identifying-field drop.

#### Public API

```python
def discover_alembic_ini(start: Path | None = None) -> Path | None
def current_revision(engine: Engine) -> str | None
def head_revision(config: Config) -> str | None
def is_pending(engine: Engine, config: Config) -> bool

def generate(
    config: Config, *, message: str, schema_path: Path,
    snapshots_dir: Path | None = None,
) -> Path
def apply(config: Config, *, target: str = "head") -> None
def downgrade(config: Config, *, target: str) -> None
def stamp(config: Config, *, revision: str = "head") -> None
def history(config: Config) -> list[Script]

def write_snapshot(revision_id: str, schema_path: Path, snapshots_dir: Path) -> Path
def find_collisions_after_drop(store: Store, field_name: str) -> list[CollisionGroup]

@dataclass(frozen=True)
class CollisionGroup:
    field_values: dict[str, Any]   # values of the *remaining* identifying fields
    row_ids: list[int]             # the runs.id values of the colliding rows (>=2)
```

`current_revision`, `head_revision`, `find_collisions_after_drop`, and `CollisionGroup` are also re-exported from `wallow.__init__` for convenience.

#### Alembic `Config` construction

```python
def _make_config(alembic_ini_path: Path, db_url: str | None = None) -> Config:
    ini = Path(alembic_ini_path).resolve()
    cfg = Config(str(ini))
    # Anchor script_location and a relative sqlite URL to the ini's directory
    # so the project is portable regardless of the caller's cwd.
    sl = cfg.get_main_option("script_location") or "alembic"
    if not Path(sl).is_absolute():
        cfg.set_main_option("script_location", str(ini.parent / sl))
    if db_url is None:
        db_url = cfg.get_main_option("sqlalchemy.url") or ""
    cfg.set_main_option("sqlalchemy.url", _resolve_sqlite_url(db_url, ini.parent))
    return cfg
```

`_resolve_sqlite_url(url, ini_dir)` turns a relative `sqlite:///runs.db` into an absolute `sqlite:////abs/path/to/runs.db`. The same resolution runs in three places: `_make_config` (for `migrations.py` callers), `cli._db_url_from_ini` (for `inspect`/`status`), and `env.py.template` (for Alembic's own engine construction). Keeping them in sync means the project directory is fully portable; only `alembic.ini` and the schema TOML need to travel together.

#### Discovery rule

`discover_alembic_ini(start=None)` walks up from `start` (defaulting to cwd) looking for `alembic.ini`. The CLI calls this with `start=cwd`; `Store` falls back to walking up from `db_path.parent` if cwd doesn't yield a result.

#### `env.py` integration

`alembic.ini` carries a custom main option `wallow_schema = wallow.toml` (relative to the ini's directory). `env.py.template` reads it via `config.get_main_option("wallow_schema")`, calls `wallow.load_schema(...)`, and sets `target_metadata = schema.Base.metadata`. Both `run_migrations_offline()` and `run_migrations_online()` pass `render_as_batch=True` to `context.configure()` — required for SQLite to drop or rebuild constraints (used by the identifying-field-add migration in §8.2).

#### Snapshot mechanism

After `alembic.command.revision(...)`, the returned `Script` object's `.revision` is the new revision id and `.path` is the revision file. `write_snapshot(rev_id, schema_path, snapshots_dir)` copies `wallow.toml` to `{snapshots_dir}/{rev_id}.toml`, prepending a three-line header:

```
# wallow migration snapshot — DO NOT EDIT
# revision = <rev_id>
# generated = <utc iso8601>
```

TOML treats `#` lines as comments, so the snapshot remains a valid wallow schema and can be loaded via `load_schema()` for the pre-flight diff.

#### Pre-flight diff

`generate()` runs two checks before invoking `alembic.command.revision`, both gated on the head snapshot existing:

1. **Identifying drop.** If `new_schema.identifying ⊊ head_schema.identifying`, raise `WallowError` directing the user at `find_collisions_after_drop` (spec §8.2 mandates the abort).
2. **New identifying without default.** For each newly-added identifying field with `default is None`, raise `WallowError` (spec §9 — NOT NULL columns can't be added to a non-empty table without a default).

If no head snapshot exists (i.e., the user deleted it, or this is the first migration) the checks are skipped silently.

#### Schema diff for status

`Store.check_schema()` and `wallow status` need to compare:

- Current applied revision: read from the database's `alembic_version` table.
- Head revision: introspect `alembic/versions/` via `ScriptDirectory`.

If they differ, schema is pending. The implementer should not attempt to compare the *content* of the TOML against the applied schema — that's what Alembic autogenerate is for.

### 6.5 `cli.py`

Responsibilities:

- Top-level `argparse` entry point.
- Subcommands: `init`, `migrate {generate, apply, downgrade, history, stamp}`, `status`, `inspect`.
- Resolve `--schema`, `--db`, and `--alembic-ini` paths, defaulting to discovery (see §5.9).

The CLI is thin: it parses args, calls into `migrations.py` or `store.py`, formats output. Errors are printed via `sys.stderr.write("wallow: error: ...\n")` and the process exits non-zero.

#### Template loading

Templates ship under `wallow/templates/` (a regular package — `templates/__init__.py` is empty so `importlib.resources` can traverse it). `cli._read_template(*parts)` calls `importlib.resources.files("wallow.templates").joinpath(*parts).read_text()`. This works for editable installs, wheels, and zipped wheels uniformly — `pathlib.Path(__file__).parent / "templates"` would break for the zipped-wheel case.

`pyproject.toml` ships the templates via:

```toml
[tool.setuptools.package-data]
wallow = ["templates/*", "templates/**/*"]
```

#### Exit codes

- `init`: 0 on success, 1 if any target file exists and `--force` was not passed.
- `migrate generate`: 1 if pre-flight detects an unsafe identifying-drop or missing-default.
- `migrate downgrade base` without `--yes`: 1.
- `status`: 0 when in sync, 1 when pending OR no `alembic.ini` discoverable.
- `inspect`: 0 on success, 1 if the id is not found.
- 2 reserved for argparse usage errors (argparse's default).

### 6.6 `errors.py`

See §4.6.

---

## 7. Storage model

The implementer should produce a `runs` table matching the TOML, with one column per declared field plus `id`, `created_at`, `updated_at`. The `UniqueConstraint` is named `uq_runs_identifying` and includes all identifying fields (sorted alphabetically). Indexes are named `ix_runs_<field>` per SQLAlchemy convention.

Example DDL produced for the matching-feedback schema:

```sql
CREATE TABLE runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      DATETIME NOT NULL,
    updated_at      DATETIME NOT NULL,

    cell_k          INTEGER NOT NULL,
    cell_sigma      REAL    NOT NULL,
    generation      INTEGER NOT NULL,
    candidate_id    INTEGER NOT NULL,
    seed            INTEGER NOT NULL DEFAULT 0,

    git_commit      VARCHAR,
    host            VARCHAR,
    status          VARCHAR,
    started_at      DATETIME,
    wall_clock_sec  REAL,
    val_loss        REAL,
    val_accuracy    REAL,
    artefacts_dir   VARCHAR,
    structural_traj JSON,
    discovered_T    JSON,

    CONSTRAINT uq_runs_identifying UNIQUE (candidate_id, cell_k, cell_sigma, generation, seed)
);

CREATE INDEX ix_runs_cell_k          ON runs(cell_k);
-- ... one per identifying + indexed annotating field
```

The `alembic_version` table is created by Alembic.

---

## 8. Migrations

### 8.1 What's free (O(1) on existing rows)

- Adding an annotating field. New nullable column; no backfill needed.
- Adding an index to an existing field. Single `CREATE INDEX`.
- Changing field documentation (`doc`). Documentation lives in TOML and the snapshot; doesn't change DDL. Alembic won't generate a migration for `doc`-only changes — the implementer must accept this and not surface it as a problem.
- Renaming an annotating field. Implemented as a copy + drop via Alembic's `batch_alter_table`. Doesn't affect dedup.

### 8.2 What's O(n)

- Adding an identifying field. The migration generator emits an `add_column` plus a `drop_constraint` + `create_unique_constraint` to rebuild `uq_runs_identifying` with the new field included. The user must specify a `default` in the TOML for the new identifying field; Alembic uses it as `server_default` to populate existing rows.

  Example generated migration:

  ```python
  def upgrade():
      with op.batch_alter_table("runs") as batch_op:
          batch_op.add_column(
              sa.Column("warmup_steps", sa.Integer(), nullable=False,
                        server_default="0")
          )
          batch_op.drop_constraint("uq_runs_identifying", type_="unique")
          batch_op.create_unique_constraint(
              "uq_runs_identifying",
              ["candidate_id", "cell_k", "cell_sigma", "generation",
               "seed", "warmup_steps"],
          )
  ```

- Removing an identifying field. The migration generator must detect this and *abort* with a clear message: removing an identifying field can produce duplicate keys, which the user must resolve manually before the migration can proceed. The detection is implemented as a **pre-flight diff** inside `migrations.generate()`: it compares `new_schema.identifying` against the head revision's snapshot at `alembic/snapshots/{head}.toml`, and if the new set is a strict subset, raises `WallowError` before invoking Alembic autogenerate. The implementer provides a helper:

  ```python
  wallow.migrations.find_collisions_after_drop(store, field_name) -> list[CollisionGroup]
  ```

  that returns groups of rows that would collide if `field_name` were dropped from identifying. Each `CollisionGroup` carries `field_values: dict[str, Any]` (the values of the *remaining* identifying fields shared by the colliding rows) and `row_ids: list[int]` (always ≥2). Empty list = safe to drop. The user resolves them (delete duplicates, merge annotating data, etc.), then re-runs the migration generator.

- Promoting an annotating field to identifying, or demoting identifying to annotating. Both are implemented as paired migrations; the implementer should generate them as the user expects — promotion may also trigger collision resolution, demotion is free.

### 8.3 Snapshot mechanism

Each generated revision file has a corresponding `alembic/snapshots/{revision_id}.toml` containing the schema state at the time of generation. This is for human review and future tooling, not for Alembic itself.

### 8.4 Migration safety

Every migration runs in a single transaction (Alembic's default). Migration files are immutable once applied to any environment — the implementer should document this in the README and include it in the `script.py.mako` template comment.

---

## 9. Edge cases

The implementer must handle these explicitly. Tests for each are required.

**Float normalisation.** Identifying float values are rounded to `schema.float_precision` significant figures (default 12) at register / find / DSL-compile time so IEEE-754 mantissa noise (e.g. `0.1 + 0.2` vs `0.3`) collapses to the same canonical float. NaN remains rejected; ±inf and 0.0 pass through normalisation unchanged. Annotating floats are not normalised — they preserve full precision. Set `[project] float_precision = N` to tune (large N → bit-exact identity; small N → looser dedup).

**`claim_if_stale` semantics.** On a UNIQUE collision, compare `existing.updated_at` (made tz-aware UTC if SQLite stripped tz) to `_utcnow()`. If `now - updated_at > stale_after`, overwrite the annotating fields and bump `updated_at`; return `RegisterResult(was_updated=True)`. Otherwise return the existing row untouched with `was_skipped=True`. Calling without `stale_after` is a `ValueError`. The companion `heartbeat()` function bumps `updated_at` without other side effects so a live worker can signal liveness during long silent training intervals.

**NaN in identifying floats.** Reject at register time with `SchemaValidationError`.

**Bool vs int.** Distinct types. Pass `True`, not `1`, for bool fields. The implementer must use `type(value) is bool` not `isinstance(value, bool)` because `isinstance(True, int)` is True in Python.

**Concurrent `register()` from two processes.** SQLite WAL handles this: one INSERT wins, the other catches IntegrityError and applies its `on_duplicate` policy. Test this.

**Very large JSON blobs.** Warn at insert time when an annotating value > 1 MB; do not enforce. Document the recommendation to use `path` fields for blobs > 1 MB.

**Schema file edited while Store is open.** The Store snapshots the schema at construction. Subsequent edits to `wallow.toml` aren't seen until the Store is recreated.

**Missing TOML default for identifying field.** When a user adds an identifying field to TOML without a `default`, the migration generator must abort with a message instructing them to add one (or to use a different migration strategy).

**Reserved name collision.** If a user declares a field named `id`, raise `SchemaParseError` immediately at TOML load.

---

## 10. Testing strategy

The test suite must cover:

### `test_schema.py`
- TOML parsing happy paths for each type.
- Reserved name rejection.
- Duplicate field name rejection.
- Identifying field with disallowed type.
- Schema with no identifying fields.
- Default value type coercion.
- Dynamically generated `Run` class has the expected columns and constraints.

### `test_store.py`
- `register()` happy path: new run inserts, returns Run.
- `register()` with each `on_duplicate` value.
- `find()` returns existing run; returns None for missing.
- Schema validation errors for mismatched identifying keys.
- Type validation errors for wrong-type values.
- NaN rejection.
- Concurrent register from two processes (use threading or multiprocessing).
- WAL mode is enabled.
- `check_schema()` passes for current schema, fails for pending migrations.

### `test_dsl.py`
- Each comparison operator compiles correctly.
- `&`, `|`, `~` compose correctly.
- `in_`, `not_in` with various iterables.
- String operators on string fields.
- String operators on non-string fields raise.
- JSON path on json fields compiles to `json_extract`.
- JSON path on non-json fields raises.
- Field name resolution: known names work, unknown names raise with helpful message.
- `Query.all()`, `.first()`, `.one()`, `.count()`, `.exists()`.
- Iteration is streaming (doesn't materialise all rows).
- `order_by`, `limit`, `offset`.

### `test_migrations.py`
- `wallow init` creates expected files.
- `migrate generate` produces a revision file when schema differs from DB.
- Snapshot is written next to revision.
- `migrate apply` brings DB to head.
- Adding annotating field: migration applies cleanly.
- Adding identifying field with default: migration applies cleanly.
- Dropping identifying field: collision detection works.
- `migrate downgrade` reverses changes.

### `test_cli.py`
- `wallow init` end-to-end in a temp directory.
- `wallow status` reports correctly with no DB, with current DB, with pending migrations.
- `wallow inspect <id>` formats correctly.
- Error paths: missing TOML, malformed TOML, etc.

---

## 11. Implementation phases

**Phase 1 (~1 week): schema + store core.**
Implement `schema.py` with TOML parsing and dynamic model generation. Implement `store.py` with `register()` and `find()`. Tests for schema validation and dedup. No DSL, no CLI, no migrations yet — schema changes require dropping the database.

**Phase 2 (~3-4 days): DSL.**
Implement `dsl.py` with `F`, `Field`, `Expr`, `Query`. Tests for compilation, field validation, all operators. Integrate with `Store.where()`.

**Phase 3 (~1 week): migrations.**
Implement `migrations.py` and the CLI's `migrate` subcommands. Set up Alembic templates. Implement snapshot mechanism. Tests for free-vs-O(n) migration paths and collision detection.

**Phase 4 (~3 days): CLI polish, init command, status.**
Implement `cli.py` fully. Implement `init` with file copying. Tests for CLI happy and error paths.

**Phase 5 (~3 days): documentation, packaging, release.**
README with quickstart and examples. Docstrings on all public API. Package for PyPI release (or for installation via local pip if not yet ready).

Total estimate: 3-4 weeks for a careful implementation with thorough tests. A coding agent that's already familiar with SQLAlchemy + Alembic could compress this to ~1-2 weeks.

---

## 12. Out of scope for v1

Documented here so the implementer doesn't drift towards them:

- Postgres backend. The abstraction supports it (Alembic + SQLAlchemy are backend-agnostic), but v1 tests SQLite only.
- Multi-writer / distributed dispatcher coordination beyond SQLite WAL.
- Time-series metrics tables.
- Run lineage (parent-child relationships).
- Tagging.
- Web UI / dashboard.
- `wallow query` CLI subcommand.
- Custom user-defined types.
- Schema validation hooks beyond type checks.
- Cross-database migrations.

---

## 13. Open questions for implementer

These aren't blockers; flagging where the implementer should pick a sensible default and document it in code:

1. **`Schema.Run` is regenerated each load.** Two separate `load_schema()` calls produce two distinct `Run` classes that aren't `is`-equal. Document this; users should hold a single `Schema` reference.
2. **`F` is a free function vs a class.** Either works. Recommend `F` as a function returning a `Field` instance, for visual lightness.
3. **`Query` lazy semantics.** Current spec: lazy until materialization method is called. Implementer should ensure the SQL is built once at materialization, not on every chained method.
4. **Migration name slug truncation.** Alembic's default truncates at 40 characters. Keep it; document it.
5. **Whether to publish to PyPI.** Out of scope for the implementation spec; the implementer produces the package, the maintainer decides on release.

---

## 14. Validation example

The implementer should include an `examples/matching_feedback/` directory containing:

- `wallow.toml` with the schema from §3.
- `dispatcher.py` showing how the matching-feedback project from `structured_feedback_proposal_v4.md` (§10) integrates: the dispatcher calls `wallow.register(..., on_duplicate="return_existing")` for each (cell, generation, candidate) tuple, and skips re-running candidates that already exist in the database.
- `analysis.ipynb` (or a `.py` script if Jupyter is too heavy) showing how downstream plotting code uses the DSL: `store.where((F("cell_k") == 4) & (F("status") == "completed")).order_by(F("val_accuracy").desc()).limit(10).all()`.
- A second migration in the example showing the "add warmup_steps" workflow end to end.

If the example runs and the DSL produces sensible output, the implementation is on track.

---

## 15. Implementation notes (as-built)

This section was added after Phases 3–4 landed. It records the resolved decisions and the small set of additions beyond the original §1–§14, so a future maintainer can quickly see what was actually built.

### Decisions resolved during implementation

- **`server_default` for identifying-with-default fields.** §6.1 was extended with the `_server_default_literal(decl)` rule. Without it, Alembic autogenerate produced unapplyable migrations for the §8.2 "add identifying field" case.
- **Per-instance `Schema.Base`.** Each `load_schema()` call builds a fresh `DeclarativeBase` subclass so two schemas don't share metadata. This was already in §6.1's sketch; calling it out because `env.py` has to call `load_schema()` exactly once per Alembic invocation.
- **`render_as_batch=True`** in `env.py.template`. Required for SQLite to drop or rebuild constraints (the §8.2 example uses `op.batch_alter_table`).
- **Custom `wallow_schema` main option in `alembic.ini`.** `env.py` reads the schema path via `config.get_main_option("wallow_schema")`. Idiomatic Alembic, survives across `alembic.command.X` invocations, no env-var pollution.
- **Lazy resolution of relative SQLite URLs.** `alembic.ini` ships `sqlalchemy.url = sqlite:///runs.db` (relative). `_resolve_sqlite_url(url, ini_dir)` rewrites this to the absolute path against `ini_dir` in three places: `migrations._make_config`, `cli._db_url_from_ini`, and `env.py.template`. Without this, running from any cwd other than the project root creates a stray DB file.
- **`Store` and Alembic coexistence.** `Store.__init__` calls `create_all` only when there's no `alembic_version` table. See §6.2.

### Additions beyond §1–§14

- **`wallow init --dir DIR`** (§5.1). Convenience for scripting and tests; without it, callers have to `chdir` into the target directory.
- **`wallow migrate stamp <revision>`** (§5.5b). Wraps `alembic.command.stamp`. Lets users adopt Alembic on an existing Phase 1/2 database without installing alembic separately.
- **`--alembic-ini PATH` flag** on every `migrate`/`status`/`inspect` subcommand (§5.9). Auto-discovers from cwd by default; the explicit flag is for tests, scripts, and projects whose layout differs from the default.
- **Snapshot header comment** (§6.4). Three lines marking the file auto-generated. TOML treats `#` as comments so the snapshot remains a valid wallow schema and can be re-loaded for the pre-flight diff.
- **Pre-flight diff inside `migrations.generate()`** (§5.2 + §6.4). Catches identifying-drop and missing-default before invoking Alembic autogenerate, producing clearer errors than what Alembic would emit downstream.

### Known limitations / follow-ups

- **Concurrent-register tests need a WAL bootstrap.** When the parent test process didn't open a `Store` first (e.g., the post-migration version of the concurrent test), the SQLite DB is still in rollback-journal mode and two writer subprocesses can deadlock. The fix in `tests/test_migrations.py` is to construct a parent `Store` once before forking workers; `Store._install_pragmas` sets `journal_mode=WAL` on its first connection.
- **Phase 5 (README, packaging, examples)** is still pending. The implementation is feature-complete and the test suite is green, but docstrings and an `examples/matching_feedback/` directory haven't been written.

### v0.2 ergonomics pass

Five API changes applied after the initial intuitiveness review. All are surfaced in §3, §4, §6, and §9; this section is the index.

1. **Identifying defaults are honoured at register / find time** (§4.3, §4.4, §6.1). `register(..., identifying={"lr": 1e-3})` now succeeds when the schema declares `default = 0` for `seed`. `Schema.fill_identifying_defaults` is the new helper; missing fields without a default still raise. Removes the long-standing surprise that the word "default" did everything except what users expected.
2. **`register` returns `RegisterResult`** (§4.3, §6.2). The dataclass carries `run`, `was_inserted`, `was_updated`, `was_skipped`. Breaking change: callers that did `run = register(...)` must now do `run = register(...).run`. Lets workers distinguish "I claimed this combo" from "I rejoined someone else's row" without a separate query.
3. **Eager DSL field validation** (§4.5, §6.1, §6.3). Two new opt-in forms: `F(name, schema=...)` raises `SchemaValidationError` at the callsite on a typo, and `schema.f.<name>` does the same via attribute access (with `dir(schema.f)` for IDE autocomplete). `F(name)` keeps the deferred-resolution semantics for cross-schema reuse.
4. **`claim_if_stale` policy + `heartbeat()`** (§4.3, §4.4b, §9). New `on_duplicate="claim_if_stale"` with required `stale_after: timedelta` enables live multi-worker dispatch by reading the existing row's `updated_at` and either claiming a stale row (was_updated) or skipping a fresh one (was_skipped). `wallow.heartbeat(store, identifying=...)` bumps `updated_at` without other side effects so a long training run can signal liveness. No new column needed.
5. **Identifying floats normalised by default** (§3, §4.3, §4.5, §9). Round to `[project] float_precision` significant figures (default 12) at register / find / DSL-compile time so `0.1 + 0.2` dedupes with `0.3`. Annotating floats are untouched. NaN stays rejected; ±inf and 0.0 pass through. Configurable per-schema; bit-exact behaviour requires `float_precision >= 17`.

---

*End of specification. The implementer should treat this as authoritative; deviations require justification in code comments and should be flagged for review.*
