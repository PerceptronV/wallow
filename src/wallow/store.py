"""Store: connection / session management, plus `register`, `find`, `heartbeat`."""

from __future__ import annotations

import datetime as _dt
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, TYPE_CHECKING

from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .errors import DuplicateRunError, PendingMigrationError, SchemaValidationError, WallowError
from .schema import Schema, _utcnow

if TYPE_CHECKING:
    from .dsl import Expr, Query


_OnDuplicate = Literal[
    "raise", "return_existing", "overwrite", "skip", "claim_if_stale"
]
_VALID_ON_DUPLICATE: frozenset[str] = frozenset(
    {"raise", "return_existing", "overwrite", "skip", "claim_if_stale"}
)


@dataclass(frozen=True)
class RegisterResult:
    """Outcome of a ``register`` call.

    ``run`` is the resulting Run (or ``None`` for ``on_duplicate='skip'``
    on a duplicate). The boolean flags describe what happened:

    - ``was_inserted=True`` — a new row was created by this call.
    - ``was_updated=True`` — an existing row's annotating fields were
      written by this call (``overwrite``, or ``claim_if_stale`` claiming
      a stale row).
    - ``was_skipped=True`` — an existing row was returned unchanged
      (``skip`` on duplicate, or ``claim_if_stale`` finding a fresh row).

    Exactly one flag is set per call. ``return_existing`` on a duplicate
    leaves all three False — the row was neither inserted, updated, nor
    skipped in any meaningful sense; the caller asked for it.
    """

    run: Any | None
    was_inserted: bool
    was_updated: bool = False
    was_skipped: bool = False


def _build_url(db_path: str | Path) -> str:
    if isinstance(db_path, str) and db_path == ":memory:":
        return "sqlite:///:memory:"
    p = Path(db_path)
    return f"sqlite:///{p}"


class Store:
    """SQLite-backed store for a wallow schema.

    When Alembic has been set up for the project (`alembic_version` table
    present), the Store defers all DDL to migrations: `create_all` is
    skipped on init, and `check_schema()` raises PendingMigrationError if
    the DB revision is behind the schema head.

    When Alembic is not set up (Phase 1/2 quick-start, in-memory tests),
    the Store falls back to `Base.metadata.create_all` and `check_schema()`
    is a no-op.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        schema: Schema,
        check_schema: bool = True,
    ) -> None:
        self._schema = schema
        self._db_path = db_path
        url = _build_url(db_path)
        # `check_same_thread=False` lets the multiprocessing concurrent test
        # share a session factory across threads if the OS allows it; SQLite
        # handles cross-process via WAL.
        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite:"):
            connect_args["check_same_thread"] = False
        self._engine: Engine = create_engine(
            url, future=True, connect_args=connect_args
        )

        is_memory = url.endswith(":memory:")
        self._install_pragmas(is_memory=is_memory)

        # Skip create_all once Alembic is in charge: it would silently add
        # columns that the DB hasn't recorded a revision for, masking drift.
        if not self._alembic_version_present():
            schema.Base.metadata.create_all(self._engine)

        self._session_factory = sessionmaker(
            self._engine, expire_on_commit=False, future=True
        )

        if check_schema:
            self._maybe_check_schema()

    # ---- engine / pragma setup -----------------------------------------

    def _install_pragmas(self, *, is_memory: bool) -> None:
        @event.listens_for(self._engine, "connect")
        def _on_connect(dbapi_connection, _record):  # type: ignore[no-untyped-def]
            cur = dbapi_connection.cursor()
            try:
                # WAL is a no-op on :memory: — skip to avoid a noisy warning.
                if not is_memory:
                    cur.execute("PRAGMA journal_mode=WAL")
                    cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA foreign_keys=ON")
            finally:
                cur.close()

    def _alembic_version_present(self) -> bool:
        return "alembic_version" in inspect(self._engine).get_table_names()

    def _discover_alembic_ini(self) -> Any:
        """Find an alembic.ini by walking up from cwd then from db_path.

        Lazy import of `migrations` to avoid pulling Alembic at module load.
        Returns a Path or None.
        """
        from . import migrations

        ini = migrations.discover_alembic_ini()
        if ini is not None:
            return ini
        if isinstance(self._db_path, (str, Path)) and str(self._db_path) != ":memory:":
            return migrations.discover_alembic_ini(Path(self._db_path).parent)
        return None

    def _maybe_check_schema(self) -> None:
        """Silently skip when Alembic isn't configured; otherwise enforce."""
        if not self._alembic_version_present():
            return
        # Alembic is in use. Defer to check_schema for the real comparison;
        # if no alembic.ini is discoverable we can't compare, so stay silent
        # rather than block opening the Store.
        try:
            self.check_schema()
        except WallowError as e:
            if isinstance(e, PendingMigrationError):
                raise
            return

    # ---- public properties ---------------------------------------------

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def engine(self) -> Engine:
        return self._engine

    # ---- session / execute --------------------------------------------

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def execute(self, statement: Any) -> Any:
        with self.session() as s:
            return s.execute(statement)

    # ---- DSL entry points ---------------------------------------------

    def where(self, *exprs: "Expr") -> "Query":
        from .dsl import Query  # local import to break circularity

        return Query(self).where(*exprs)

    def all(self) -> list[Any]:
        from .dsl import Query

        return Query(self).all()

    def count(self) -> int:
        from .dsl import Query

        return Query(self).count()

    # ---- migrations -----------------------------------------------------

    def check_schema(self) -> None:
        """Raise `PendingMigrationError` if the DB is behind the schema head.

        No-op when Alembic isn't configured for this project (no alembic.ini
        discoverable). Once `alembic_version` exists, an unreachable ini is
        a configuration bug and is reported as `WallowError`.
        """
        from . import migrations

        ini = self._discover_alembic_ini()
        if ini is None:
            if self._alembic_version_present():
                raise WallowError(
                    "database is alembic-managed but no alembic.ini was found; "
                    "pass --alembic-ini or run from the project directory"
                )
            return
        cfg = migrations._make_config(ini, db_url=str(self._engine.url))
        cur = migrations.current_revision(self._engine)
        head = migrations.head_revision(cfg)
        if cur != head:
            raise PendingMigrationError(
                f"db revision is {cur!r}, schema head is {head!r}; "
                f"run `wallow migrate apply`",
                current_rev=cur,
                head_rev=head,
            )

    def migrate(self) -> None:
        """Apply all pending migrations to the database."""
        from . import migrations

        ini = self._discover_alembic_ini()
        if ini is None:
            raise WallowError(
                "no alembic.ini found; run `wallow init` first or pass "
                "--alembic-ini explicitly"
            )
        cfg = migrations._make_config(ini, db_url=str(self._engine.url))
        migrations.apply(cfg)


# --- module-level register / find / heartbeat ------------------------------


def _prepare_identifying(schema: Schema, identifying: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults, validate, and normalise an identifying dict.

    Centralises the three-step preparation shared by register, find, and
    heartbeat: (1) fill missing keys from declared TOML defaults, (2) check
    the key set and per-field types, (3) normalise float identifying fields
    so IEEE-754 mantissa noise doesn't split dedup groups.
    """
    out = schema.fill_identifying_defaults(identifying)
    schema.validate_identifying_keys(out)
    for k, v in out.items():
        schema.validate_value(schema.field(k), v, allow_none=False)
    return {k: schema.normalise_identifying_value(k, v) for k, v in out.items()}


def _make_naive_aware(t: _dt.datetime | None) -> _dt.datetime | None:
    """Coerce a datetime read back from SQLite to tz-aware UTC.

    SQLAlchemy's default ``DateTime`` column on SQLite doesn't preserve
    tzinfo, so values written by ``_utcnow`` (tz-aware UTC) come back
    naive. We attach UTC back so arithmetic against a tz-aware ``now()``
    works without a TypeError.
    """
    if t is None:
        return None
    if t.tzinfo is None:
        return t.replace(tzinfo=_dt.timezone.utc)
    return t


def register(
    store: Store,
    *,
    identifying: dict[str, Any],
    annotating: dict[str, Any] | None = None,
    on_duplicate: _OnDuplicate,
    stale_after: _dt.timedelta | None = None,
) -> RegisterResult:
    """Register a run; returns a :class:`RegisterResult`.

    Identifying keys may omit any field that declares a TOML ``default``;
    the default is filled in automatically. Identifying float values are
    rounded to ``schema.float_precision`` significant figures so IEEE-754
    noise doesn't split dedup.

    ``on_duplicate='claim_if_stale'`` requires ``stale_after``: on a
    collision, if the existing row's ``updated_at`` is older than that
    delta, the row is overwritten with the provided annotating fields and
    the result has ``was_updated=True``. Otherwise the existing row is
    returned untouched and ``was_skipped=True``.
    """
    if on_duplicate not in _VALID_ON_DUPLICATE:
        raise ValueError(
            f"on_duplicate must be one of {sorted(_VALID_ON_DUPLICATE)}, "
            f"got {on_duplicate!r}"
        )
    if on_duplicate == "claim_if_stale":
        if not isinstance(stale_after, _dt.timedelta):
            raise ValueError(
                "on_duplicate='claim_if_stale' requires a datetime.timedelta "
                "`stale_after` (e.g. timedelta(minutes=5))"
            )

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
        except IntegrityError:
            s.rollback()
            existing = s.scalar(select(Run).filter_by(**identifying))
            assert existing is not None, "IntegrityError without a prior row?"

            if on_duplicate == "raise":
                # Detach so the caller can read attributes after this commit closes.
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
                return RegisterResult(
                    run=existing, was_inserted=False, was_updated=True
                )
            if on_duplicate == "claim_if_stale":
                now = _utcnow()
                last = _make_naive_aware(existing.updated_at)
                is_stale = last is None or (now - last) > stale_after  # type: ignore[operator]
                if is_stale:
                    for k, v in annotating.items():
                        setattr(existing, k, v)
                    # Bump updated_at unconditionally — onupdate only fires
                    # on a dirty UPDATE, and a claim with no annotating
                    # fields shouldn't silently leave the heartbeat stale.
                    existing.updated_at = now
                    s.flush()
                    return RegisterResult(
                        run=existing, was_inserted=False, was_updated=True
                    )
                return RegisterResult(
                    run=existing, was_inserted=False, was_skipped=True
                )
            # Unreachable: validated at the top of the function.
            raise AssertionError(f"unhandled on_duplicate={on_duplicate!r}")
        return RegisterResult(run=new_run, was_inserted=True)


def find(store: Store, **identifying: Any) -> Any:
    """Direct identifying-fields lookup. Returns the Run or None.

    Identifying keys may omit any field with a declared TOML ``default``;
    float identifying values are normalised the same way as ``register``
    so a lookup matches the row that was registered.
    """
    schema = store.schema
    identifying = _prepare_identifying(schema, identifying)
    Run = schema.Run
    with store.session() as s:
        return s.scalar(select(Run).filter_by(**identifying))


def heartbeat(store: Store, *, identifying: dict[str, Any]) -> _dt.datetime:
    """Bump ``updated_at`` for the run with this identifying tuple.

    Pairs with ``on_duplicate='claim_if_stale'`` for live multi-worker
    dispatch: a worker calls ``heartbeat`` periodically while training so
    other workers see the row as fresh and don't claim it. Returns the
    new ``updated_at`` (tz-aware UTC).

    Raises :class:`SchemaValidationError` when no row matches.
    """
    schema = store.schema
    identifying = _prepare_identifying(schema, identifying)
    Run = schema.Run
    now = _utcnow()
    with store.session() as s:
        existing = s.scalar(select(Run).filter_by(**identifying))
        if existing is None:
            raise SchemaValidationError(
                f"no run with identifying={identifying} to heartbeat",
            )
        existing.updated_at = now
        s.flush()
    return now
