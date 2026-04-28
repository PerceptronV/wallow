"""Store: connection / session management, plus `register` and `find`."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, TYPE_CHECKING

from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .errors import DuplicateRunError, PendingMigrationError, SchemaValidationError, WallowError
from .schema import Schema

if TYPE_CHECKING:
    from .dsl import Expr, Query


_OnDuplicate = Literal["raise", "return_existing", "overwrite", "skip"]
_VALID_ON_DUPLICATE: frozenset[str] = frozenset(
    {"raise", "return_existing", "overwrite", "skip"}
)


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


# --- module-level register / find -------------------------------------------


def register(
    store: Store,
    *,
    identifying: dict[str, Any],
    annotating: dict[str, Any] | None = None,
    on_duplicate: _OnDuplicate,
) -> Any:
    """Register a run; returns a Run (or None on 'skip' duplicate)."""
    if on_duplicate not in _VALID_ON_DUPLICATE:
        raise ValueError(
            f"on_duplicate must be one of {sorted(_VALID_ON_DUPLICATE)}, "
            f"got {on_duplicate!r}"
        )

    schema = store.schema
    annotating = dict(annotating or {})
    schema.validate_identifying_keys(identifying)
    schema.validate_annotating_keys(annotating)

    for k, v in identifying.items():
        schema.validate_value(schema.field(k), v, allow_none=False)
    for k, v in annotating.items():
        schema.validate_value(schema.field(k), v, allow_none=True)

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
                return existing
            if on_duplicate == "skip":
                return None
            if on_duplicate == "overwrite":
                for k, v in annotating.items():
                    setattr(existing, k, v)
                s.flush()
                return existing
            # Unreachable: validated at the top of the function.
            raise AssertionError(f"unhandled on_duplicate={on_duplicate!r}")
        return new_run


def find(store: Store, **identifying: Any) -> Any:
    """Direct identifying-fields lookup. Returns the Run or None."""
    schema = store.schema
    schema.validate_identifying_keys(identifying)
    for k, v in identifying.items():
        schema.validate_value(schema.field(k), v, allow_none=False)

    Run = schema.Run
    with store.session() as s:
        return s.scalar(select(Run).filter_by(**identifying))
