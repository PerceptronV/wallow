"""Crash-safe claim → train → finalise/fail context manager around ``register``.

Most sweep dispatchers want the same three-step pattern:

  1. Claim a row idempotently (``register(..., on_duplicate='return_existing')``).
     If it's already completed, skip.
  2. Mark the row ``running`` with a timestamp + provenance.
  3. On the body's success, write ``status='completed'`` plus results;
     on any exception, write ``status='failed'`` with a truncated traceback
     and re-raise.

This module wraps that into a context manager so worker code can focus on
training. Lives under ``contrib`` because the schema needs to declare specific
annotating fields for the helper to write into (``status``, ``started_at``,
``completed_at``, ``wallclock_seconds``, ``error_excerpt``); not every wallow
project will want that ceremony.
"""

from __future__ import annotations

import datetime as _dt
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator, Mapping, TYPE_CHECKING

from ..errors import WallowError
from ..store import register

if TYPE_CHECKING:
    from ..store import Store


# Maximum length of the truncated traceback excerpt persisted on failure.
_ERROR_EXCERPT_MAX = 1000


# ----- exceptions -----------------------------------------------------------


class AlreadyCompleted(WallowError):
    """Raised by ``run_lifecycle`` when the existing row is already complete.

    Carries the existing :attr:`run` so the caller can read its uuid / paths
    / metrics without re-querying the store.
    """

    def __init__(self, run: Any) -> None:
        uuid = getattr(run, "uuid", None)
        super().__init__(
            f"run already completed (uuid={uuid!r}); pass force=True to re-run"
        )
        self.run = run


# ----- handle yielded by the context manager -------------------------------


@dataclass
class WorkerHandle:
    """State shared between ``run_lifecycle`` and the body it wraps.

    The body reads :attr:`run` (the SQLAlchemy Run row) and :attr:`uuid` to
    locate artefacts, then calls :meth:`finalise` once training succeeds. On
    exception the lifecycle writes status='failed' automatically — the body
    does not need to call anything.
    """

    store: "Store"
    identifying: dict[str, Any]
    run: Any
    uuid: str
    started_at: _dt.datetime
    _t0: float = field(default_factory=perf_counter)
    _finalised: bool = False

    def elapsed(self) -> float:
        """Seconds since the lifecycle started (wallclock)."""
        return perf_counter() - self._t0

    def finalise(
        self,
        *,
        annotating: Mapping[str, Any] | None = None,
    ) -> None:
        """Write ``status='completed'`` plus the caller's result fields.

        Idempotent: calling twice in a row is a no-op (the second call is
        ignored). The lifecycle itself never calls ``finalise`` — the body
        must, and will be marked failed if it forgets.
        """
        if self._finalised:
            return
        ann: dict[str, Any] = {
            "status": "completed",
            "completed_at": _dt.datetime.now(_dt.timezone.utc),
            "wallclock_seconds": float(self.elapsed()),
        }
        if annotating:
            ann.update(annotating)
        register(
            self.store,
            identifying=self.identifying,
            annotating=ann,
            on_duplicate="overwrite",
        )
        self._finalised = True


# ----- the context manager --------------------------------------------------


@contextmanager
def run_lifecycle(
    store: "Store",
    *,
    identifying: dict[str, Any],
    force: bool = False,
    start_annotating: Mapping[str, Any] | None = None,
) -> Iterator[WorkerHandle]:
    """Claim a row, yield a :class:`WorkerHandle`, then finalise/fail on exit.

    Workflow:

    1. **Claim** — ``register(..., on_duplicate='return_existing')`` writes
       ``status='pending'``. If the row already exists with
       ``status='completed'`` and *force* is False, raises
       :class:`AlreadyCompleted` before yielding.
    2. **Start** — ``register(..., on_duplicate='overwrite')`` writes
       ``status='running'``, ``started_at=now``, plus any *start_annotating*
       fields the caller supplied (typically ``host``/``git_hash``/...).
    3. **Body runs** under ``yield``.
    4. **Success** — caller invokes ``handle.finalise(annotating={...})``.
       If the body returns without finalising, the lifecycle treats it as
       success and writes a minimal completion record itself.
    5. **Failure** — any exception triggers ``status='failed'`` with a
       truncated ``error_excerpt`` and the exception is re-raised.

    The schema must declare these annotating fields for the helper to write
    into: ``status`` (string), ``started_at``/``completed_at`` (datetime),
    ``wallclock_seconds`` (float), ``error_excerpt`` (string). Missing any
    of these raises :class:`SchemaValidationError` at the first
    ``register()`` call.
    """
    # --- 1. Claim -------------------------------------------------------
    claim_ann: dict[str, Any] = {"status": "pending"}
    if start_annotating:
        claim_ann.update(start_annotating)
    pre = register(
        store,
        identifying=identifying,
        annotating=claim_ann,
        on_duplicate="return_existing",
    )
    if not pre.was_inserted and pre.run.status == "completed" and not force:
        raise AlreadyCompleted(pre.run)

    # --- 2. Start -------------------------------------------------------
    started_at = _dt.datetime.now(_dt.timezone.utc)
    start_ann: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
    }
    if start_annotating:
        start_ann.update(start_annotating)
    started = register(
        store,
        identifying=identifying,
        annotating=start_ann,
        on_duplicate="overwrite",
    )
    run = started.run

    handle = WorkerHandle(
        store=store,
        identifying=dict(identifying),
        run=run,
        uuid=run.uuid,
        started_at=started_at,
    )

    # --- 3 & 4. Body + success/failure ----------------------------------
    try:
        yield handle
    except BaseException as exc:
        excerpt = _format_exception_excerpt(exc)
        register(
            store,
            identifying=identifying,
            annotating={
                "status": "failed",
                "completed_at": _dt.datetime.now(_dt.timezone.utc),
                "wallclock_seconds": float(handle.elapsed()),
                "error_excerpt": excerpt,
            },
            on_duplicate="overwrite",
        )
        raise
    else:
        # Body succeeded but didn't call finalise — write a minimal
        # completion record so the row doesn't stay 'running' forever.
        if not handle._finalised:
            handle.finalise()


def _format_exception_excerpt(exc: BaseException) -> str:
    """Short, single-line traceback summary for the ``error_excerpt`` field."""
    text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    if len(text) > _ERROR_EXCERPT_MAX:
        text = text[: _ERROR_EXCERPT_MAX - 3] + "..."
    return text
