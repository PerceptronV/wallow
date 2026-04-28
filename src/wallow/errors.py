"""Exception hierarchy for wallow.

All errors carry structured context as attributes so callers can branch
programmatically without parsing messages.
"""

from __future__ import annotations

from typing import Any


class WallowError(Exception):
    """Base class for all wallow-raised exceptions."""


class SchemaError(WallowError):
    """Base for schema-related problems (parse + runtime validation)."""


class SchemaParseError(SchemaError):
    """Raised when wallow.toml is malformed or violates declaration rules."""


class SchemaValidationError(SchemaError):
    """Raised when runtime values don't match the declared schema.

    Used both for `register`/`find` argument mismatches (wrong key set,
    wrong types) and for DSL compile-time errors (unknown field, operator
    misapplied to a field type).
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        expected_type: str | None = None,
        actual_type: str | None = None,
        missing_keys: frozenset[str] | None = None,
        extra_keys: frozenset[str] | None = None,
        valid_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.expected_type = expected_type
        self.actual_type = actual_type
        self.missing_keys = missing_keys
        self.extra_keys = extra_keys
        self.valid_names = valid_names


class PendingMigrationError(SchemaError):
    """Raised when the database is behind the schema's head revision."""

    def __init__(
        self,
        message: str,
        *,
        current_rev: str | None,
        head_rev: str | None,
    ) -> None:
        super().__init__(message)
        self.current_rev = current_rev
        self.head_rev = head_rev


class DuplicateRunError(WallowError):
    """Raised by `register(..., on_duplicate='raise')` on conflict."""

    def __init__(self, existing: Any) -> None:
        super().__init__(
            f"a run with these identifying fields already exists (id={getattr(existing, 'id', '?')})"
        )
        self.existing = existing
