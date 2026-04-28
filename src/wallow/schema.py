"""Schema loading: TOML → Schema with a dynamically generated `Run` model.

The schema lives in `wallow.toml`. This module parses it, validates the
declarations against the type catalogue and reserved-name list, and
synthesises a SQLAlchemy declarative `Base` + `Run` class that downstream
code uses with both the DSL and raw SQLAlchemy.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase

from .errors import SchemaParseError, SchemaValidationError

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover - exercised only on 3.10
    import tomli as _toml


# --- type catalogue ----------------------------------------------------------

# `python_type` is the canonical Python type used for isinstance-style checks.
# `sa_factory` builds a fresh SQLAlchemy type instance every time it's called
# (SQLAlchemy types are sometimes mutable; safer to mint per-column).
_TYPE_CATALOGUE: dict[str, tuple[type, Any]] = {
    "int": (int, lambda: Integer()),
    "float": (float, lambda: Float()),
    "string": (str, lambda: String()),
    "bool": (bool, lambda: Boolean()),
    "json": (object, lambda: JSON()),
    "path": (str, lambda: String()),
    "datetime": (_dt.datetime, lambda: DateTime()),
}

_IDENTIFYING_ALLOWED: frozenset[str] = frozenset({"int", "float", "string", "bool"})

_RESERVED_NAMES: frozenset[str] = frozenset({"id", "created_at", "updated_at"})
_RESERVED_PREFIX = re.compile(r"^_wallow_", re.IGNORECASE)

_VALID_FIELD_KEYS: frozenset[str] = frozenset(
    {"type", "doc", "indexed", "default", "nullable"}
)


# --- public dataclasses ------------------------------------------------------


@dataclass(frozen=True)
class FieldDecl:
    """Declared field as it came out of the TOML parser."""

    name: str
    type: str
    is_identifying: bool
    indexed: bool
    nullable: bool
    default: Any | None
    doc: str | None

    def python_type(self) -> type:
        return _TYPE_CATALOGUE[self.type][0]

    def sa_type(self) -> Any:
        return _TYPE_CATALOGUE[self.type][1]()


# --- parsing -----------------------------------------------------------------


def load_schema(path: str | Path) -> "Schema":
    """Parse a wallow.toml file and return a Schema with a generated model."""
    p = Path(path)
    try:
        with p.open("rb") as fh:
            data = _toml.load(fh)
    except FileNotFoundError as e:
        raise SchemaParseError(f"schema file not found: {p}") from e
    except _toml.TOMLDecodeError as e:
        raise SchemaParseError(f"invalid TOML in {p}: {e}") from e
    return _parse(data)


def _parse(data: Mapping[str, Any]) -> "Schema":
    project = data.get("project")
    if not isinstance(project, Mapping):
        raise SchemaParseError("missing required [project] table")
    name = project.get("name")
    if not isinstance(name, str) or not name:
        raise SchemaParseError("[project].name must be a non-empty string")
    description = project.get("description")
    if description is not None and not isinstance(description, str):
        raise SchemaParseError("[project].description must be a string if present")

    raw_id = data.get("identifying", {})
    raw_an = data.get("annotating", {})
    if not isinstance(raw_id, Mapping):
        raise SchemaParseError("[identifying] must be a table")
    if not isinstance(raw_an, Mapping):
        raise SchemaParseError("[annotating] must be a table")

    if not raw_id:
        raise SchemaParseError("schema must declare at least one identifying field")

    overlap = set(raw_id) & set(raw_an)
    if overlap:
        raise SchemaParseError(
            f"field names appear in both [identifying] and [annotating]: {sorted(overlap)}"
        )

    fields: dict[str, FieldDecl] = {}
    for fname, decl in raw_id.items():
        fields[fname] = _parse_field_decl(fname, decl, is_identifying=True)
    for fname, decl in raw_an.items():
        fields[fname] = _parse_field_decl(fname, decl, is_identifying=False)

    identifying = frozenset(raw_id.keys())
    annotating = frozenset(raw_an.keys())
    return Schema(
        project_name=name,
        description=description,
        fields=fields,
        identifying=identifying,
        annotating=annotating,
    )


def _parse_field_decl(
    name: str, decl: Any, *, is_identifying: bool
) -> FieldDecl:
    if _RESERVED_NAMES.__contains__(name.lower()) or _RESERVED_PREFIX.match(name):
        raise SchemaParseError(
            f"field name {name!r} is reserved by wallow "
            f"(reserved: {sorted(_RESERVED_NAMES)} and any name starting with '_wallow_')"
        )
    if not isinstance(decl, Mapping):
        raise SchemaParseError(f"field {name!r}: declaration must be a TOML table")

    extra_keys = set(decl) - _VALID_FIELD_KEYS
    if extra_keys:
        raise SchemaParseError(
            f"field {name!r}: unknown keys {sorted(extra_keys)} "
            f"(valid keys: {sorted(_VALID_FIELD_KEYS)})"
        )

    ftype = decl.get("type")
    if not isinstance(ftype, str):
        raise SchemaParseError(f"field {name!r}: 'type' is required and must be a string")
    if ftype not in _TYPE_CATALOGUE:
        raise SchemaParseError(
            f"field {name!r}: unknown type {ftype!r} "
            f"(valid types: {sorted(_TYPE_CATALOGUE)})"
        )

    if is_identifying and ftype not in _IDENTIFYING_ALLOWED:
        raise SchemaParseError(
            f"field {name!r}: type {ftype!r} is not allowed for identifying fields "
            f"(allowed: {sorted(_IDENTIFYING_ALLOWED)})"
        )

    doc = decl.get("doc")
    if doc is not None and not isinstance(doc, str):
        raise SchemaParseError(f"field {name!r}: 'doc' must be a string")

    indexed = decl.get("indexed")
    if indexed is None:
        indexed = is_identifying  # spec §3: default true for identifying, false for annotating
    if not isinstance(indexed, bool):
        raise SchemaParseError(f"field {name!r}: 'indexed' must be a bool")

    nullable = decl.get("nullable")
    if nullable is None:
        nullable = not is_identifying  # default false for identifying, true for annotating
    if not isinstance(nullable, bool):
        raise SchemaParseError(f"field {name!r}: 'nullable' must be a bool")
    if is_identifying and nullable:
        # NULL in a UNIQUE constraint silently breaks dedup on most backends.
        raise SchemaParseError(
            f"field {name!r}: identifying fields must be NOT NULL"
        )

    default = decl.get("default")
    if default is not None:
        py_type = _TYPE_CATALOGUE[ftype][0]
        # bool/int strictness mirrors register-time validation: bool isn't an int.
        if ftype == "int" and type(default) is bool:
            raise SchemaParseError(
                f"field {name!r}: default {default!r} is bool, not int"
            )
        if ftype == "float" and type(default) is bool:
            raise SchemaParseError(
                f"field {name!r}: default {default!r} is bool, not float"
            )
        if ftype == "float" and isinstance(default, int):
            default = float(default)
        elif not isinstance(default, py_type):
            raise SchemaParseError(
                f"field {name!r}: default {default!r} is not coercible to {ftype}"
            )

    return FieldDecl(
        name=name,
        type=ftype,
        is_identifying=is_identifying,
        indexed=bool(indexed),
        nullable=bool(nullable),
        default=default,
        doc=doc,
    )


# --- Schema ------------------------------------------------------------------


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class Schema:
    """Parsed wallow schema with the dynamically generated `Run` model."""

    project_name: str
    description: str | None
    fields: dict[str, FieldDecl]
    identifying: frozenset[str]
    annotating: frozenset[str]
    Base: type
    Run: type

    def __init__(
        self,
        *,
        project_name: str,
        description: str | None,
        fields: dict[str, FieldDecl],
        identifying: frozenset[str],
        annotating: frozenset[str],
    ) -> None:
        self.project_name = project_name
        self.description = description
        self.fields = fields
        self.identifying = identifying
        self.annotating = annotating
        self.Base, self.Run = _build_model(self)

    # iteration / lookup --------------------------------------------------

    def field(self, name: str) -> FieldDecl:
        try:
            return self.fields[name]
        except KeyError:
            raise SchemaValidationError(
                f"unknown field {name!r}; valid: {sorted(self.fields)}",
                field=name,
                valid_names=tuple(sorted(self.fields)),
            )

    def __iter__(self) -> Iterator[FieldDecl]:
        return iter(self.fields.values())

    # validation used by Store.register ----------------------------------

    def validate_identifying_keys(self, keys: Mapping[str, Any]) -> None:
        provided = frozenset(keys)
        missing = self.identifying - provided
        extra = provided - self.identifying
        if missing or extra:
            raise SchemaValidationError(
                f"identifying keys mismatch (missing={sorted(missing)}, "
                f"extra={sorted(extra)}); expected exactly {sorted(self.identifying)}",
                missing_keys=frozenset(missing),
                extra_keys=frozenset(extra),
            )

    def validate_annotating_keys(self, keys: Mapping[str, Any]) -> None:
        provided = frozenset(keys)
        extra = provided - self.annotating
        if extra:
            raise SchemaValidationError(
                f"unknown annotating fields: {sorted(extra)}; "
                f"valid: {sorted(self.annotating)}",
                extra_keys=frozenset(extra),
            )

    def validate_value(self, decl: FieldDecl, value: Any, *, allow_none: bool) -> None:
        if value is None:
            if allow_none:
                return
            raise SchemaValidationError(
                f"field {decl.name!r}: NULL not allowed (identifying)",
                field=decl.name,
            )

        # bool/int strictness: bool is a subclass of int in Python, but the spec
        # demands they be distinct. Use `type(...) is ...` rather than isinstance.
        if decl.type == "int":
            if type(value) is not int:
                raise SchemaValidationError(
                    f"field {decl.name!r}: expected int, got {type(value).__name__}",
                    field=decl.name,
                    expected_type="int",
                    actual_type=type(value).__name__,
                )
            return
        if decl.type == "bool":
            if type(value) is not bool:
                raise SchemaValidationError(
                    f"field {decl.name!r}: expected bool, got {type(value).__name__}",
                    field=decl.name,
                    expected_type="bool",
                    actual_type=type(value).__name__,
                )
            return
        if decl.type == "float":
            # int is acceptable for float (per spec §6.1); bool is not.
            if type(value) is bool or not isinstance(value, (int, float)):
                raise SchemaValidationError(
                    f"field {decl.name!r}: expected float, got {type(value).__name__}",
                    field=decl.name,
                    expected_type="float",
                    actual_type=type(value).__name__,
                )
            if isinstance(value, float) and math.isnan(value) and decl.is_identifying:
                # NaN identity semantics break dedup; reject.
                raise SchemaValidationError(
                    f"field {decl.name!r}: NaN not allowed in identifying float",
                    field=decl.name,
                )
            return
        if decl.type in ("string", "path"):
            if not isinstance(value, str):
                raise SchemaValidationError(
                    f"field {decl.name!r}: expected str, got {type(value).__name__}",
                    field=decl.name,
                    expected_type="str",
                    actual_type=type(value).__name__,
                )
            return
        if decl.type == "datetime":
            if not isinstance(value, _dt.datetime):
                raise SchemaValidationError(
                    f"field {decl.name!r}: expected datetime.datetime, got {type(value).__name__}",
                    field=decl.name,
                    expected_type="datetime",
                    actual_type=type(value).__name__,
                )
            if value.tzinfo is None:
                # Naive datetimes are ambiguous across machines; reject per spec recommendation.
                raise SchemaValidationError(
                    f"field {decl.name!r}: naive datetime not allowed; use a tz-aware datetime",
                    field=decl.name,
                )
            return
        if decl.type == "json":
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                raise SchemaValidationError(
                    f"field {decl.name!r}: value is not JSON-serialisable: {e}",
                    field=decl.name,
                    expected_type="json",
                    actual_type=type(value).__name__,
                )
            return
        # Unreachable: every type in the catalogue is handled above.
        raise SchemaValidationError(  # pragma: no cover
            f"field {decl.name!r}: unhandled type {decl.type!r}", field=decl.name
        )


# --- dynamic model generation ------------------------------------------------


def _server_default_literal(decl: FieldDecl) -> str:
    """Convert an identifying field's `default` into a SQL DEFAULT literal.

    Only called for identifying fields, which are restricted to int/float/
    string/bool — every branch handles one of those. Bool maps to '1'/'0'
    because SQLite stores booleans as integers; floats use repr() to keep
    full precision; ints stringify; strings pass through (Alembic quotes
    them when rendering DDL).
    """
    v = decl.default
    if decl.type == "bool":
        return "1" if v else "0"
    if decl.type == "int":
        return str(int(v))
    if decl.type == "float":
        return repr(float(v))
    if decl.type == "string":
        return str(v)
    # Unreachable: identifying fields are restricted to the four types above.
    raise AssertionError(  # pragma: no cover
        f"unexpected identifying type {decl.type!r}"
    )


def _build_model(schema: Schema) -> tuple[type, type]:
    """Construct a fresh DeclarativeBase and Run class for this schema.

    A new Base is created per Schema so that Schema instances don't share
    SQLAlchemy metadata (which would conflict on identical __tablename__s).
    """

    # Each Schema gets its own DeclarativeBase subclass with isolated metadata.
    Base = type("Base", (DeclarativeBase,), {})

    attrs: dict[str, Any] = {
        "__tablename__": "runs",
        "__wallow_schema__": schema,
        "_wallow_identifying": schema.identifying,
        "id": Column(Integer, primary_key=True, autoincrement=True),
        "created_at": Column(DateTime, nullable=False, default=_utcnow),
        "updated_at": Column(
            DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
        ),
    }

    for f in schema.fields.values():
        col_kwargs: dict[str, Any] = {
            "nullable": f.nullable,
            "index": f.indexed,
            "default": f.default,
        }
        # Render server_default for identifying fields with a declared default,
        # so Alembic autogenerate emits a DDL-level DEFAULT that backfills
        # existing rows when the field is added in a later migration (spec §8.2).
        # Identifying fields are restricted to int/float/string/bool, so the
        # literal conversion is well-defined.
        if f.is_identifying and f.default is not None:
            col_kwargs["server_default"] = _server_default_literal(f)
        attrs[f.name] = Column(f.sa_type(), **col_kwargs)

    # Sort identifying field names so re-declaring them in different order in
    # the TOML doesn't generate a spurious migration.
    attrs["__table_args__"] = (
        UniqueConstraint(
            *sorted(schema.identifying), name="uq_runs_identifying"
        ),
    )

    Run = type("Run", (Base,), attrs)
    return Base, Run
