"""Expression DSL: F, Field, Expr, OrderClause, Query.

`F("name")` produces a `Field`. Comparison operators on `Field` and the
boolean operators (`&`, `|`, `~`) on `Expr` build an internal AST that's
compiled to SQLAlchemy when a `Query` is materialized.

Field name resolution is deferred until compile time, where the
`Run.__wallow_schema__` back-reference provides the type catalogue.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Iterator, Literal, TYPE_CHECKING, Union

from sqlalchemy import and_, func, not_, or_, select
from sqlalchemy.sql import ColumnElement

from .errors import SchemaValidationError

if TYPE_CHECKING:
    from .schema import Schema
    from .store import Store


# --- AST nodes (private) ----------------------------------------------------


@dataclass(frozen=True)
class _FieldRef:
    name: str
    json_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Compare:
    op: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    field: _FieldRef
    value: Any


@dataclass(frozen=True)
class _In:
    field: _FieldRef
    values: tuple[Any, ...]
    negate: bool


@dataclass(frozen=True)
class _StringOp:
    op: Literal["contains", "startswith", "endswith"]
    field: _FieldRef
    value: str


@dataclass(frozen=True)
class _IsNull:
    field: _FieldRef
    negate: bool


@dataclass(frozen=True)
class _And:
    children: tuple["_Node", ...]


@dataclass(frozen=True)
class _Or:
    children: tuple["_Node", ...]


@dataclass(frozen=True)
class _Not:
    child: "_Node"


_Node = Union[_Compare, _In, _StringOp, _IsNull, _And, _Or, _Not]


# --- public Field / Expr -----------------------------------------------------


def F(name: str) -> "Field":
    """Create a field reference. Resolved against the schema at compile time."""
    return Field(name)


class Field:
    """A named field reference. Supports comparison, JSON path, ordering."""

    # Field has overridden __eq__/__ne__ that return Expr (not bool) — must
    # disable hashing to avoid accidental dict/set membership.
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, name: str, json_path: tuple[str, ...] = ()) -> None:
        self.name = name
        self._json_path = json_path

    def _ref(self) -> _FieldRef:
        return _FieldRef(self.name, self._json_path)

    # comparisons -------------------------------------------------------

    def __eq__(self, other: Any) -> "Expr":  # type: ignore[override]
        if other is None:
            return Expr(_IsNull(self._ref(), negate=False))
        return Expr(_Compare("eq", self._ref(), other))

    def __ne__(self, other: Any) -> "Expr":  # type: ignore[override]
        if other is None:
            return Expr(_IsNull(self._ref(), negate=True))
        return Expr(_Compare("ne", self._ref(), other))

    def __lt__(self, other: Any) -> "Expr":
        return Expr(_Compare("lt", self._ref(), other))

    def __le__(self, other: Any) -> "Expr":
        return Expr(_Compare("le", self._ref(), other))

    def __gt__(self, other: Any) -> "Expr":
        return Expr(_Compare("gt", self._ref(), other))

    def __ge__(self, other: Any) -> "Expr":
        return Expr(_Compare("ge", self._ref(), other))

    # set membership ----------------------------------------------------

    def in_(self, values: Iterable[Any]) -> "Expr":
        return Expr(_In(self._ref(), tuple(values), negate=False))

    def not_in(self, values: Iterable[Any]) -> "Expr":
        return Expr(_In(self._ref(), tuple(values), negate=True))

    # string ops --------------------------------------------------------

    def contains(self, substr: str) -> "Expr":
        return Expr(_StringOp("contains", self._ref(), substr))

    def startswith(self, prefix: str) -> "Expr":
        return Expr(_StringOp("startswith", self._ref(), prefix))

    def endswith(self, suffix: str) -> "Expr":
        return Expr(_StringOp("endswith", self._ref(), suffix))

    # null checks -------------------------------------------------------

    def is_null(self) -> "Expr":
        return Expr(_IsNull(self._ref(), negate=False))

    def is_not_null(self) -> "Expr":
        return Expr(_IsNull(self._ref(), negate=True))

    # JSON path ---------------------------------------------------------

    def json_path(self, path: str) -> "Field":
        if not path:
            raise ValueError("json_path requires a non-empty path")
        parts = tuple(path.split("."))
        return Field(self.name, self._json_path + parts)

    # ordering ----------------------------------------------------------

    def asc(self) -> "OrderClause":
        return OrderClause(self._ref(), "asc")

    def desc(self) -> "OrderClause":
        return OrderClause(self._ref(), "desc")


class Expr:
    """A boolean expression. Composable via `&`, `|`, `~`."""

    def __init__(self, node: _Node) -> None:
        self._node = node

    def __and__(self, other: "Expr") -> "Expr":
        return Expr(_And((self._node, other._node)))

    def __or__(self, other: "Expr") -> "Expr":
        return Expr(_Or((self._node, other._node)))

    def __invert__(self) -> "Expr":
        return Expr(_Not(self._node))

    def compile(self, model_class: type) -> ColumnElement:
        return _compile(self._node, model_class)


@dataclass(frozen=True)
class OrderClause:
    field: _FieldRef
    direction: Literal["asc", "desc"]

    def compile(self, model_class: type) -> ColumnElement:
        col = _resolve_column(self.field, model_class)
        return col.asc() if self.direction == "asc" else col.desc()


# --- compilation -------------------------------------------------------------


def _resolve_column(ref: _FieldRef, model_class: type) -> Any:
    schema: "Schema" = getattr(model_class, "__wallow_schema__")
    if ref.name not in schema.fields:
        valid = tuple(sorted(schema.fields))
        raise SchemaValidationError(
            f"unknown field {ref.name!r}; valid: {list(valid)}",
            field=ref.name,
            valid_names=valid,
        )
    decl = schema.field(ref.name)
    column = getattr(model_class, ref.name)
    if ref.json_path:
        if decl.type != "json":
            raise SchemaValidationError(
                f"json_path used on non-JSON field {ref.name!r} (type={decl.type})",
                field=ref.name,
            )
        path_str = "$." + ".".join(ref.json_path)
        return func.json_extract(column, path_str)
    return column


def _coerce_value(decl_type: str, value: Any, *, field_name: str) -> Any:
    """Coerce / validate the RHS of a comparison.

    Special cases per spec: bool→{0,1} round trip is allowed only when the
    RHS is bool and the field is bool, or RHS is 0/1 and the field is bool.
    For float fields, ints are accepted and coerced.
    """
    if decl_type == "bool":
        if type(value) is bool:
            return value
        if value in (0, 1):
            return bool(value)
        raise SchemaValidationError(
            f"field {field_name!r}: cannot compare bool field to {value!r}",
            field=field_name,
            expected_type="bool",
            actual_type=type(value).__name__,
        )
    if decl_type == "int":
        if type(value) is int:
            return value
        raise SchemaValidationError(
            f"field {field_name!r}: cannot compare int field to {type(value).__name__}",
            field=field_name,
            expected_type="int",
            actual_type=type(value).__name__,
        )
    if decl_type == "float":
        if type(value) is bool:
            raise SchemaValidationError(
                f"field {field_name!r}: cannot compare float field to bool",
                field=field_name,
                expected_type="float",
                actual_type="bool",
            )
        if isinstance(value, (int, float)):
            return float(value)
        raise SchemaValidationError(
            f"field {field_name!r}: cannot compare float field to {type(value).__name__}",
            field=field_name,
            expected_type="float",
            actual_type=type(value).__name__,
        )
    if decl_type in ("string", "path"):
        if not isinstance(value, str):
            raise SchemaValidationError(
                f"field {field_name!r}: cannot compare string field to "
                f"{type(value).__name__}",
                field=field_name,
                expected_type="str",
                actual_type=type(value).__name__,
            )
        return value
    if decl_type == "datetime":
        if not isinstance(value, _dt.datetime):
            raise SchemaValidationError(
                f"field {field_name!r}: cannot compare datetime field to "
                f"{type(value).__name__}",
                field=field_name,
                expected_type="datetime",
                actual_type=type(value).__name__,
            )
        return value
    # JSON: any value passes (json_extract result vs a Python literal).
    return value


_OP_MAP = {
    "eq": lambda c, v: c == v,
    "ne": lambda c, v: c != v,
    "lt": lambda c, v: c < v,
    "le": lambda c, v: c <= v,
    "gt": lambda c, v: c > v,
    "ge": lambda c, v: c >= v,
}


def _compile(node: _Node, model_class: type) -> ColumnElement:
    schema: "Schema" = getattr(model_class, "__wallow_schema__")

    if isinstance(node, _Compare):
        col = _resolve_column(node.field, model_class)
        decl = schema.field(node.field.name)
        # On a JSON path, the column is a json_extract() expression of arbitrary
        # type; we accept the RHS as-is.
        decl_type = "json" if node.field.json_path else decl.type
        v = _coerce_value(decl_type, node.value, field_name=node.field.name)
        return _OP_MAP[node.op](col, v)

    if isinstance(node, _In):
        col = _resolve_column(node.field, model_class)
        decl = schema.field(node.field.name)
        decl_type = "json" if node.field.json_path else decl.type
        coerced = tuple(
            _coerce_value(decl_type, v, field_name=node.field.name)
            for v in node.values
        )
        expr = col.in_(coerced)
        return not_(expr) if node.negate else expr

    if isinstance(node, _StringOp):
        decl = schema.field(node.field.name)
        if decl.type not in ("string", "path"):
            raise SchemaValidationError(
                f"field {node.field.name!r}: string operator {node.op!r} "
                f"not valid on {decl.type} field",
                field=node.field.name,
            )
        if not isinstance(node.value, str):
            raise SchemaValidationError(
                f"field {node.field.name!r}: {node.op} requires str RHS, "
                f"got {type(node.value).__name__}",
                field=node.field.name,
            )
        col = _resolve_column(node.field, model_class)
        if node.op == "contains":
            return col.contains(node.value)
        if node.op == "startswith":
            return col.startswith(node.value)
        if node.op == "endswith":
            return col.endswith(node.value)
        raise AssertionError(f"unhandled string op {node.op!r}")  # pragma: no cover

    if isinstance(node, _IsNull):
        col = _resolve_column(node.field, model_class)
        return col.isnot(None) if node.negate else col.is_(None)

    if isinstance(node, _And):
        return and_(*(_compile(c, model_class) for c in node.children))

    if isinstance(node, _Or):
        return or_(*(_compile(c, model_class) for c in node.children))

    if isinstance(node, _Not):
        return not_(_compile(node.child, model_class))

    raise AssertionError(f"unhandled AST node: {node!r}")  # pragma: no cover


# --- Query --------------------------------------------------------------------


@dataclass(frozen=True)
class _QueryState:
    where_exprs: tuple[Expr, ...] = ()
    order_clauses: tuple[OrderClause, ...] = ()
    limit: int | None = None
    offset: int | None = None


class Query:
    """A pending query. Lazily executed; chained calls return new Query objects."""

    def __init__(
        self,
        store: "Store",
        *,
        state: _QueryState | None = None,
    ) -> None:
        self._store = store
        self._state = state or _QueryState()

    def _replace(self, **kwargs: Any) -> "Query":
        new_state = _QueryState(
            where_exprs=kwargs.get("where_exprs", self._state.where_exprs),
            order_clauses=kwargs.get("order_clauses", self._state.order_clauses),
            limit=kwargs.get("limit", self._state.limit),
            offset=kwargs.get("offset", self._state.offset),
        )
        return Query(self._store, state=new_state)

    # builder ----------------------------------------------------------

    def where(self, *exprs: Expr) -> "Query":
        return self._replace(where_exprs=self._state.where_exprs + tuple(exprs))

    def order_by(self, *clauses: OrderClause | Field) -> "Query":
        normalized: list[OrderClause] = []
        for c in clauses:
            if isinstance(c, Field):
                normalized.append(c.asc())
            elif isinstance(c, OrderClause):
                normalized.append(c)
            else:
                raise TypeError(
                    f"order_by accepts Field or OrderClause, got {type(c).__name__}"
                )
        return self._replace(
            order_clauses=self._state.order_clauses + tuple(normalized)
        )

    def limit(self, n: int) -> "Query":
        return self._replace(limit=n)

    def offset(self, n: int) -> "Query":
        return self._replace(offset=n)

    # internal: build SQLAlchemy stmt --------------------------------

    def _build_select(self) -> Any:
        Run = self._store.schema.Run
        stmt = select(Run)
        for e in self._state.where_exprs:
            stmt = stmt.where(e.compile(Run))
        for c in self._state.order_clauses:
            stmt = stmt.order_by(c.compile(Run))
        if self._state.limit is not None:
            stmt = stmt.limit(self._state.limit)
        if self._state.offset is not None:
            stmt = stmt.offset(self._state.offset)
        return stmt

    # materialization -----------------------------------------------

    def all(self) -> list[Any]:
        with self._store.session() as s:
            return list(s.scalars(self._build_select()).all())

    def first(self) -> Any | None:
        # `first()` doesn't care about explicit limit; if the user didn't
        # pin one, restrict to LIMIT 1 to avoid loading the world.
        q = self if self._state.limit is not None else self.limit(1)
        with self._store.session() as s:
            return s.scalars(q._build_select()).first()

    def one(self) -> Any:
        with self._store.session() as s:
            results = list(s.scalars(self._build_select()).all())
        if len(results) == 0:
            raise SchemaValidationError("Query.one() found no rows")
        if len(results) > 1:
            raise SchemaValidationError(
                f"Query.one() found {len(results)} rows; expected exactly 1"
            )
        return results[0]

    def count(self) -> int:
        from sqlalchemy import func as _f

        Run = self._store.schema.Run
        # Build a count(*) variant that respects where + (limit/offset)?
        # Spec: count() returns the total matching the where clause, ignoring
        # limit/offset. That matches "WHERE … COUNT(*)" semantics.
        stmt = select(_f.count()).select_from(Run)
        for e in self._state.where_exprs:
            stmt = stmt.where(e.compile(Run))
        with self._store.session() as s:
            return int(s.scalar(stmt) or 0)

    def exists(self) -> bool:
        return self.first() is not None

    def __iter__(self) -> Iterator[Any]:
        # Streaming: hold the session open while the generator advances.
        # We open it eagerly so a caller iterating partway and abandoning
        # the generator still gets cleanup via session.__exit__ when the
        # generator is GC'd.
        return self._stream()

    def _stream(self) -> Iterator[Any]:
        s = self._store._session_factory()
        try:
            result = s.scalars(self._build_select()).yield_per(100)
            for row in result:
                yield row
        finally:
            s.close()
