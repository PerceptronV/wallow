"""wallow — a deduplicating run registry for ML research."""

from __future__ import annotations

from .dsl import F, Expr, Field, OrderClause, Query
from .errors import (
    DuplicateRunError,
    PendingMigrationError,
    SchemaError,
    SchemaParseError,
    SchemaValidationError,
    WallowError,
)
from .migrations import (
    CollisionGroup,
    current_revision,
    find_collisions_after_drop,
    head_revision,
)
from .schema import FieldDecl, Schema, load_schema
from .store import Store, find, register

__all__ = [
    # schema
    "FieldDecl",
    "Schema",
    "load_schema",
    # store
    "Store",
    "register",
    "find",
    # dsl
    "F",
    "Field",
    "Expr",
    "Query",
    "OrderClause",
    # migrations
    "CollisionGroup",
    "current_revision",
    "find_collisions_after_drop",
    "head_revision",
    # errors
    "WallowError",
    "SchemaError",
    "SchemaParseError",
    "SchemaValidationError",
    "PendingMigrationError",
    "DuplicateRunError",
]
