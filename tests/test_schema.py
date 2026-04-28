"""Tests for wallow.schema (TOML parsing + dynamic model generation)."""

from __future__ import annotations

import datetime as _dt

import pytest
from sqlalchemy import UniqueConstraint, inspect

from wallow import FieldDecl, Schema, SchemaParseError, SchemaValidationError


def test_parse_each_type(schema_from_string):
    s = schema_from_string(
        """
        [project]
        name = "alltypes"

        [identifying.x_int]
        type = "int"

        [identifying.x_float]
        type = "float"

        [identifying.x_string]
        type = "string"

        [identifying.x_bool]
        type = "bool"

        [annotating.x_json]
        type = "json"

        [annotating.x_path]
        type = "path"

        [annotating.x_datetime]
        type = "datetime"
        """
    )
    types = {f.name: f.type for f in s}
    assert types == {
        "x_int": "int",
        "x_float": "float",
        "x_string": "string",
        "x_bool": "bool",
        "x_json": "json",
        "x_path": "path",
        "x_datetime": "datetime",
    }


def test_identifying_defaults(schema_from_string):
    s = schema_from_string(
        """
        [project]
        name = "p"
        [identifying.k]
        type = "int"
        """
    )
    f = s.field("k")
    assert f.is_identifying
    assert f.indexed is True
    assert f.nullable is False


def test_annotating_defaults(schema_from_string):
    s = schema_from_string(
        """
        [project]
        name = "p"
        [identifying.k]
        type = "int"
        [annotating.note]
        type = "string"
        """
    )
    f = s.field("note")
    assert f.is_identifying is False
    assert f.indexed is False
    assert f.nullable is True


@pytest.mark.parametrize("name", ["id", "created_at", "updated_at", "_wallow_foo"])
def test_reserved_name_rejected(schema_from_string, name):
    with pytest.raises(SchemaParseError, match="reserved"):
        schema_from_string(
            f"""
            [project]
            name = "p"
            [identifying.{name}]
            type = "int"
            """
        )


def test_duplicate_field_name_rejected(schema_from_string):
    with pytest.raises(SchemaParseError, match="both"):
        schema_from_string(
            """
            [project]
            name = "p"
            [identifying.k]
            type = "int"
            [annotating.k]
            type = "string"
            """
        )


@pytest.mark.parametrize("bad_type", ["json", "path", "datetime"])
def test_identifying_with_disallowed_type_rejected(schema_from_string, bad_type):
    with pytest.raises(SchemaParseError, match="not allowed"):
        schema_from_string(
            f"""
            [project]
            name = "p"
            [identifying.k]
            type = "{bad_type}"
            """
        )


def test_no_identifying_fields_rejected(schema_from_string):
    with pytest.raises(SchemaParseError, match="at least one identifying"):
        schema_from_string(
            """
            [project]
            name = "p"
            [annotating.note]
            type = "string"
            """
        )


def test_missing_project_rejected(schema_from_string):
    with pytest.raises(SchemaParseError, match=r"\[project\]"):
        schema_from_string(
            """
            [identifying.k]
            type = "int"
            """
        )


def test_unknown_type_rejected(schema_from_string):
    with pytest.raises(SchemaParseError, match="unknown type"):
        schema_from_string(
            """
            [project]
            name = "p"
            [identifying.k]
            type = "uuid"
            """
        )


def test_default_int_to_float_coerces(schema_from_string):
    s = schema_from_string(
        """
        [project]
        name = "p"
        [identifying.k]
        type = "int"
        [identifying.x]
        type = "float"
        default = 1
        """
    )
    f = s.field("x")
    assert isinstance(f.default, float)
    assert f.default == 1.0


def test_default_wrong_type_rejected(schema_from_string):
    with pytest.raises(SchemaParseError, match="not coercible"):
        schema_from_string(
            """
            [project]
            name = "p"
            [identifying.k]
            type = "int"
            default = "zero"
            """
        )


def test_default_bool_for_int_rejected(schema_from_string):
    with pytest.raises(SchemaParseError, match="bool, not int"):
        schema_from_string(
            """
            [project]
            name = "p"
            [identifying.k]
            type = "int"
            default = true
            """
        )


def test_identifying_cannot_be_nullable(schema_from_string):
    with pytest.raises(SchemaParseError, match="NOT NULL"):
        schema_from_string(
            """
            [project]
            name = "p"
            [identifying.k]
            type = "int"
            nullable = true
            """
        )


def test_unknown_field_key_rejected(schema_from_string):
    with pytest.raises(SchemaParseError, match="unknown keys"):
        schema_from_string(
            """
            [project]
            name = "p"
            [identifying.k]
            type = "int"
            primary = true
            """
        )


def test_run_columns_match_schema(example_schema: Schema):
    Run = example_schema.Run
    cols = {c.name for c in Run.__table__.columns}
    assert {"id", "created_at", "updated_at"}.issubset(cols)
    assert example_schema.identifying <= cols
    assert example_schema.annotating <= cols


def test_unique_constraint_uses_sorted_identifying(example_schema: Schema):
    Run = example_schema.Run
    uniques = [
        c for c in Run.__table__.constraints if isinstance(c, UniqueConstraint)
    ]
    target = [u for u in uniques if u.name == "uq_runs_identifying"]
    assert len(target) == 1
    cols = [c.name for c in target[0].columns]
    assert cols == sorted(example_schema.identifying)


def test_indexes_on_identifying_and_indexed_annotating(example_schema: Schema):
    Run = example_schema.Run
    indexed_cols = {idx.columns.keys()[0] for idx in Run.__table__.indexes}
    # All identifying fields indexed by default.
    assert example_schema.identifying <= indexed_cols
    # Spot-check a few annotating fields.
    assert "val_accuracy" in indexed_cols
    assert "wall_clock_sec" not in indexed_cols  # not marked indexed in fixture


def test_schema_field_lookup_unknown_raises(example_schema: Schema):
    with pytest.raises(SchemaValidationError, match="unknown field"):
        example_schema.field("nope")


def test_two_loads_isolated_metadata(schema_from_string):
    s1 = schema_from_string(
        """
        [project]
        name = "p"
        [identifying.k]
        type = "int"
        """
    )
    s2 = schema_from_string(
        """
        [project]
        name = "p2"
        [identifying.k]
        type = "int"
        """
    )
    # Each schema gets its own Base/metadata so the second build doesn't clash.
    assert s1.Run is not s2.Run
    assert s1.Run.__table__.metadata is not s2.Run.__table__.metadata


def test_fielddecl_is_frozen():
    f = FieldDecl(
        name="x", type="int", is_identifying=True,
        indexed=True, nullable=False, default=None, doc=None,
    )
    with pytest.raises(Exception):
        f.name = "y"  # type: ignore[misc]


# ---- server_default rendering (Phase 3 prep) -------------------------------


def _decl(name: str, type_: str, default, is_identifying: bool = True) -> FieldDecl:
    return FieldDecl(
        name=name, type=type_, is_identifying=is_identifying,
        indexed=True, nullable=False, default=default, doc=None,
    )


@pytest.mark.parametrize(
    "type_, value, expected",
    [
        ("int", 0, "0"),
        ("int", 42, "42"),
        ("int", -7, "-7"),
        ("bool", True, "1"),
        ("bool", False, "0"),
        ("float", 0.0, "0.0"),
        ("float", 0.1, "0.1"),
        ("string", "hello", "hello"),
        ("string", "", ""),
    ],
)
def test_server_default_literal(type_, value, expected):
    from wallow.schema import _server_default_literal
    assert _server_default_literal(_decl("k", type_, value)) == expected


def test_server_default_literal_float_preserves_precision():
    """str(0.1+0.2) loses precision in some Python builds; repr does not."""
    from wallow.schema import _server_default_literal
    v = 0.1 + 0.2
    out = _server_default_literal(_decl("k", "float", v))
    # Round-tripping repr must reproduce the original float exactly.
    assert float(out) == v


def test_server_default_rendered_on_identifying_with_default(schema_from_string):
    s = schema_from_string(
        """
        [project]
        name = "p"
        [identifying.seed]
        type = "int"
        default = 0
        [identifying.k]
        type = "int"
        """
    )
    seed_col = s.Run.__table__.columns["seed"]
    k_col = s.Run.__table__.columns["k"]
    assert seed_col.server_default is not None
    assert seed_col.server_default.arg == "0"
    # No default declared → no server_default.
    assert k_col.server_default is None


def test_no_server_default_on_annotating_with_default(schema_from_string):
    s = schema_from_string(
        """
        [project]
        name = "p"
        [identifying.k]
        type = "int"
        [annotating.note]
        type = "string"
        default = "hello"
        """
    )
    note_col = s.Run.__table__.columns["note"]
    # Annotating fields are nullable, so they don't need a DDL-level default
    # for backfill — only the Python-side `default=` is set.
    assert note_col.server_default is None
    assert note_col.default is not None  # python-side default still wired


def test_server_default_for_bool_identifying(schema_from_string):
    s = schema_from_string(
        """
        [project]
        name = "p"
        [identifying.k]
        type = "int"
        [identifying.flag]
        type = "bool"
        default = true
        """
    )
    flag_col = s.Run.__table__.columns["flag"]
    assert flag_col.server_default.arg == "1"
