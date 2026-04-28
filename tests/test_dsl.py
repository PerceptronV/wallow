"""Tests for wallow.dsl: F, Field, Expr, Query."""

from __future__ import annotations

import pytest

from wallow import (
    Expr,
    F,
    Field,
    OrderClause,
    Query,
    SchemaValidationError,
    Store,
    register,
)

from conftest import make_identifying


# ---- helpers ---------------------------------------------------------------


def _seed(store: Store) -> list:
    runs = []
    for k, sigma, val_acc, status, host in [
        (4, 0.1, 0.90, "completed", "h1"),
        (4, 0.2, 0.85, "completed", "h2"),
        (8, 0.1, 0.70, "running", "h1"),
        (8, 0.2, 0.95, "completed", "h3"),
    ]:
        runs.append(
            register(
                store,
                identifying=make_identifying(
                    cell_k=k, cell_sigma=sigma, candidate_id=k * 100 + int(sigma * 10)
                ),
                annotating={
                    "val_accuracy": val_acc,
                    "status": status,
                    "host": host,
                    "discovered_T": {"T_1100": val_acc, "nested": {"x": k}},
                },
                on_duplicate="raise",
            )
        )
    return runs


# ---- Expr / compilation ---------------------------------------------------


def test_eq_compiles_to_expr():
    expr = F("cell_k") == 4
    assert isinstance(expr, Expr)


@pytest.mark.parametrize(
    "build_expr,expected_count",
    [
        (lambda: F("cell_k") == 4, 2),
        (lambda: F("cell_k") != 4, 2),
        (lambda: F("cell_k") < 8, 2),
        (lambda: F("cell_k") <= 4, 2),
        (lambda: F("cell_k") > 4, 2),
        (lambda: F("cell_k") >= 8, 2),
    ],
)
def test_comparison_operators(memory_store: Store, build_expr, expected_count):
    _seed(memory_store)
    assert memory_store.where(build_expr()).count() == expected_count


def test_and_or_not_composition(memory_store: Store):
    _seed(memory_store)
    q = memory_store.where((F("cell_k") == 4) & (F("status") == "completed"))
    assert q.count() == 2
    q2 = memory_store.where((F("cell_k") == 4) | (F("cell_k") == 8))
    assert q2.count() == 4
    q3 = memory_store.where(~(F("status") == "completed"))
    assert q3.count() == 1


def test_in_and_not_in(memory_store: Store):
    _seed(memory_store)
    assert memory_store.where(F("cell_k").in_([4])).count() == 2
    assert memory_store.where(F("cell_k").in_((4, 8))).count() == 4
    assert memory_store.where(F("cell_k").not_in([4])).count() == 2
    # Generator works too.
    assert memory_store.where(F("cell_k").in_(x for x in [4])).count() == 2


def test_string_ops(memory_store: Store):
    _seed(memory_store)
    assert memory_store.where(F("status").contains("plet")).count() == 3
    assert memory_store.where(F("host").startswith("h")).count() == 4
    assert memory_store.where(F("host").endswith("1")).count() == 2


def test_string_op_on_non_string_raises(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="not valid on int"):
        memory_store.where(F("cell_k").contains("4")).count()


def test_is_null_and_not_null(memory_store: Store):
    register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    register(
        memory_store,
        identifying=make_identifying(seed=1),
        annotating={"status": "running"},
        on_duplicate="raise",
    )
    assert memory_store.where(F("status").is_null()).count() == 1
    assert memory_store.where(F("status").is_not_null()).count() == 1


def test_eq_none_means_is_null(memory_store: Store):
    register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    register(
        memory_store,
        identifying=make_identifying(seed=1),
        annotating={"status": "running"},
        on_duplicate="raise",
    )
    assert memory_store.where(F("status") == None).count() == 1  # noqa: E711
    assert memory_store.where(F("status") != None).count() == 1  # noqa: E711


def test_unknown_field_compile_error(memory_store: Store):
    with pytest.raises(SchemaValidationError) as ei:
        memory_store.where(F("nope") == 1).count()
    # error must list valid field names so user can self-correct
    assert ei.value.valid_names is not None
    assert "cell_k" in ei.value.valid_names


def test_compare_string_field_to_int_raises(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="cannot compare string"):
        memory_store.where(F("status") == 4).count()


# ---- JSON paths ------------------------------------------------------------


def test_json_path_compiles_and_filters(memory_store: Store):
    _seed(memory_store)
    # discovered_T.T_1100 > 0.85 should match the runs with val_acc > 0.85.
    hits = memory_store.where(F("discovered_T").json_path("T_1100") > 0.85).all()
    accs = sorted(r.val_accuracy for r in hits)
    assert accs == [0.90, 0.95]


def test_nested_json_path(memory_store: Store):
    _seed(memory_store)
    hits = memory_store.where(
        F("discovered_T").json_path("nested.x") == 8
    ).all()
    assert {r.cell_k for r in hits} == {8}


def test_json_path_on_non_json_raises(memory_store: Store):
    _seed(memory_store)
    with pytest.raises(SchemaValidationError, match="non-JSON"):
        memory_store.where(F("cell_k").json_path("a") == 1).count()


# ---- Query materialization -------------------------------------------------


def test_all(memory_store: Store):
    _seed(memory_store)
    rows = memory_store.where(F("cell_k") == 4).all()
    assert len(rows) == 2


def test_first(memory_store: Store):
    _seed(memory_store)
    row = memory_store.where(F("cell_k") == 4).first()
    assert row is not None
    assert row.cell_k == 4


def test_first_returns_none(memory_store: Store):
    assert memory_store.where(F("cell_k") == 999).first() is None


def test_one(memory_store: Store):
    register(memory_store, identifying=make_identifying(), on_duplicate="raise")
    row = memory_store.where(F("cell_k") == 4).one()
    assert row.cell_k == 4


def test_one_raises_on_zero(memory_store: Store):
    with pytest.raises(SchemaValidationError, match="no rows"):
        memory_store.where(F("cell_k") == 999).one()


def test_one_raises_on_many(memory_store: Store):
    _seed(memory_store)
    with pytest.raises(SchemaValidationError, match="found 2 rows"):
        memory_store.where(F("cell_k") == 4).one()


def test_count(memory_store: Store):
    _seed(memory_store)
    assert memory_store.count() == 4
    assert memory_store.where(F("status") == "running").count() == 1


def test_exists(memory_store: Store):
    _seed(memory_store)
    assert memory_store.where(F("cell_k") == 4).exists() is True
    assert memory_store.where(F("cell_k") == 999).exists() is False


# ---- order_by / limit / offset ---------------------------------------------


def test_order_by_desc(memory_store: Store):
    _seed(memory_store)
    rows = memory_store.where().order_by(F("val_accuracy").desc()).all()
    accs = [r.val_accuracy for r in rows]
    assert accs == sorted(accs, reverse=True)


def test_order_by_field_default_asc(memory_store: Store):
    _seed(memory_store)
    rows = memory_store.where().order_by(F("val_accuracy")).all()
    accs = [r.val_accuracy for r in rows]
    assert accs == sorted(accs)


def test_order_by_invalid_type_raises(memory_store: Store):
    with pytest.raises(TypeError, match="Field or OrderClause"):
        memory_store.where().order_by("val_accuracy")  # type: ignore[arg-type]


def test_limit_and_offset(memory_store: Store):
    _seed(memory_store)
    rows = (
        memory_store.where()
        .order_by(F("val_accuracy").desc())
        .limit(2)
        .all()
    )
    assert [r.val_accuracy for r in rows] == [0.95, 0.90]
    rows2 = (
        memory_store.where()
        .order_by(F("val_accuracy").desc())
        .limit(2)
        .offset(1)
        .all()
    )
    assert [r.val_accuracy for r in rows2] == [0.90, 0.85]


# ---- streaming iteration --------------------------------------------------


def test_iter_yields_runs(memory_store: Store):
    _seed(memory_store)
    rows = list(memory_store.where(F("cell_k") == 4))
    assert len(rows) == 2


def test_iter_uses_yield_per(memory_store: Store, monkeypatch):
    """Confirm streaming path calls yield_per (doesn't materialise via .all())."""
    _seed(memory_store)

    called = {"yield_per": False}
    from sqlalchemy.engine.result import ScalarResult

    real_yield_per = ScalarResult.yield_per

    def spy(self, n):
        called["yield_per"] = True
        return real_yield_per(self, n)

    monkeypatch.setattr(ScalarResult, "yield_per", spy)
    list(iter(memory_store.where()))
    assert called["yield_per"]


# ---- Query is immutable / chainable ---------------------------------------


def test_query_chained_returns_new_object(memory_store: Store):
    q1 = memory_store.where()
    q2 = q1.limit(5)
    assert q1 is not q2
    assert isinstance(q2, Query)


def test_field_in_dict_disallowed():
    # Hashing is disabled; using Field as a dict key should fail loudly.
    with pytest.raises(TypeError):
        {F("cell_k"): 1}  # type: ignore[misc]


# ---- Store.where / .all / .count entry points -----------------------------


def test_store_all_equivalent_to_query_all(memory_store: Store):
    _seed(memory_store)
    assert len(memory_store.all()) == 4


def test_store_count_no_filter(memory_store: Store):
    _seed(memory_store)
    assert memory_store.count() == 4


# ---- order_by accepts an OrderClause directly -----------------------------


def test_order_by_explicit_orderclause(memory_store: Store):
    _seed(memory_store)
    clause = F("val_accuracy").desc()
    assert isinstance(clause, OrderClause)
    rows = memory_store.where().order_by(clause).all()
    accs = [r.val_accuracy for r in rows]
    assert accs == sorted(accs, reverse=True)
