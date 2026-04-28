"""Downstream analysis using the wallow DSL.

Demonstrates the patterns a notebook or plotting script would use:
    - Boolean composition with `&`, `|`, `~` and parenthesised comparisons.
    - `order_by` + `limit` for top-k.
    - `count()` and `exists()`.
    - JSON path indexing into an annotating json field.
    - `find()` for direct identifying-key lookup.

Run after `dispatcher.py` has populated the database.
"""

from __future__ import annotations

from pathlib import Path

from wallow import F, Store, find, load_schema

HERE = Path(__file__).parent
SCHEMA_PATH = HERE / "wallow.toml"
DB_PATH = HERE / "runs.db"


def main() -> None:
    schema = load_schema(SCHEMA_PATH)
    store = Store(DB_PATH, schema=schema)

    total = store.count()
    completed = store.where(F("status") == "completed").count()
    print(f"runs: {total} total, {completed} completed")

    # Top-10 by val_accuracy among completed cell_k=4 runs.
    top = (
        store.where((F("cell_k") == 4) & (F("status") == "completed"))
             .order_by(F("val_accuracy").desc())
             .limit(10)
             .all()
    )
    print(f"\ntop-10 cell_k=4 completed runs by val_accuracy:")
    for r in top:
        print(
            f"  k={r.cell_k} sigma={r.cell_sigma} gen={r.generation} "
            f"cand={r.candidate_id} seed={r.seed}  "
            f"acc={r.val_accuracy:.3f} loss={r.val_loss:.3f}"
        )

    # Set membership + range over an annotating float.
    high_loss_ks = (
        store.where(
            F("cell_k").in_([2, 4]) & (F("val_loss") > 0.5)
        )
        .count()
    )
    print(f"\ncell_k in (2,4) with val_loss > 0.5: {high_loss_ks}")

    # JSON-path query: find runs whose discovered_T['T_44'] is set.
    has_T44 = (
        store.where(F("discovered_T").json_path("T_44").is_not_null())
             .count()
    )
    print(f"runs with discovered_T.T_44 present: {has_T44}")

    # Direct identifying-key lookup. Useful when you know the exact run id.
    one = find(
        store,
        cell_k=4, cell_sigma=0.1, generation=0,
        candidate_id=0, seed=0, warmup_steps=0,
    )
    if one is not None:
        print(f"\nfound run id={one.id}: status={one.status} acc={one.val_accuracy}")


if __name__ == "__main__":
    main()
