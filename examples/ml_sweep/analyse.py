"""Sweep analysis using the wallow DSL.

After `sweep.py` has populated runs.db, this script answers questions a
researcher would ask while the sweep is still running:

    - How far along is the sweep?
    - Which architecture is winning so far?
    - What's the best run, and where are its artefacts?
    - Which combos failed, and why?

Every query goes through the DSL, so field names are validated at compile
time and typos surface as `SchemaValidationError` instead of empty results.
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
    failed = store.where(F("status") == "failed").count()
    pending = total - completed - failed
    print(f"sweep progress: {completed}/{total} completed  ({failed} failed, {pending} not done)")

    # Best run overall — by val_accuracy, ties broken by val_loss.
    best = (
        store.where(F("status") == "completed")
        .order_by(F("val_accuracy").desc(), F("val_loss").asc())
        .first()
    )
    if best is not None:
        print(
            f"\nbest run id={best.id}: "
            f"{best.architecture}/{best.optimiser} lr={best.learning_rate} bs={best.batch_size}"
            f"  acc={best.val_accuracy:.4f} loss={best.val_loss:.4f}"
        )
        # artefacts_dir is no longer an annotation — derive it on the fly.
        print(f"  artefacts: {store.artefacts_dir(best)}/{best.best_checkpoint}")

    # Per-architecture top run. Useful when comparing model families.
    print("\ntop val_accuracy per architecture:")
    for arch in ("resnet18", "resnet50", "vit_small"):
        top = (
            store.where((F("architecture") == arch) & (F("status") == "completed"))
            .order_by(F("val_accuracy").desc())
            .first()
        )
        if top is not None:
            print(
                f"  {arch:10}  acc={top.val_accuracy:.4f}  "
                f"opt={top.optimiser} lr={top.learning_rate} bs={top.batch_size} wd={top.weight_decay}"
            )

    # Failed runs — pull error messages so you can decide what to retry.
    failures = store.where(F("status") == "failed").all()
    if failures:
        print(f"\nfailures ({len(failures)}):")
        for r in failures[:5]:
            print(f"  id={r.id} {r.architecture}/{r.optimiser} lr={r.learning_rate}: {r.error_excerpt}")

    # Combos worth picking out for downstream use — adamw runs at lr<=1e-3
    # whose best checkpoint is a late epoch (we read into the JSON curve).
    late_winners = (
        store.where(
            (F("optimiser") == "adamw")
            & (F("learning_rate") <= 1e-3)
            & (F("status") == "completed")
        )
        .order_by(F("val_accuracy").desc())
        .limit(5)
        .all()
    )
    if late_winners:
        print("\ntop adamw runs (lr<=1e-3):")
        for r in late_winners:
            print(f"  acc={r.val_accuracy:.4f}  bs={r.batch_size} wd={r.weight_decay} seed={r.seed}")

    # Direct lookup by full identifying tuple — useful inside other scripts
    # that already know which combo they want.
    one = find(
        store,
        architecture="resnet18",
        optimiser="adamw",
        learning_rate=1e-3,
        batch_size=128,
        weight_decay=0.0,
        num_epochs=10,
        seed=0,
    )
    if one is not None:
        print(f"\nfind() lookup: id={one.id} status={one.status} acc={one.val_accuracy}")


if __name__ == "__main__":
    main()
