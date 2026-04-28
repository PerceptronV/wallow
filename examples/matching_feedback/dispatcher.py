"""Toy dispatcher for the matching-feedback meta-learning loop.

Walks a small grid of (cell_k, cell_sigma, generation, candidate_id, seed)
tuples, registers each via `wallow.register(..., on_duplicate="return_existing")`,
then "trains" the candidate (here: synthesizes a deterministic val_loss /
val_accuracy from the inputs) and overwrites the run with the result.

The `return_existing` policy is what makes this resume-safe: rerunning the
script after a crash skips work that's already done.

Run order:
    wallow migrate apply        # create runs.db at head revision
    python dispatcher.py        # populate
    python analysis.py          # query via the DSL
"""

from __future__ import annotations

import datetime as dt
import math
import random
import socket
from itertools import product
from pathlib import Path

from wallow import Store, load_schema, register

HERE = Path(__file__).parent
SCHEMA_PATH = HERE / "wallow.toml"
DB_PATH = HERE / "runs.db"


def fake_train(cell_k: int, cell_sigma: float, seed: int) -> tuple[float, float, float]:
    """Stand-in for the inner training loop. Returns (val_loss, val_acc, wall_clock_sec)."""
    rng = random.Random(f"{cell_k}-{round(cell_sigma * 1000)}-{seed}")
    base_loss = 1.0 / math.sqrt(cell_k) + cell_sigma + rng.uniform(-0.05, 0.05)
    val_loss = max(0.0, base_loss)
    val_accuracy = max(0.0, min(1.0, 1.0 - val_loss + rng.uniform(-0.02, 0.02)))
    wall_clock_sec = 30.0 + rng.uniform(0, 60)
    return val_loss, val_accuracy, wall_clock_sec


def main() -> None:
    schema = load_schema(SCHEMA_PATH)
    store = Store(DB_PATH, schema=schema)

    cell_ks = [2, 4, 8]
    cell_sigmas = [0.05, 0.1, 0.2]
    generations = [0, 1]
    candidate_ids = list(range(4))
    seeds = [0, 1]

    new_runs = 0
    skipped_runs = 0

    for cell_k, cell_sigma, gen, cand, seed in product(
        cell_ks, cell_sigmas, generations, candidate_ids, seeds
    ):
        identifying = {
            "cell_k": cell_k,
            "cell_sigma": cell_sigma,
            "generation": gen,
            "candidate_id": cand,
            "seed": seed,
            "warmup_steps": 0,
        }
        run = register(
            store,
            identifying=identifying,
            annotating={
                "status": "running",
                "host": socket.gethostname(),
                "git_commit": "deadbeef",
                "started_at": dt.datetime.now(dt.timezone.utc),
            },
            on_duplicate="return_existing",
        )
        if run.status != "running":
            skipped_runs += 1
            continue

        # "Train" — in a real pipeline this is the expensive inner loop.
        val_loss, val_accuracy, wall_clock_sec = fake_train(cell_k, cell_sigma, seed)

        # Mark the run completed with metrics. Use overwrite for the final
        # write so the run lands in a known state regardless of prior status.
        register(
            store,
            identifying=identifying,
            annotating={
                "status": "completed",
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "wall_clock_sec": wall_clock_sec,
                "discovered_T": {f"T_{cell_k}{cell_k}": round(val_accuracy, 3)},
            },
            on_duplicate="overwrite",
        )
        new_runs += 1

    print(f"completed: {new_runs} new, {skipped_runs} already done")


if __name__ == "__main__":
    main()
