"""Large-scale ML sweep gated by wallow.

Pattern
-------
Every unique combination of identifying hyperparameters maps to one experiment.
Before doing any work for a combo, we `register(..., on_duplicate="return_existing")`
and check the returned run's `status`:

    - if `status == "completed"`: skip — already done.
    - otherwise: train, write artefacts to a deterministic directory, and
      `register(..., on_duplicate="overwrite")` to record the final metrics
      and the artefacts_dir path.

This makes the dispatcher idempotent: rerunning after any kind of crash —
preempted job, OOM, network blip, even a full process kill — picks up exactly
where it left off without redoing finished work.

The artefacts_dir is derived from the identifying fields (a short hash), so a
rerun of the same combo writes to the same directory and overwrites any partial
artefacts from a failed prior attempt.

Run order:
    wallow migrate apply       # create runs.db at the head revision
    python sweep.py            # train all 96 combos
    python sweep.py            # reruns are no-ops on completed combos

The schema is alembic-managed: the initial migration is checked in under
`alembic/versions/`. To evolve (add a hyperparameter, etc.), edit wallow.toml
then `wallow migrate generate "<message>"` and review/apply the diff.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import random
import socket
import time
from itertools import product
from pathlib import Path

from wallow import F, Store, load_schema, register

HERE = Path(__file__).parent
SCHEMA_PATH = HERE / "wallow.toml"
DB_PATH = HERE / "runs.db"
ARTEFACTS_ROOT = HERE / "artefacts"


def artefacts_dir_for(identifying: dict) -> Path:
    """Deterministic per-run directory.

    A short content hash of the identifying fields keeps directory names
    stable across reruns (so we can overwrite partial artefacts) and short
    enough to fit on a typical filesystem.
    """
    payload = json.dumps(identifying, sort_keys=True).encode()
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return ARTEFACTS_ROOT / identifying["architecture"] / digest


def fake_train(identifying: dict, artefacts_dir: Path) -> dict:
    """Stand-in for a real training loop.

    Writes a few files into `artefacts_dir`, then returns a dict of metrics
    and final paths. Replace this with your actual training code; the wallow
    integration around it is unchanged.
    """
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(json.dumps(identifying, sort_keys=True))
    n_epochs = identifying["num_epochs"]
    base = 1.0 / math.sqrt(identifying["batch_size"]) + identifying["learning_rate"]

    curve = []
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(n_epochs):
        decay = math.exp(-epoch / max(n_epochs / 3, 1))
        train_loss = max(0.01, base * decay + rng.uniform(-0.02, 0.02))
        val_loss = train_loss + rng.uniform(0.02, 0.10)
        val_acc = max(0.0, min(0.99, 1.0 - val_loss + rng.uniform(-0.02, 0.02)))
        curve.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

    # Mock checkpoint files — in practice these are torch.save / safetensors.
    for epoch in range(n_epochs):
        (artefacts_dir / f"ckpt_epoch_{epoch:02d}.pt").write_bytes(b"")
    metrics_path = artefacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(curve, indent=2))

    return {
        "training_curve": curve,
        "train_loss": curve[-1]["train_loss"],
        "val_loss": curve[-1]["val_loss"],
        "val_accuracy": curve[-1]["val_acc"],
        "best_checkpoint": f"ckpt_epoch_{best_epoch:02d}.pt",
    }


def grid() -> list[dict]:
    """The hyperparameter grid. Edit this to change what the sweep covers."""
    architectures = ["resnet18", "resnet50", "vit_small"]
    optimisers = ["sgd", "adamw"]
    learning_rates = [1e-3, 3e-4]
    batch_sizes = [32, 128]
    weight_decays = [0.0, 1e-4]
    num_epochs = [10]
    seeds = [0, 1]

    combos = []
    for arch, opt, lr, bs, wd, ne, seed in product(
        architectures, optimisers, learning_rates, batch_sizes,
        weight_decays, num_epochs, seeds,
    ):
        combos.append(
            {
                "architecture": arch,
                "optimiser": opt,
                "learning_rate": lr,
                "batch_size": bs,
                "weight_decay": wd,
                "num_epochs": ne,
                "seed": seed,
            }
        )
    return combos


def main() -> None:
    schema = load_schema(SCHEMA_PATH)
    store = Store(DB_PATH, schema=schema)

    combos = grid()
    print(f"sweep grid: {len(combos)} combos")

    skipped = trained = failed = 0
    host = socket.gethostname()
    git_commit = "deadbeef"  # `subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()` in real code

    for identifying in combos:
        # 1. Dedup gate: claim the slot or read back the existing row.
        run = register(
            store,
            identifying=identifying,
            annotating={
                "status": "running",
                "host": host,
                "git_commit": git_commit,
                "started_at": dt.datetime.now(dt.timezone.utc),
            },
            on_duplicate="return_existing",
        )

        if run.status == "completed":
            skipped += 1
            continue

        # 2. Do the expensive work. Artefacts go to a deterministic directory
        #    so a rerun of this combo overwrites any partial state from before.
        artefacts_dir = artefacts_dir_for(identifying)
        t0 = time.perf_counter()
        try:
            result = fake_train(identifying, artefacts_dir)
        except Exception as e:
            register(
                store,
                identifying=identifying,
                annotating={
                    "status": "failed",
                    "error_message": f"{type(e).__name__}: {e}",
                    "wall_clock_sec": time.perf_counter() - t0,
                    "completed_at": dt.datetime.now(dt.timezone.utc),
                },
                on_duplicate="overwrite",
            )
            failed += 1
            continue

        # 3. Record success: metrics + artefacts path. `overwrite` so the row
        #    lands in a known state regardless of whether this combo had a
        #    prior `running`/`failed` attempt.
        register(
            store,
            identifying=identifying,
            annotating={
                "status": "completed",
                "artefacts_dir": str(artefacts_dir),
                "best_checkpoint": result["best_checkpoint"],
                "train_loss": result["train_loss"],
                "val_loss": result["val_loss"],
                "val_accuracy": result["val_accuracy"],
                "training_curve": result["training_curve"],
                "wall_clock_sec": time.perf_counter() - t0,
                "completed_at": dt.datetime.now(dt.timezone.utc),
            },
            on_duplicate="overwrite",
        )
        trained += 1

    print(f"done: {trained} trained, {skipped} already complete, {failed} failed")

    # Headline numbers for confidence.
    n_completed = store.where(F("status") == "completed").count()
    n_failed = store.where(F("status") == "failed").count()
    print(f"db state: {n_completed} completed, {n_failed} failed, {store.count()} total")


if __name__ == "__main__":
    main()
