"""Large-scale ML sweep gated by wallow.

Pattern
-------
Every unique combination of identifying hyperparameters maps to one experiment.
The :func:`wallow.contrib.lifecycle.run_lifecycle` helper handles the
crash-safe claim → train → finalise/fail flow for us; we only write the
training body.

For each combo:

    - if the row exists with ``status='completed'``: ``run_lifecycle`` raises
      :class:`AlreadyCompleted` and we skip.
    - otherwise: train, and ``handle.finalise(annotating={...})`` records the
      final metrics. If the body raises, the lifecycle marks the row failed
      (with a truncated error excerpt) and re-raises.

Artefacts go into the directory returned by ``store.artefacts_dir(run)``,
which is derived from ``[project].artefacts_layout`` in wallow.toml. The
layout substitutes ``{architecture}`` and ``{uuid}`` (the latter is
auto-generated and stable across reclaims, so retries overwrite the failed
attempt's files in place).

Run order:
    wallow migrate apply       # create runs.db at the head revision
    python sweep.py            # train all 96 combos
    python sweep.py            # reruns are no-ops on completed combos
"""

from __future__ import annotations

import json
import math
import random
import socket
from itertools import product
from pathlib import Path

from wallow import F, Store, load_schema
from wallow.contrib.lifecycle import AlreadyCompleted, run_lifecycle

HERE = Path(__file__).parent
SCHEMA_PATH = HERE / "wallow.toml"
DB_PATH = HERE / "runs.db"


def fake_train(identifying: dict, artefacts_dir: Path) -> dict:
    """Stand-in for a real training loop.

    Writes a few files into *artefacts_dir*, then returns a dict of metrics
    and final paths. Replace with your actual training code; the wallow
    integration around it is unchanged.
    """
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
    git_commit = "deadbeef"  # subprocess.check_output(["git", "rev-parse", "HEAD"]) in real code

    for identifying in combos:
        try:
            with run_lifecycle(
                store,
                identifying=identifying,
                start_annotating={"host": host, "git_commit": git_commit},
            ) as h:
                # The lifecycle has marked the row 'running'; uuid + dir are
                # stable across reclaim attempts.
                artefacts_dir = store.artefacts_dir(h.run, mkdir=True)
                result = fake_train(identifying, artefacts_dir)
                h.finalise(
                    annotating={
                        "best_checkpoint": result["best_checkpoint"],
                        "train_loss": result["train_loss"],
                        "val_loss": result["val_loss"],
                        "val_accuracy": result["val_accuracy"],
                        "training_curve": result["training_curve"],
                    }
                )
                trained += 1
        except AlreadyCompleted:
            skipped += 1
        except Exception:
            # Lifecycle has already written status='failed' with the excerpt;
            # we just count the failure and keep going.
            failed += 1

    print(f"done: {trained} trained, {skipped} already complete, {failed} failed")

    # Headline numbers for confidence.
    n_completed = store.where(F("status") == "completed").count()
    n_failed = store.where(F("status") == "failed").count()
    print(f"db state: {n_completed} completed, {n_failed} failed, {store.count()} total")


if __name__ == "__main__":
    main()
