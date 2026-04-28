# ml_sweep — wallow as a dedup gate for ML sweeps

A self-contained example showing the canonical pattern: every unique combination
of hyperparameters defines one experiment, training artefacts land in a
deterministic directory, and the dispatcher is idempotent — rerun it after any
crash and finished combos are skipped.

## What it shows

- **Identifying = dedup key, annotating = everything else.** The seven
  identifying fields (`architecture`, `optimiser`, `learning_rate`, `batch_size`,
  `weight_decay`, `num_epochs`, `seed`) form a composite UNIQUE constraint. The
  artefacts directory, metrics, status, training curve, host, git commit — all
  annotating.
- **Artefacts paths as annotating data.** `artefacts_dir` is declared as
  `type = "path"` (a typed string with semantic meaning) and points to a
  directory derived from a hash of the identifying fields. Same combo → same
  directory, on every host.
- **The two-call resume-safe pattern.**
  `register(..., on_duplicate="return_existing")` before training is the dedup
  gate; `register(..., on_duplicate="overwrite")` after training records the
  final state. A crash between them leaves a `running` row that the next
  dispatch picks up and re-runs.
- **Alembic-managed schema.** The initial migration is checked in. Adding a
  new hyperparameter dimension later is one `wallow migrate generate` away,
  and existing rows backfill via the field's `default` (see
  [Evolving the schema](#evolving-the-schema) below). This is the whole point
  of using wallow over an ad-hoc SQLite table.
- **DSL queries.** `analyse.py` shows the analysis patterns a researcher uses
  while a sweep is still in flight: progress count, best run, per-architecture
  leader, failure inspection, direct identifying-tuple lookup.

## Layout

```
examples/ml_sweep/
├── wallow.toml                schema (7 identifying + 13 annotating fields)
├── alembic.ini                anchored to this directory
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   ├── versions/
│   │   └── ..._initial_schema.py
│   └── snapshots/
│       └── <rev>.toml         wallow.toml as of the revision
├── sweep.py                   idempotent dispatcher
├── analyse.py                 DSL queries
└── README.md
```

`runs.db` and `artefacts/` are created at runtime and gitignored.

## End-to-end

From this directory:

```bash
wallow migrate apply        # create runs.db at the head revision
python sweep.py             # train 96 combos
python sweep.py             # second run: every combo "already complete"
python analyse.py           # progress + winners + failures + direct lookup
```

To inspect the migration state at any point:

```bash
wallow migrate history      # revisions, current marked with *
wallow status               # current rev, head rev, run count
```

## The pattern in 12 lines

```python
run = register(
    store,
    identifying=combo,
    annotating={"status": "running", "started_at": now()},
    on_duplicate="return_existing",
)
if run.status == "completed":
    continue                 # already done — skip the expensive work

result = train(combo, artefacts_dir_for(combo))   # write artefacts to disk

register(
    store,
    identifying=combo,
    annotating={"status": "completed", "artefacts_dir": ..., "val_loss": ..., ...},
    on_duplicate="overwrite",
)
```

The `register → train → register` triple is the only wallow-specific code in
the loop; everything else is your training code unchanged.

## Evolving the schema

The whole reason to use wallow over a hand-rolled SQLite table is that real ML
projects always grow new hyperparameters. Suppose mid-sweep you decide
`dropout` is worth ablating. The flow is:

1. Edit `wallow.toml`:

   ```toml
   [identifying.dropout]
   type = "float"
   default = 0.0
   ```

   The `default` is required for any new identifying field — it's what backfills
   existing rows so the migration applies cleanly to a non-empty table. wallow
   aborts `migrate generate` with a clear error if you forget.

2. Generate and apply the migration:

   ```bash
   wallow migrate generate "add dropout"
   # review alembic/versions/<rev>_add_dropout.py
   wallow migrate apply
   ```

   All existing 96 runs now have `dropout=0.0`. They're still valid; their
   identifying tuple is the same one they had before, just with the new field
   added. New combos including non-default `dropout` values are new runs and
   will be picked up by the dispatcher's resume-safe gate.

3. Update `sweep.py`'s grid to include `dropout`, rerun. Existing runs are
   skipped; new combos train.

For an example of the two-migration evolution flow, see
[`../matching_feedback/`](../matching_feedback/) — it walks through adding
`warmup_steps` to a populated DB.

## Scaling up

- **Many workers, one DB**: SQLite + WAL handles a few concurrent writers fine.
  Each worker reads the grid, calls `register(..., return_existing)`, and only
  does work for combos whose `status != "completed"`. Race-on-claim resolves at
  the DB via the UNIQUE constraint.
- **Distributed workers**: put `runs.db` on a shared filesystem, or move to
  Postgres by changing the `sqlalchemy.url` in `alembic.ini` (the abstraction
  supports it; not currently tested).
