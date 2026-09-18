# casmi26

Entry for the Kaggle competition Enveda CASMI 2026 (predict SMILES from LC-MS/MS spectra, metric MRR@25 on tautomer-canonical InChIKey14). Runs locally on Windows for validation, submitted as an offline Kaggle kernel. Public repo: https://github.com/shiferaxa/casmi26 (branch main, Apache 2.0, see NOTICE for the forked public kernel attribution). The repo is public during the competition; anything that must stay private (data, assets, tokens) is gitignored.

## Warnings

- `data/` and `artifacts/` are gitignored. Data comes from `kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data`. Never commit it.
- Direct CSV submissions are rejected (code competition). Submit by pushing the kernel in `kaggle/kernel/`, then `kaggle competitions submit -k amhashiferaw/casmi26-library-search -v <version> -f submission.csv -m "..."`.
- The Kaggle image has no RDKit and internet is off. The kernel installs it from the private dataset amhashiferaw/rdkit-wheel-cp312 (wheel in `kaggle/rdkit_wheel/`). Source code ships as the private dataset amhashiferaw/casmi26-src, a flat copy of `src/casmi/*.py` in `kaggle/dataset/`. After editing `src/casmi`, copy the files there and run `kaggle datasets version -p . -m "..."` from inside that folder.
- Kaggle CLI 2.2.4 gotchas: run `datasets create`, `datasets version`, and `kernels push` from inside the folder with `-p .` (a relative folder path crashes on a temp-dir bug). Wait for `kaggle datasets status <id>` to say ready before pushing a kernel that depends on a new dataset version, or the kernel sees an empty folder.
- Kaggle auth is the access token file in the user's .kaggle folder (username amhashiferaw). It is never committed; `data/`, `artifacts/`, and dataset payloads under `kaggle/` are gitignored.
- No CUDA on this machine (AMD integrated GPU). Train models on Kaggle GPU, not here. Submission kernels run with GPU off (same 65 min runtime, fork_v4_cpu is the template); the 6 weekly GPU hours are for training only.
- Paths handed to Python must be Windows style (`C:/Users/...`). Git Bash style `/c/Users/...` works in bash tests and `ls` but Python cannot open it, and the failure only shows as FileNotFoundError deep in numpy. Also use absolute paths in background shell chains, the working directory drifts between calls.
- Windows multiprocessing spawns fresh workers, so module globals set at runtime (like `core.BITS`) are None inside a Pool unless passed through an initializer. The public notebook relies on Linux fork semantics. See `_worker_init` in `src/casmi/fork/harness.py`. Also delete `data/cache/train_fp.npz` whenever fingerprint code changes, the cache is not keyed on it.
- `src/casmi/fork/core.py` is a transplant of the public 0.339 kernel (`kaggle/fork_v2/notebook.py`). Change numerics in the notebook first, then re-transplant, so Kaggle and local stay identical.
- Validation must group by InChIKey14 and hold out by instrument, never by spectrum. See the table below.
- Commit messages: no em dashes, never mention AI.

## Architecture

train.parquet -> `Library.from_table` (vectorised 1 Da binning over pyarrow offsets, keyed by inchikey14) -> `predict` (precursor ppm window, cosine, RRF over the spectra of a molecule, nearest-mass fill) -> `write_submission` -> submission.csv

- `src/casmi/schema.py` maps real column names (ms2_mzs, ms2_normalized_intensities, normalized_smiles, inchikey14, instrument_type, ...) to canonical ones. Add aliases here, never rename columns elsewhere.
- `src/casmi/metric.py` exact competition metric: `key14`, `reciprocal_rank`, `mrr_at_k`
- `src/casmi/spectra.py` per-spectrum peak cleaning, binning, entropy similarity (used for queries and tests)
- `src/casmi/library_search.py` `Library` index, `binned_matrix` (vectorised build), `predict`
- `src/casmi/validation.py` metadata frame and the three splits
- `src/casmi/submission.py` writes and validates submission.csv against sample_submission.csv
- `scripts/inspect_data.py` schema and stats
- `scripts/run_baseline.py` modes val, val2x, val3, submit
- `kaggle/kernel/notebook.py` offline kernel script, `kaggle/dataset/` source dataset, `kaggle/rdkit_wheel/` wheel dataset
- `docs/plan.md` competition facts, data facts, strategy phases, dated score log

## Commands

```
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests
python scripts\inspect_data.py data
python scripts\run_baseline.py --mode val2x --limit 300000
python scripts\run_baseline.py --mode val2x
python scripts\run_baseline.py --mode submit
```

Full library build is about 90 s and a validation run about 3 min on this machine.

## Validation

Splits in `src/casmi/validation.py`, all with timsTOF queries in the test mass range (240 to 465). Baseline numbers, Sep 17 2026, full 2.5M library:

| split | held out | MRR@25 |
|---|---|---|
| val | one timsTOF scan, replicates at other energies stay | 0.82 (too easy, ignore) |
| val2x | all timsTOF scans of a structure, structure kept via other instruments | 0.53 |
| val3 | every spectrum of the structure | 0.00 |

Report val2x and val3. The leaderboard sits between them depending on the class mix.

## Conventions

- Canonical column names after `resolve_columns`: molecule_id, smiles, inchikey14, precursor_mz, mzs, intensities, adduct, instrument, ingest_lib.
- Every experiment gets a dated line in the log at the bottom of `docs/plan.md` with val2x, val3, and public LB.
- Structures are identified by inchikey14 (provided column, agrees with the metric's `key14` on 99 percent of structures).

## Fork of the public pipeline

`kaggle/fork_base`, `fork_v2`, `fork_v3` are script kernels forked from the public 0.339 notebook; `src/casmi/fork/` is the same code as an importable module with a local validation harness (`scripts/run_fork_val.py`, `scripts/diag_fork_val.py`) on the 250 enveda-np-examples structures. Public assets are under `data/public/` (gitignored), ranker training file versions under `artifacts/ranker_versions/` (must stay outside `data/`, the file finder searches it). Always pass `--ranker` explicitly. The ranker file the public author ships changes without notice and moved the LB by 0.05.

## Validation rule

Only `scripts/run_fork_val.py --split npx` (the 250 enveda-np-examples structures) predicts the leaderboard. Held-out random training structures (train_ranker.py's holdout, the `--split overlap` set) predicted the opposite ordering of three rankers and were wrong on the board. Never submit on the strength of the training-structure holdout alone.

## Current state

Sep 18 2026: public LB 0.339, rank about 11 of 600, kernel `kaggle/fork_v4` (public pipeline, no candidate prune, ranker file pinned to v4). Ranker retraining is parked: three variants (own rows, mixed, natural-product-only) all scored below the public ranker on np-examples and two of them lost on the board (0.292, 0.331). Row dumps and rankers are in `artifacts/rank_rows_*.npz` and `artifacts/rank_train_*.npz` if a new feature set makes them worth revisiting. Next phase: own fingerprint model on Kaggle GPU, see the status section at the bottom of `docs/plan.md`. Not started, needs the user's go since it consumes days of GPU quota.
