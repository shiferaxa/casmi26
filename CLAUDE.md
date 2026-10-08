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
- Fingerprint checkpoints for the harness come only from the folder given by `--models` (default `data/public/casmi26-fp-models-v2`). Never rely on a sweep of `data/`: every downloaded or smoke-tested `fp_*.pt` under it would be ensembled silently (this corrupted one evaluation on Sep 21). Public v2 and v4 share file names, so a combined folder needs renamed copies. Files with "merged" in the name form the merged group; singles are averaged, then averaged 50/50 with the merged group, so an own single mixed with v4 gets a minority weight.
- Validation must group by InChIKey14 and hold out by instrument, never by spectrum. See the table below.
- Background shell waiters do not survive the end of a session. Never schedule a Kaggle push behind a local timer; push immediately when quota allows, otherwise write the exact command in the notes for the next session (Sep 20 2026: two GPU days lost this way).
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

## Automation (GitHub Actions)

- `.github/workflows/kaggle-train.yml` pushes a kernel folder to Kaggle. Runs every Saturday 00:05 UTC (the GPU quota refresh) for `kaggle/train_fp_v2`, and on demand from the Actions tab with any kernel folder and accelerator. Pushing a kernel that is already running restarts it as a new version.
- `.github/workflows/kaggle-checkpoint.yml` runs every six hours: reports the training kernel status and, once complete, publishes new `fp_*.pt` checkpoints to the Kaggle dataset amhashiferaw/casmi26-fp-own (file names carry the epoch, so nothing is published twice) and keeps them as a workflow artifact for 14 days. Inference kernels attach that dataset to use our model.
- Both need the repository secret `KAGGLE_API_TOKEN` (set Sep 20 2026 from the local token file). Rotate the Kaggle token by updating that secret.
- Dispatch from the CLI without gh: POST `/repos/shiferaxa/casmi26/actions/workflows/<file>/dispatches` with the credential from `git credential fill`.

## Fork of the public pipeline

`kaggle/fork_base`, `fork_v2`, `fork_v3` are script kernels forked from the public 0.339 notebook; `src/casmi/fork/` is the same code as an importable module with a local validation harness (`scripts/run_fork_val.py`, `scripts/diag_fork_val.py`) on the 250 enveda-np-examples structures. Public assets are under `data/public/` (gitignored), ranker training file versions under `artifacts/ranker_versions/` (must stay outside `data/`, the file finder searches it). Always pass `--ranker` explicitly. The ranker file the public author ships changes without notice and moved the LB by 0.05.

## Validation rule

Only `scripts/run_fork_val.py --split npx` (the 250 enveda-np-examples structures) predicts the leaderboard. Held-out random training structures (train_ranker.py's holdout, the `--split overlap` set) predicted the opposite ordering of three rankers and were wrong on the board. Never submit on the strength of the training-structure holdout alone.

## Training on Runpod

Model training moved off the Kaggle quota to a rented Runpod pod on Oct 8 2026 (Community Cloud, RTX PRO 5000 Blackwell 48 GB, about $0.79/h, account amhashiferaw@gmail.com). Connection details and gotchas are in the memory file `runpod-training-box`. On the pod: `/workspace/input` (training inputs, pulled once by `scripts/runpod_setup.sh`), `/workspace/src` (copies of `kaggle/train_fp_v2/notebook.py` and the two scripts), `/workspace/runs/<tag>/` (train.log plus `fp_single_<tag>_s<step>.pt` best and `_last.pt`). `scripts/runpod_train.sh <tag> CASMI_CFG_<KEY>=<value> ...` runs one job; any key of the notebook's CFG can be overridden. Detach jobs with `setsid nohup ... < /dev/null &`. The Runpod MCP (plugin `runpod@runpod`) can stop and start the pod from a session started after Oct 7 2026; stop the pod when nothing is queued, the volume keeps everything. Pull checkpoints with `scp -P <port> root@<ip>:/workspace/runs/<tag>/fp_single_*.pt data/own_runs/<tag>/` and score with `scripts/eval_channel4.py --models data/own_runs/<tag>`.

## Current state

Oct 8 2026: public LB 0.361, rank 798 of 2,472 (forks of the public v4m/v4l pipeline; the crowd is at 0.41 to 0.42). Strategy: make our own fingerprint model the edge. Own checkpoints: Kaggle seed 7 in `artifacts/models_v2run`, Runpod seed11 (best s24000) and seed7_80k (best s78000) in `data/own_runs/<tag>/`; the three together are `artifacts/models_own3`, with the public v2 pair added in `artifacts/models_own3_v2`. On channel 4 alone own three score 0.527 Bernoulli against public v4's 0.515, but through the full pipeline on np-examples (ranker v4, prune 400, power 4) class 2 drops: public v2 0.758, own three 0.715, own three plus v2 0.729. The public ranker was fitted on v2-model features, so a channel 4 gain needs ranker rows rebuilt from our model's logits before it can show. Prepared but not pushed: dataset folder `kaggle/fp_own3_dataset` (amhashiferaw/casmi26-fp-own3) and kernel `kaggle/casmi26-fork-v4n-own3` (fork of the public 0.421 v4n fusion kernel, engine-2 takes v2 plus own three, T4 GPU), details in the last `docs/plan.md` line. To get the board's verdict, run from the repo root in Git Bash, in order: (1) `cd kaggle/fp_own3_dataset && ../../.venv/Scripts/kaggle.exe datasets create -p .`; (2) `../../.venv/Scripts/kaggle.exe datasets status amhashiferaw/casmi26-fp-own3`, repeat until ready; (3) `cd ../casmi26-fork-v4n-own3 && ../../.venv/Scripts/kaggle.exe kernels push -p .`; (4) `../../.venv/Scripts/kaggle.exe kernels status amhashiferaw/casmi26-fork-v4n-own3`, repeat until COMPLETE, then `cd ../.. && .venv/Scripts/kaggle.exe kernels output amhashiferaw/casmi26-fork-v4n-own3 -p C:/Users/amhas/Desktop/Projects/casmi26/artifacts/casmi26-fork-v4n-own3` and check that the log line 'engine fp models' lists the three own files plus fp_single_s2.pt and fp_merged_m1.pt (if the notebook's 3,000 character tail of the engine-2 output cut it off, look for 'fp models: 4 single + 1 merged on cuda'); (5) `.venv/Scripts/kaggle.exe competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k amhashiferaw/casmi26-fork-v4n-own3 -v 1 -f submission.csv -m "v4n fusion with own three-model fingerprint ensemble in engine-2"`. Pod is stopped. Log every number at the bottom of `docs/plan.md`.
