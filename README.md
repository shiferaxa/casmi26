# casmi26

Working entry for the Kaggle competition [Enveda CASMI 2026: Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra). The task is to predict the 2D structure (a SMILES string) of a small molecule from its tandem mass spectra, submitting up to 25 ranked candidates per molecule. The metric is mean reciprocal rank at 25 on the connectivity block of a tautomer-canonical InChIKey.

Status is kept in [docs/plan.md](docs/plan.md) (facts, findings, a dated log of every run with its validation numbers and public score) and [docs/schedule.md](docs/schedule.md) (the plan to the deadline).

## How the pipeline works

Data flow for one test molecule:

1. Every spectrum of the molecule gives a neutral mass from its precursor m/z and adduct. The median is the target mass.
2. Candidates are every structure within 8.5 ppm of the target mass in a pool of about 712k structures: COCONUT plus every structure in the training set.
3. Four kinds of evidence are computed per candidate: spectral entropy similarity to training spectra of that structure (library channel), Tanimoto similarity to structures whose training spectra resemble the query after a mass shift (analog channel), the fraction of peak intensity explained by one or two bond breaks of the candidate (fragmentation channel), and the dot product of the candidate fingerprint with the logits of a spectrum-to-fingerprint transformer (model channel).
4. A gradient-boosted ranker turns the evidence into a probability, candidates are deduplicated by canonical key, and the top 25 are written.

Steps 2 to 4 are the public pipeline published on Kaggle by prvsiyan and haideptry under the Apache License 2.0 (see NOTICE). This repository adds a local harness that runs that pipeline on a validation split that predicts the leaderboard, the experiments that came out of it, and the training pipeline for an own fingerprint model.

## Layout

- `src/casmi/` metric, schema mapping, a vectorised library-search baseline, and validation splits.
- `src/casmi/fork/` the public pipeline as an importable module (`core.py`) plus the local harness (`harness.py`) with masks for the three test classes and hooks for ranking variants.
- `scripts/` validation runs (`run_fork_val.py`), failure breakdowns, ranker training-row dumps, ranker training, channel evaluations.
- `kaggle/` the submission kernels (`fork_v*`), the preprocessing kernel that tokenises 2.5M spectra, and the fingerprint model training kernel. Dataset payloads are not committed.
- `tests/` unit tests for the metric and the library build.

## Validation

The only split that has predicted the public leaderboard is the 250 structures the host measured on the test instrument (`ingest_lib == enveda-np-examples`), evaluated under three masks: everything visible, the structure's own spectra hidden, and the structure removed from the candidate pool. Random held-out training structures predicted the opposite ordering of three rankers and were wrong on the board.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install numba torch --index-url https://download.pytorch.org/whl/cpu
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data
python -m pytest -q tests
python scripts\run_fork_val.py --split npx --ranker artifacts\ranker_versions\v4\rank_train.npz --prunes 400:lib
```

The public assets (candidate pool, model checkpoints, ranker rows) are Kaggle datasets by prvsiyan; the harness expects them under `data/public/`.

## License

Apache License 2.0. See LICENSE and NOTICE.
