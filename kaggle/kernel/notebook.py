# Offline Kaggle script for the CASMI 2026 library-search baseline.
# Source files come from the private dataset amhashiferaw/casmi26-src (flat .py files),
# rebuilt here into a casmi/ package so imports match the repo.
import shutil, subprocess, sys, time
from pathlib import Path

# RDKit is not on the Kaggle image and internet is off: install from the wheel dataset.
try:
    import rdkit  # noqa: F401
except ImportError:
    wheels = list(Path("/kaggle/input").rglob("rdkit-*.whl"))
    if not wheels:
        raise SystemExit("rdkit wheel dataset not attached")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", str(wheels[0])], check=True)
    import rdkit  # noqa: F401
print("rdkit", rdkit.__version__, flush=True)

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

comp_hits = list(Path("/kaggle/input").rglob("train.parquet"))
if not comp_hits:
    raise SystemExit("competition data not found under /kaggle/input: " + str(sorted(Path("/kaggle/input").iterdir())))
COMP = comp_hits[0].parent
print("competition data at", COMP, flush=True)
# locate the source files wherever Kaggle mounted the dataset
hits = [p for p in Path("/kaggle/input").rglob("library_search.py")]
if not hits:
    for p in sorted(Path("/kaggle/input").rglob("*"))[:200]:
        print(p)
    raise SystemExit("casmi26-src dataset files not found under /kaggle/input")
SRC = hits[0].parent
PKG = Path("/kaggle/working/pkg/casmi")
PKG.mkdir(parents=True, exist_ok=True)
for f in SRC.glob("*.py"):
    shutil.copy(f, PKG / f.name)
print("source files:", sorted(x.name for x in PKG.glob("*.py")), flush=True)
sys.path.insert(0, str(PKG.parent))

from casmi.library_search import Library, predict
from casmi.schema import resolve_columns
from casmi.submission import validate_submission, write_submission

NEEDED = ["smiles", "inchikey14", "precursor_mz", "mzs", "intensities", "molecule_id"]


def load_canonical(path):
    resolved = resolve_columns(pq.read_schema(path).names)
    cols = {c: r for c, r in resolved.items() if c in NEEDED and r}
    table = pq.read_table(path, columns=list(cols.values()))
    return table.rename_columns([next(c for c, r in cols.items() if r == n) for n in table.column_names])


t0 = time.time()
train = load_canonical(COMP / "train.parquet")
test = load_canonical(COMP / "test.parquet").to_pandas()
print(len(train), len(test), f"loaded {time.time()-t0:.0f}s", flush=True)

lib = Library.from_table(train)
del train
print(f"library {lib.matrix.shape[0]} spectra, {len(lib.key_to_smiles)} structures, {time.time()-t0:.0f}s", flush=True)

preds = predict(lib, test, ppm=20.0, show_progress=False)
sub = write_submission(preds, str(COMP / "sample_submission.csv"), "/kaggle/working/submission.csv")
validate_submission(sub, pd.read_csv(COMP / "sample_submission.csv"))
print(sub.head(3), f"\ndone {time.time()-t0:.0f}s")
