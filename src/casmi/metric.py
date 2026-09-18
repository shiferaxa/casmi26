"""Competition metric: MRR@25 on tautomer-canonicalised InChIKey14.

Matching rule (from the competition description): each predicted SMILES is
parsed with RDKit, tautomer-canonicalised, converted to an InChIKey, and only
the first 14 characters (connectivity block) are compared. Stereo and charge
layers are ignored. Score for a molecule is 1/rank of the first matching
candidate within the top 25, else 0. Final score is the mean over molecules.
"""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_ENUM = rdMolStandardize.TautomerEnumerator()
K = 25


@lru_cache(maxsize=2_000_000)
def key14(smiles: str | None) -> str | None:
    """Tautomer-canonical InChIKey connectivity block, or None if unparsable."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _ENUM.Canonicalize(mol)
    except Exception:
        pass
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return ik[:14] if ik else None


def reciprocal_rank(truth_smiles: str, candidates: list[str], k: int = K) -> float:
    t = key14(truth_smiles)
    if t is None:
        return 0.0
    seen: set[str] = set()
    rank = 0
    for smi in candidates:
        c = key14(smi)
        if c is None or c in seen:
            continue  # unparsable and duplicate keys do not consume a rank slot
        seen.add(c)
        rank += 1
        if rank > k:
            break
        if c == t:
            return 1.0 / rank
    return 0.0


def mrr_at_k(truth: dict[str, str], preds: dict[str, list[str]], k: int = K) -> float:
    """truth: molecule_id -> smiles. preds: molecule_id -> ranked smiles list."""
    if not truth:
        return 0.0
    total = 0.0
    for mid, smi in truth.items():
        total += reciprocal_rank(smi, preds.get(mid, []), k)
    return total / len(truth)


def parse_submission_cell(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [s.strip() for s in cell.split(";") if s.strip()]
