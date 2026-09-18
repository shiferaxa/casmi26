"""Column-name resolution.

The competition parquet schema is not documented outside Kaggle, so every
loader goes through resolve_columns(), which maps our canonical names to
whatever the file actually calls them. Run scripts/inspect_data.py after
downloading and fix ALIASES if a canonical name resolves to None.
"""
from __future__ import annotations

import pandas as pd

# canonical name -> candidate column names in priority order
ALIASES: dict[str, list[str]] = {
    "spectrum_id": ["spectrum_id", "spec_id", "id", "scan_id"],
    "molecule_id": ["molecule_id", "mol_id", "compound_id"],
    "smiles": ["normalized_smiles", "smiles", "SMILES", "canonical_smiles"],
    "inchikey": ["inchikey", "InChIKey", "inchi_key"],
    "inchikey14": ["inchikey14", "inchikey_14", "inchikey_first_block"],
    "ingest_lib": ["ingest_lib", "library", "source_lib"],
    "num_peaks": ["num_peaks", "n_peaks"],
    "precursor_mz": ["precursor_mz", "precursor_m_z", "prec_mz", "PrecursorMZ"],
    "mzs": ["ms2_mzs", "mzs", "mz", "peaks_mz", "fragment_mz", "mz_array"],
    "intensities": ["ms2_normalized_intensities", "intensities", "intensity", "peaks_intensity", "intensity_array"],
    "adduct": ["adduct", "precursor_type", "ion_type"],
    "charge": ["charge", "precursor_charge"],
    "ionmode": ["ionization_mode", "ionmode", "ion_mode", "polarity"],
    "collision_energy": ["collision_energy_ev", "collision_energy", "ce", "nce"],
    "instrument": ["instrument_type", "instrument", "instrument_name", "source_instrument"],
    "formula": ["molecular_formula", "formula", "molecular_formula", "mol_formula"],
    "exact_mass": ["exact_mass", "monoisotopic_mass", "mono_mass", "parent_mass"],
}


def resolve_columns(columns: list[str]) -> dict[str, str | None]:
    cols = set(columns)
    out: dict[str, str | None] = {}
    for canon, cands in ALIASES.items():
        out[canon] = next((c for c in cands if c in cols), None)
    return out


def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {v: k for k, v in resolve_columns(list(df.columns)).items() if v}
    return df.rename(columns=mapping)
