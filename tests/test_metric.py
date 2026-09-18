import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.metric import key14, mrr_at_k, reciprocal_rank  # noqa: E402


def test_key14_ignores_stereo_and_tautomer():
    assert key14("C[C@H](N)C(=O)O") == key14("C[C@@H](N)C(=O)O") == key14("CC(N)C(=O)O")
    # 2-hydroxypyridine vs 2-pyridone are tautomers
    assert key14("Oc1ccccn1") == key14("O=c1cccc[nH]1")


def test_key14_bad_smiles():
    assert key14("not a smiles") is None
    assert key14(None) is None


def test_reciprocal_rank_dedup_and_cutoff():
    caffeine = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    assert reciprocal_rank(caffeine, [caffeine]) == 1.0
    assert reciprocal_rank(caffeine, ["CCO", caffeine]) == 0.5
    # duplicate key of the wrong answer does not consume a slot
    assert reciprocal_rank(caffeine, ["CCO", "OCC", caffeine]) == 0.5
    # invalid smiles does not consume a slot
    assert reciprocal_rank(caffeine, ["xx", caffeine]) == 1.0
    decoys = ["C" * n + "O" for n in range(1, 31)]  # 30 distinct alcohols
    assert reciprocal_rank(caffeine, decoys + [caffeine]) == 0.0


def test_mrr():
    truth = {"a": "CCO", "b": "CCN"}
    preds = {"a": ["CCO"], "b": ["CCO", "CCN"]}
    assert abs(mrr_at_k(truth, preds) - 0.75) < 1e-9
