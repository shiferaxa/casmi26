#!/usr/bin/env bash
# Fork a public Kaggle notebook unchanged as a private CPU kernel, run it, and submit its submission.csv.
#   scripts/fork_public.sh <owner/kernel-slug> <our-slug> "<submission message>"
set -u
P=/c/Users/amhas/Desktop/Projects/casmi26; K=$P/.venv/Scripts/kaggle; SRC=$1; OURS=$2; MSG=$3
D=$P/kaggle/$OURS; rm -rf $D; mkdir -p $D
$K kernels pull $SRC -p $D -m >/dev/null 2>&1
mv $D/*.ipynb $D/notebook.ipynb 2>/dev/null
$P/.venv/Scripts/python - "$D" "$SRC" "$OURS" <<'PY'
import json, sys
d, src, ours = sys.argv[1:]
m = json.load(open(d + '/kernel-metadata.json'))
out = {"id": f"amhashiferaw/{ours}", "title": ours, "code_file": "notebook.ipynb", "language": "python",
       "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false", "enable_internet": "false",
       "dataset_sources": [x for x in m.get("dataset_sources", []) if x],
       "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [x for x in m.get("kernel_sources", []) if x],
       "model_sources": [x for x in m.get("model_sources", []) if x]}
json.dump(out, open(d + '/kernel-metadata.json', 'w'), indent=2)
nb = json.load(open(d + '/notebook.ipynb', encoding='utf-8'))
nb['cells'].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [f"Private fork of {src} (Apache 2.0), run unchanged on CPU."]})
json.dump(nb, open(d + '/notebook.ipynb', 'w', encoding='utf-8'))
print('prepared', out['id'], 'datasets', len(out['dataset_sources']))
PY
cd $D && $K kernels push -p . 2>&1 | tail -1
sleep 300; S=""
for i in $(seq 1 110); do S=$($K kernels status amhashiferaw/$OURS 2>&1 | tail -1); case "$S" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 300; done
echo "$(date +%H:%M) $OURS: $S"
case "$S" in
  *COMPLETE*) $K competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k amhashiferaw/$OURS -v 1 -f submission.csv -m "$MSG" 2>&1 | tail -1 ;;
  *) mkdir -p $P/artifacts/$OURS && $K kernels output amhashiferaw/$OURS -p $P/artifacts/$OURS >/dev/null 2>&1; tail -c 2500 $P/artifacts/$OURS/*.log ;;
esac
