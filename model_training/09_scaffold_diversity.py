#!/usr/bin/env python3
"""Bemis-Murcko scaffold diversity of the top-ranked screening hits.

The screening results file is sorted by predicted pIC50 (descending), so the
first N rows are the top-N hits. Reports the number of distinct scaffolds and
the largest scaffold share for several top-N cutoffs.
"""

import csv
from collections import Counter

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SRC = "../data/enamine_screening_results.csv"
OUT = "scaffold_diversity.csv"

rows = []
with open(SRC) as f:
    r = csv.DictReader(f)
    for line in r:
        rows.append((float(line["predicted_pIC50"]), line["smiles"], line["catalog_id"]))
print(f"library rows: {len(rows)}")


def scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else MurckoScaffold.MurckoScaffoldSmiles(mol=m)


out = [["top_n", "n_parsed", "n_unique_scaffolds", "scaffold_frac", "max_same_scaffold"]]
for N in [5, 10, 100, 500, 1000, 2000]:
    sub = rows[:N]
    counts = Counter()
    ok = 0
    for _, smi, _cid in sub:
        s = scaffold(smi)
        if s is None:
            continue
        ok += 1
        counts[s if s else "(acyclic/empty)"] += 1
    uniq = len(counts)
    maxshare = max(counts.values()) if counts else 0
    out.append([N, ok, uniq, round(uniq / ok, 3) if ok else 0, maxshare])
    print(f"Top {N:<5} parsed {ok:<5} unique scaffolds={uniq:<5} "
          f"fraction={uniq/ok:.2f} largest share={maxshare}")

with open(OUT, "w", newline="") as fh:
    csv.writer(fh).writerows(out)
print(f"\nwritten: {OUT}")
