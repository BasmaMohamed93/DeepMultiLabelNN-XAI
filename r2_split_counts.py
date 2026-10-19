"""Per-split, per-drug positive counts (train / validation / test) for every dataset, setting and seed."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from build_dataset import build
from splits import random_split, gene_split
KW = {'all': {}, 'high_evidence': {'drop_levels': ('3',)}, 'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}, 'no_tramadol': {'exclude_drugs': ('tramadol',)}}
rows = []
for ds, kw in KW.items():
    V, Y, drugs, _ = build(**kw); genes = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
    for st in ('A_random', 'B_gene_disjoint'):
        for sd in range(5):
            tr, va, te = random_split(Y, sd) if st == 'A_random' else gene_split(genes, Y, sd)
            for j, d in enumerate(drugs):
                rows.append(dict(dataset=ds, setting=st, seed=sd, drug=d, train_pos=int(Y[tr, j].sum()), val_pos=int(Y[va, j].sum()), test_pos=int(Y[te, j].sum()),
                                 train_variants=len(tr), val_variants=len(va), test_variants=len(te)))
D = pd.DataFrame(rows); D.to_csv('split_positive_counts.csv', index=False); print(D.shape, D.groupby(['dataset','setting']).test_pos.apply(lambda s: (s == 0).sum()).to_dict())
