import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from build_dataset import build
from splits import random_split, gene_split
from pairwise import gene_prior
L = pd.read_csv('label_records_raw.csv')
hi = L[(L.source == 'PharmGKB_CA') & L.level.astype(str).isin(['1A', '1B'])][['variant', 'drug']].drop_duplicates()
KW = {'all': {}, 'high_evidence': {'drop_levels': ('3',)}, 'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}}
rows = []
for ds in KW:
    V, Y, drugs, _ = build(**KW[ds]); genes = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
    vidx = {v: i for i, v in enumerate(V.index)}
    pairs = [(vidx[v], drugs.index(d)) for v, d in zip(hi.variant, hi.drug) if v in vidx and d in drugs]
    for st in ('A_random', 'B_gene_disjoint'):
        for s in range(5):
            z = np.load(f'results/{ds}_{st}_{s}_region=True_rank=False_knn=True.npz'); te = z['te']
            tr, va, te2 = random_split(Y, s) if st == 'A_random' else gene_split(genes, Y, s); assert (te == te2).all()
            Gs = gene_prior(genes, Y, tr, va, te, s)[2]; pos = {v: k for k, v in enumerate(te)}
            for name, P in (('framework', z['Ps']), ('gene_only', Gs)):
                for i, j in pairs:
                    if i in pos:
                        row = P[pos[i]]; r = int((row > row[j]).sum()) + 1; t = int((row >= row[j]).sum())
                        rows.append(dict(dataset=ds, setting=st, seed=s, model=name, rank=r, rank_ties=t))
D = pd.DataFrame(rows); D.to_csv('r2_curated_check.csv', index=False)
g = D.groupby(['dataset', 'setting', 'model']).agg(pairs=('rank', 'size'), top1=('rank', lambda r: (r <= 1).mean()), top1_strict=('rank_ties', lambda r: (r <= 1).mean()),
                                                  top3=('rank', lambda r: (r <= 3).mean()), top5=('rank', lambda r: (r <= 5).mean()), median=('rank', 'median'))
print(g.round(3).to_string())
