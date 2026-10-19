"""Regional-prior coverage, within-gene group counts, multi-allelic rsIDs, ClinVar review status."""
import sys, os, json, gzip, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from build_dataset import build
from splits import gene_split, random_split, within_gene_auroc
WINDOWS = (1e3, 1e4, 1e5, 1e6)
R = {}
for ds, kw in (('all', {}), ('high_evidence', {'drop_levels': ('3',)})):
    V, Y, drugs, _ = build(**kw); genes = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
    chrom = V.seq_region_name.astype(str).values; pos = V.start.astype(float).values
    for st in ('A_random', 'B_gene_disjoint'):
        cov = {w: [] for w in WINDOWS}; cov_lab = {w: [] for w in WINDOWS}; ng = []; same_gene_test = []
        for sd in range(5):
            tr, va, te = random_split(Y, sd) if st == 'A_random' else gene_split(genes, Y, sd)
            for w in WINDOWS:
                hit = np.zeros(len(te), bool); hitp = np.zeros(len(te), bool)
                for c in np.unique(chrom[te]):
                    qi = np.where(chrom[te] == c)[0]; ri = tr[chrom[tr] == c]
                    if len(ri) == 0: continue
                    D = np.abs(pos[te[qi]][:, None] - pos[ri][None, :]); M = (D <= w) & (D > 0)
                    hit[qi] = M.any(1)
                    hitp[qi] = ((M.astype(np.float32) @ (Y[ri] > 0).astype(np.float32)) > 0).any(1)   # neighbour with any positive label
                cov[w].append(hit.mean()); cov_lab[w].append(hitp.mean())
            z = np.load(f'results/{ds}_{st}_{sd}_region=True_rank=False_knn=True.npz')
            assert (z['te'] == te).all()
            _, n = within_gene_auroc(Y[te], z['Ps'], genes[te]); ng.append(n)
        R[f'{ds}|{st}'] = dict(regional_coverage={f'{int(w)}': round(float(np.mean(cov[w])), 3) for w in WINDOWS},
                              regional_coverage_with_positive_neighbour={f'{int(w)}': round(float(np.mean(cov_lab[w])), 3) for w in WINDOWS},
                              within_gene_groups_mean=float(np.mean(ng)), within_gene_groups_range=[int(min(ng)), int(max(ng))])
# multi-allelic rsIDs in the main dataset
V, Y, drugs, _ = build()
v = pd.read_csv('vep.csv.gz', low_memory=False)
na = v.groupby('input').size(); multi = na[na > 1].index.intersection(V.index)
R['multiallelic'] = dict(variants=len(V), multiallelic=int(len(multi)), share=round(len(multi) / len(V), 3))
json.dump(R, open('r2_misc.json', 'w'), indent=1); print(json.dumps(R, indent=1))
