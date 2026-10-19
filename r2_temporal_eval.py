"""Temporal hold-out: variant-drug pairs first curated in PharmGKB clinical annotations from 2021 onwards were hidden;
models were trained on the remaining (earlier) evidence and the held-out variants were scored."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from build_dataset import build
from pairwise import gene_prior
V, Y, drugs, _ = build(); G = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
H = pd.read_csv('temporal_pairs.csv'); vi = {v: i for i, v in enumerate(V.index)}
hid = [(vi[v], drugs.index(d)) for v, d in zip(H.variant, H.drug) if v in vi and d in drugs]
te = np.array(sorted({i for i, _ in hid})); other = np.setdiff1d(np.arange(len(V)), te)
Yh = Y.copy()
for i, j in hid: Yh[i, j] = 0
rows = []; aucs = []
for sd in range(5):
    fn = f'results_r2/all_D_temporal_{sd}_region=True_rank=False_knn=True.npz'
    if not os.path.exists(fn): continue
    z = np.load(fn); assert (z['te'] == te).all()
    o = np.random.default_rng(sd).permutation(other); nv = int(round(0.125 * len(o))); va, tr = np.sort(o[:nv]), np.sort(o[nv:])
    Gs = gene_prior(G, Yh, tr, va, te, sd)[2]; pos = {v: k for k, v in enumerate(te)}
    for name, P in (('framework', z['Ps']), ('gene_only', Gs)):
        lab, sc = [], []
        for i, j in hid:
            row = P[pos[i]].copy(); known = Yh[i] > 0; cand = ~known          # candidates: drugs not documented before the cut-off
            r_best = int((row[cand] > row[j]).sum()) + 1; r_worst = int((row[cand] >= row[j]).sum())
            rows.append(dict(seed=sd, model=name, r_best=r_best, r_worst=r_worst, n_cand=int(cand.sum())))
        # AUROC: held-out pairs vs all undocumented pairs of the held-out variants
        M = np.zeros((len(te), len(drugs)), bool)
        for i, j in hid: M[pos[i], j] = True
        cand = (Yh[te] == 0); y = M[cand]; s = P[cand]
        aucs.append(dict(seed=sd, model=name, auroc=roc_auc_score(y, s)))
D = pd.DataFrame(rows)
def ek(d, k): return np.mean([(min(k, w) - b + 1) / (w - b + 1) if b <= k else 0 for b, w in zip(d.r_best, d.r_worst)])
S = D.groupby('model').apply(lambda d: pd.Series({'pairs_per_seed': len(d) / d.seed.nunique(), 'top1': ek(d, 1), 'top3': ek(d, 3), 'top5': ek(d, 5), 'top10': ek(d, 10),
                                                   'median_rank': np.median((d.r_best + d.r_worst) / 2)}))
A = pd.DataFrame(aucs).groupby('model').auroc.agg(['mean', 'std'])
print(S.round(3).to_string()); print(A.round(3).to_string()); print('seeds', D.seed.nunique(), 'held-out pairs', len(hid), 'variants', len(te), 'drugs', len({j for _, j in hid}))
D.to_csv('r2_temporal_ranks.csv', index=False)
