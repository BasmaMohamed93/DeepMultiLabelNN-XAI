"""Supplementary Table S18: drug-retention rule recomputed on the training partition of every split.
Metrics of the final model over the drugs retained by the training-only rule versus all 60 drugs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from r2_analyses import load, KW
from build_dataset import build
from splits import gene_split, random_split
from pairwise import decode
from sklearn.metrics import roc_auc_score, average_precision_score

V, Y, drugs, _ = build(); genes = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
rows = []
for st in ('A_random', 'B_gene_disjoint'):
    for sd in range(5):
        z, js = load('all', st, sd); te = z['te']; Ps = z['Ps']; th = z['th']
        tr, va, te2 = random_split(Y, sd) if st == 'A_random' else gene_split(genes, Y, sd)
        frac = 0.7   # nominal training fraction, as in the original check
        keep = np.array([(Y[tr, j].sum() >= 30 * frac) and (len(set(genes[tr][Y[tr, j] > 0])) >= 3) for j in range(Y.shape[1])])
        Yt = Y[te]; B = decode(Ps, th, js['force_top1'])
        for lab, m in (('all drugs', np.ones(Y.shape[1], bool)), ('retained on training data', keep)):
            ev = m & (Yt.sum(0) > 0) & (Yt.sum(0) < len(Yt))
            Ym, Pm, Bm = Yt[:, m], Ps[:, m], B[:, m]
            tp = (Bm & (Ym > 0)).sum(); fp = (Bm & (Ym == 0)).sum(); fn = (~Bm & (Ym > 0)).sum()
            rows.append(dict(setting=st, seed=sd, drug_set=lab, n_drugs=int(m.sum()),
                             macro_auroc=np.mean([roc_auc_score(Yt[:, j], Ps[:, j]) for j in np.where(ev)[0]]),
                             macro_auprc=np.mean([average_precision_score(Yt[:, j], Ps[:, j]) for j in np.where(ev)[0]]),
                             micro_auprc=average_precision_score(Ym.ravel(), Pm.ravel()),
                             micro_f1=2 * tp / max(2 * tp + fp + fn, 1),
                             hit1=float(np.mean([Ym[i, np.argmax(Pm[i])] > 0 for i in range(len(Ym)) if Ym[i].sum() > 0]))))
D = pd.DataFrame(rows); D.to_csv('r2_drug_rule.csv', index=False)
print(D.groupby(['setting', 'drug_set']).agg(['mean', 'std']).round(3).to_string())
