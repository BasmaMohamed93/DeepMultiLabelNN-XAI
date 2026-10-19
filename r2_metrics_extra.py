"""Metrics from saved predictions: negative-sampling ratios, precision/recall@k, micro AUPRC, calibration slope/intercept,
and variability of the 1:1 protocol across negative draws."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from build_dataset import build
from model import balanced_threshold
KW = {'all': {}, 'high_evidence': {'drop_levels': ('3',)}}
rows, cal, pk = [], [], []
for ds, kw in KW.items():
    V, Y, drugs, _ = build(**kw)
    for st in ('A_random', 'B_gene_disjoint'):
        for sd in range(5):
            f = f'results/{ds}_{st}_{sd}_region=True_rank=False_knn=True'
            z = np.load(f + '.npz'); te, va = z['te'], z['va']; P = z['Ps']; Yt = Y[te]
            th = balanced_threshold(Y[va], z['Pv'], np.random.default_rng(sd + 100))
            y, p = Yt.ravel(), P.ravel(); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
            for ratio in (1, 5, 10, 'all'):
                R = []
                for draw in range(10 if ratio != 'all' else 1):
                    rng = np.random.default_rng(sd * 1000 + draw)
                    i = np.r_[pos, neg if ratio == 'all' else rng.choice(neg, min(len(neg), ratio * len(pos)), replace=False)]
                    yy, pp = y[i], p[i]; b = pp >= th
                    tp, fp, fn = (b & (yy == 1)).sum(), (b & (yy == 0)).sum(), (~b & (yy == 1)).sum()
                    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                    R.append(dict(auroc=roc_auc_score(yy, pp), auprc=average_precision_score(yy, pp), precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-12)))
                R = pd.DataFrame(R)
                rows.append(dict(dataset=ds, setting=st, seed=sd, ratio=str(ratio), **R.mean().to_dict(), **{f'{k}_draw_sd': v for k, v in R.std(ddof=1).fillna(0).to_dict().items()}))
            # precision / recall at k and micro AUPRC
            o = np.argsort(-P, 1); npos = Yt.sum(1)
            for k in (1, 3, 5, 10):
                hits = np.take_along_axis(Yt, o[:, :k], 1).sum(1)
                pk.append(dict(dataset=ds, setting=st, seed=sd, k=k, precision_at_k=float((hits / k).mean()), recall_at_k=float((hits / np.maximum(npos, 1)).mean())))
            lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg.reshape(-1, 1), y)
            cal.append(dict(dataset=ds, setting=st, seed=sd, micro_auprc=average_precision_score(y, p), calib_slope=lr.coef_[0][0], calib_intercept=lr.intercept_[0]))
D = pd.DataFrame(rows); D.to_csv('r2_negative_ratios.csv', index=False)
PK = pd.DataFrame(pk); PK.to_csv('r2_precision_recall_at_k.csv', index=False)
CA = pd.DataFrame(cal); CA.to_csv('r2_calibration_auprc.csv', index=False)
pd.set_option('display.width', 220)
print(D.groupby(['dataset', 'setting', 'ratio'])[['auroc', 'auprc', 'precision', 'recall', 'f1', 'f1_draw_sd']].mean().round(3).to_string())
print(PK.groupby(['dataset', 'setting', 'k'])[['precision_at_k', 'recall_at_k']].agg(['mean', 'std']).round(3).to_string())
print(CA.groupby(['dataset', 'setting'])[['micro_auprc', 'calib_slope', 'calib_intercept']].agg(['mean', 'std']).round(3).to_string())
