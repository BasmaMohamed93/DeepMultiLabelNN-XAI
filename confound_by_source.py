"""Source-confound check for: performance split by label source, and how well VEP features predict the source."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from build_dataset import build
from pairwise import dec_metrics, decode
from model import balanced_eval

rows = []
for ds, kw in [('all', {}), ('high_evidence', {'drop_levels': ('3',)})]:
    V, Y, drugs, _ = build(**kw)
    src = np.where(V.label_source.str.contains('ClinVar').values, 'ClinVar', 'PharmGKB')
    for f in sorted(glob.glob(f'results/{ds}_*_region=True_rank=False_knn=True.npz')):
        d = np.load(f); te = d['te']; Ps = d['Ps']; th = d['th']; st = 'A_random' if 'A_random' in f else 'B_gene_disjoint'
        r = json.load(open(f.replace('.npz', '.json'))); seed = r['seed']
        B = decode(Ps, th, r['force_top1'])
        for s in ['ClinVar', 'PharmGKB']:
            m = src[te] == s
            if m.sum() < 10: continue
            Ys = Y[te][m]; dm = dec_metrics(Ys, B[m])
            try: au = roc_auc_score(Ys.ravel(), Ps[m].ravel())
            except ValueError: au = np.nan
            bal = balanced_eval(Ys, Ps[m], r['balanced_threshold'], np.random.default_rng(seed))
            rows.append(dict(dataset=ds, setting=st, seed=seed, source=s, n_test=int(m.sum()), micro_f1=dm['f1'], micro_auroc=au, balanced_f1=bal['f1']))
R = pd.DataFrame(rows)
S = R.groupby(['dataset', 'setting', 'source']).agg(n_test=('n_test', 'mean'), micro_f1=('micro_f1', 'mean'), micro_auroc=('micro_auroc', 'mean'), balanced_f1=('balanced_f1', 'mean')).round(3)
print(S); S.to_csv('confound_by_source.csv')
