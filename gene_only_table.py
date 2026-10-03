"""Gene-only baseline: full metric set on the same splits/decoding protocol as (for Tables 4-6)."""
import os, sys, json, warnings; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import pairwise as PW
from build_dataset import build
from splits import gene_split, random_split
KW = {'all': {}, 'high_evidence': {'drop_levels': ('3',)}, 'no_tramadol': {'exclude_drugs': ('tramadol',)}, 'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}}
DATA = {}
for ds, kw in KW.items():
    V, Y, drugs, flow = build(**kw); DATA[ds] = (V, Y, drugs, np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object), flow)


def gene_scores(ds, st, s):
    V, Y, drugs, G, _ = DATA[ds]
    tr, va, te = random_split(Y, s) if st == 'A_random' else gene_split(G, Y, s)
    Gt, Gv, Gs, p0 = PW.gene_prior(G, Y, tr, va, te, s)
    return tr, va, te, Gv, Gs, p0
from pairwise import choose_decoding, decode, dec_metrics
from evalkit import metrics
rows = []
for ds in DATA:
    V, Y, drugs, G, _ = DATA[ds]
    for st in ['A_random', 'B_gene_disjoint']:
        for s in range(5):
            tr, va, te, Gv, Gs, p0 = gene_scores(ds, st, s); Ys = Y[te]
            n, th, ft = choose_decoding(Y[va], Gv); m = dec_metrics(Ys, decode(Gs, th, ft)); M = metrics(Ys, Gs, th)
            o = np.argsort(-Gs, 1); h = {}
            for k in (1, 5, 10):
                top = np.zeros_like(Ys, bool); np.put_along_axis(top, o[:, :k], True, 1); h[f'hit@{k}'] = float(((top & (Ys > 0)).sum(1) > 0).mean())
            r = json.load(open(f'results/{ds}_{st}_{s}_region=True_rank=False_knn=True.json'))['gene_only']
            rows.append(dict(dataset=ds, setting=st, seed=s, subset_acc=m['subset_acc'], hamming_acc=m['hamming_acc'], precision=m['precision'], recall=m['recall'],
                             micro_f1=m['f1'], macro_auroc=M['macro_auroc'], macro_auprc=M['macro_auprc'], **h, bal_accuracy=r['balanced']['accuracy'],
                             bal_precision=r['balanced']['precision'], bal_recall=r['balanced']['recall'], bal_f1=r['balanced']['f1'], bal_auroc=r['balanced']['auroc'],
                             check_f1=r['f1']))
D = pd.DataFrame(rows); assert np.allclose(D.micro_f1, D.check_f1)
S = D.groupby(['dataset', 'setting']).agg(['mean', 'std']).drop(columns=['seed', 'check_f1'], level=0).round(3)
S.columns = [f'{a}_{b}' for a, b in S.columns]; S.to_csv('gene_only_full.csv'); print(S[[c for c in S.columns if c.endswith('mean')]].to_string())
