"""Results of every component of the hybrid framework on identical splits:
residual neural network alone (deep learning), XGBoost alone (machine learning), hybrid branch (residual-network
embedding -> XGBoost) and the full hybrid framework (stacking of the three). Each component gets its own
validation-selected decoding and link-prediction threshold; the test partition is scored once."""
import sys, os, json, glob, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
from build_dataset import build
from pairwise import choose_decoding, decode, dec_metrics
from evalkit import metrics
from splits import within_gene_auroc
from model import balanced_eval, balanced_threshold

RES = sys.argv[1] if len(sys.argv) > 1 else 'results_components'
KW = {'all': {}, 'high_evidence': {'drop_levels': ('3',)}, 'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}, 'no_tramadol': {'exclude_drugs': ('tramadol',)}}
NAME = {'nn': 'Residual NN alone (deep learning)', 'xgb': 'XGBoost alone (machine learning)', 'hyb': 'Hybrid branch (NN embedding -> XGBoost)', 'stack': 'Hybrid framework (stacked)'}
rows = []
for ds, kw in KW.items():
    files = sorted(glob.glob(f'{RES}/{ds}_*_branches.npz'))
    if not files: continue
    V, Y, drugs, _ = build(**kw); G = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
    for f in files:
        d = np.load(f); r = json.load(open(f.replace('_branches.npz', '.json'))); te, va = d['te'], d['va']; Yv, Ys = Y[va], Y[te]
        s = r['seed']; st = r['setting']
        for k in ['nn', 'xgb', 'hyb']:
            PV, PS = d[f'Pv_{k}'], d[f'Ps_{k}']
            n, th, ft = choose_decoding(Yv, PV); m = dec_metrics(Ys, decode(PS, th, ft)); M = metrics(Ys, PS, th); wg, _ = within_gene_auroc(Ys, PS, G[te])
            o = np.argsort(-PS, 1); h = {}
            for kk in (1, 10):
                top = np.zeros_like(Ys, bool); np.put_along_axis(top, o[:, :kk], True, 1); h[f'hit@{kk}'] = float(((top & (Ys > 0)).sum(1) > 0).mean())
            b = balanced_eval(Ys, PS, balanced_threshold(Yv, PV, np.random.default_rng(s + 100)), np.random.default_rng(s + 200))
            rows.append(dict(dataset=ds, setting=st, seed=s, component=k, subset_acc=m['subset_acc'], precision=m['precision'], recall=m['recall'], micro_f1=m['f1'],
                             macro_auroc=M['macro_auroc'], macro_auprc=M['macro_auprc'], within_gene_auroc=wg, **h, bal_accuracy=b['accuracy'],
                             bal_precision=b['precision'], bal_recall=b['recall'], bal_f1=b['f1'], bal_auroc=b['auroc']))
        t, b = r['test'], r['balanced']
        rows.append(dict(dataset=ds, setting=st, seed=s, component='stack', subset_acc=t['subset_acc'], precision=t['precision'], recall=t['recall'], micro_f1=t['f1'],
                         macro_auroc=t['macro_auroc'], macro_auprc=t['macro_auprc'], within_gene_auroc=t['within_gene_auroc'], **{'hit@1': t['hit@1'], 'hit@10': t['hit@10']},
                         bal_accuracy=b['accuracy'], bal_precision=b['precision'], bal_recall=b['recall'], bal_f1=b['f1'], bal_auroc=b['auroc']))
        ref = f.replace(RES, 'results').replace('_branches.npz', '.json')   # identical run in the main result set?
        if os.path.exists(ref):
            assert abs(json.load(open(ref))['test']['f1'] - t['f1']) < 1e-9, f'mismatch {f}'
D = pd.DataFrame(rows); D.to_csv('hybrid_components_runs.csv', index=False)
cols = [c for c in D.columns if c not in ('dataset', 'setting', 'seed', 'component')]
S = D.groupby(['dataset', 'setting', 'component'])[cols].agg(['mean', 'std'])
S.columns = [f'{a}_{b}' for a, b in S.columns]; S = S.reset_index(); S.to_csv('hybrid_components_summary.csv', index=False)
T = []
for (ds, st), g in D.groupby(['dataset', 'setting']):
    a = g[g.component == 'stack'].set_index('seed')
    for k in ['nn', 'xgb', 'hyb']:
        b = g[g.component == k].set_index('seed')
        for m in ['micro_f1', 'macro_auroc', 'within_gene_auroc', 'bal_f1']:
            x, y = a[m], b.loc[a.index, m]
            T.append(dict(dataset=ds, setting=st, vs=k, metric=m, framework=x.mean(), component=y.mean(), diff=x.mean() - y.mean(),
                          p=stats.ttest_rel(x, y).pvalue if (x - y).std() > 0 else np.nan))
pd.DataFrame(T).to_csv('hybrid_components_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(D.groupby(['dataset', 'setting', 'component'])[['micro_f1', 'macro_auroc', 'within_gene_auroc', 'hit@1', 'bal_accuracy', 'bal_precision', 'bal_recall', 'bal_f1']].mean().round(3))
print(pd.DataFrame(T).round(4).to_string())
