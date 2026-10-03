import json, glob
import numpy as np, pandas as pd
from scipy import stats

rows = []
for f in glob.glob('results/*.json'):
    r = json.load(open(f)); knn = r.get('use_knn', False); full = knn and r['use_region']
    name = 'proposed' if full else ('stacking_no_graph_priors' if not (knn or r['use_region']) else ('no_knn_prior' if r['use_region'] else 'no_regional_prior'))
    if r.get('use_rank'): continue
    base = dict(dataset=r['dataset'], setting=r['setting'], seed=r['seed'])
    t, b = r['test'], r['balanced']
    rows.append({**base, 'model': name, 'micro_f1': t['f1'], 'precision': t['precision'], 'recall': t['recall'],
                 'macro_auroc': t['macro_auroc'], 'micro_auroc': t['micro_auroc'], 'within_gene_auroc': t['within_gene_auroc'], 'hit@1': t['hit@1'],
                 'hit@10': t['hit@10'], 'bal_accuracy': b['accuracy'], 'bal_precision': b['precision'], 'bal_recall': b['recall'], 'bal_f1': b['f1'], 'bal_auroc': b['auroc']})
    if full:
        g = r['gene_only']; gb = g['balanced']
        rows.append({**base, 'model': 'gene_only', 'micro_f1': g['f1'], 'precision': g['precision'], 'recall': g['recall'], 'macro_auroc': g['macro_auroc'],
                     'micro_auroc': g['micro_auroc'], 'within_gene_auroc': 0.5, 'bal_accuracy': gb['accuracy'], 'bal_precision': gb['precision'],
                     'bal_recall': gb['recall'], 'bal_f1': gb['f1'], 'bal_auroc': gb['auroc']})
for f in glob.glob('results_pair/*sim=True_ens=True.json'):   # previous final ensemble (two-branch ensemble)
    r = json.load(open(f)); t = r['test']
    rows.append(dict(dataset=r['dataset'], setting=r['setting'], seed=r['seed'], model='two_branch_ensemble', micro_f1=t['f1'], precision=t['precision'],
                     recall=t['recall'], macro_auroc=t['macro_auroc'], micro_auroc=t.get('micro_auroc'), within_gene_auroc=t['within_gene_auroc'],
                     **{'hit@1': t['hit@1'], 'hit@10': t['hit@10']}))
D = pd.DataFrame(rows); D.to_csv('all_runs.csv', index=False)
cols = ['micro_f1', 'precision', 'recall', 'macro_auroc', 'micro_auroc', 'within_gene_auroc', 'hit@1', 'hit@10', 'bal_accuracy', 'bal_precision', 'bal_recall', 'bal_f1', 'bal_auroc']
S = D.groupby(['dataset', 'setting', 'model'])[cols].agg(['mean', 'std']).round(3)
S.columns = [f'{a}_{b}' for a, b in S.columns]; S.to_csv('summary.csv')
short = D.groupby(['dataset', 'setting', 'model'])[cols].mean().round(3)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(short)
T = []
for (ds, st), g in D.groupby(['dataset', 'setting']):
    for ref in ['two_branch_ensemble', 'gene_only', 'stacking_no_graph_priors', 'no_knn_prior', 'no_regional_prior']:
        a = g[g.model == 'proposed'].set_index('seed'); b = g[g.model == ref].set_index('seed'); s = a.index.intersection(b.index)
        if len(s) < 3: continue
        for m in ['micro_f1', 'macro_auroc', 'within_gene_auroc', 'bal_f1', 'hit@1']:
            x, y = a.loc[s, m].astype(float), b.loc[s, m].astype(float)
            if y.isna().any(): continue
            T.append(dict(dataset=ds, setting=st, vs=ref, metric=m, proposed=x.mean(), ref=y.mean(), diff=x.mean() - y.mean(), p=stats.ttest_rel(x, y).pvalue if (x - y).std() > 0 else np.nan))
T = pd.DataFrame(T); T.to_csv('paired_tests.csv', index=False); print(T.to_string())

# settings selected on validation data (stacking weights, decoding rule, link-prediction threshold) for the proposed model
sel = []
for f in glob.glob('results/*knn=True.json'):
    r = json.load(open(f))
    if not r['use_region']: continue
    sel.append(dict(dataset=r['dataset'], setting=r['setting'], seed=r['seed'], **{'coef_' + k: v for k, v in r['stack_coef'].items()},
                    decoding=r['decoding'], top1=r['force_top1'], bal_th=r['balanced_threshold']))
pd.DataFrame(sel).to_csv('selected_settings.csv', index=False)
