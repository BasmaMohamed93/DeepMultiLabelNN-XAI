"""Collect every number used in the manuscript into numbers.json (single source for the Word documents)."""
import json, glob
import numpy as np, pandas as pd

N = {}
S = pd.read_csv('summary.csv')          # proposed, stacking_no_graph_priors, no_knn_prior, no_regional_prior, gene_only, two_branch_ensemble
G = pd.read_csv('gene_only_full.csv')
LF = pd.read_csv('label_frequency.csv').groupby(['dataset', 'setting']).mean(numeric_only=True).reset_index()
from scipy import stats as _st
_R = pd.read_csv('final_runs.csv'); _ci = []
for (ds_, st_), g_ in _R.groupby(['dataset', 'setting']):
    for m_ in _R.columns[3:]:
        x_ = g_[m_].astype(float); mu_, sd_ = x_.mean(), x_.std(ddof=1); h_ = _st.t.ppf(0.975, len(x_) - 1) * sd_ / np.sqrt(len(x_))
        _ci.append(dict(dataset=ds_, setting=st_, metric=m_, mean=mu_, sd=sd_, ci_low=mu_ - h_, ci_high=mu_ + h_))
CI = pd.DataFrame(_ci)
f3 = lambda x: '-' if (x is None or (isinstance(x, float) and np.isnan(x))) else f'{x:.3f}'


def key(ds, st): return f'{ds}|{st}'


for ds in ['all', 'high_evidence', 'pharmgkb_only', 'no_tramadol']:
    for st in ['A_random', 'B_gene_disjoint']:
        k = key(ds, st); N[k] = {}
        c = CI[(CI.dataset == ds) & (CI.setting == st)].set_index('metric')
        N[k]['proposed'] = {m: dict(mean=f3(c.loc[m, 'mean']), sd=f3(c.loc[m, 'sd']), lo=f3(c.loc[m, 'ci_low']), hi=f3(c.loc[m, 'ci_high'])) for m in c.index}
        g = G[(G.dataset == ds) & (G.setting == st)].iloc[0]
        N[k]['gene_only'] = {m.replace('_mean', ''): f3(g[m]) for m in G.columns if m.endswith('_mean')}
        N[k]['gene_only_sd'] = {m.replace('_std', ''): f3(g[m]) for m in G.columns if m.endswith('_std')}
        l = LF[(LF.dataset == ds) & (LF.setting == st)].iloc[0]
        N[k]['label_freq'] = {m: f3(l[m]) for m in ['subset_acc', 'precision', 'recall', 'f1', 'bal_accuracy', 'bal_precision', 'bal_recall', 'bal_f1', 'bal_auroc']}
        for mk in ['stacking_no_graph_priors', 'no_knn_prior', 'no_regional_prior', 'two_branch_ensemble']:
            r = S[(S.dataset == ds) & (S.setting == st) & (S.model == mk)]
            if len(r): N[k][mk] = {m.replace('_mean', ''): f3(r.iloc[0][m]) for m in S.columns if m.endswith('_mean')}

# two-branch extra metrics (subset acc etc.) from results_pair
rows = []
for f in glob.glob('results_pair/*sim=True_ens=True.json'):
    r = json.load(open(f)); rows.append(dict(dataset=r['dataset'], setting=r['setting'], **r['test']))
V3 = pd.DataFrame(rows).groupby(['dataset', 'setting']).mean(numeric_only=True).reset_index()
for _, r in V3.iterrows():
    N[key(r.dataset, r.setting)]['two_branch'] = {m: f3(r[m]) for m in ['subset_acc', 'hamming_acc', 'precision', 'recall', 'f1', 'macro_auroc', 'macro_auprc', 'within_gene_auroc', 'hit@1', 'hit@5', 'hit@10']}

T = pd.read_csv('paired_tests.csv')
N['tests'] = [{**r, 'proposed': f3(r['proposed']), 'ref': f3(r['ref']), 'diff': f'{r["diff"]:+.3f}', 'p': ('< 0.001' if r['p'] < 0.001 else f'{r["p"]:.3f}') if r['p'] == r['p'] else 'n/a'}
              for r in T.to_dict('records')]
N['confound_by_source'] = pd.read_csv('confound_by_source.csv').round(3).to_dict('records')
N['calibration'] = pd.read_csv('calibration.csv').groupby(['dataset', 'setting', 'model'])[['brier', 'ece']].mean().round(4).reset_index().to_dict('records')
N['curated'] = pd.read_csv('curated_check.csv').groupby(['dataset', 'model']).mean(numeric_only=True).drop(columns='seed').round(3).reset_index().to_dict('records')
N['treeshap'] = pd.read_csv('treeshap_groups.csv').round(4).to_dict('records')
SP = pd.read_csv('split_stats.csv').groupby(['dataset', 'setting']).mean(numeric_only=True).drop(columns='seed').round(3).reset_index()
N['split'] = SP.to_dict('records')
for ds in ['all', 'high_evidence']:
    D = pd.read_csv(f'per_drug_{ds}.csv')
    N[f'per_drug_{ds}'] = D.round(3).to_dict('records')
    N[f'per_drug_{ds}_summary'] = dict(n=len(D), ge08=int((D.auroc >= 0.8).sum()), ge09=int((D.auroc >= 0.9).sum()), median=round(float(D.auroc.median()), 3),
                                       beats_gene=int((D.auroc > D.gene_auroc).sum()))
json.dump(N, open('numbers.json', 'w'), indent=1)
print('keys', list(N)[:10]); print(json.dumps(N['all|A_random']['proposed']['micro_f1']), N['all|A_random'].get('no_knn_prior'))

LP = pd.read_csv('labelprop_runs.csv').groupby(['dataset', 'setting']).mean(numeric_only=True).reset_index()
for _, r in LP.iterrows():
    N[key(r.dataset, r.setting)]['labelprop'] = {m: f3(r[m]) for m in LP.columns if m not in ('dataset', 'setting', 'seed', 'alpha')}
SS = pd.read_csv('selected_settings.csv')
N['settings'] = SS.groupby(['dataset', 'setting']).agg(coef_xgb=('coef_xgb', 'mean'), coef_hyb=('coef_hyb', 'mean'), coef_nn=('coef_nn', 'mean'),
                                                        bal_th=('bal_th', 'mean')).round(3).reset_index().to_dict('records')
dec = SS.assign(d=SS.decoding.str.replace(r'global_.*', 'global', regex=True)).groupby(['dataset', 'setting']).d.apply(lambda s: ', '.join(f'{k} ({v})' for k, v in s.value_counts().items()))
for r in N['settings']: r['decoding'] = dec.loc[(r['dataset'], r['setting'])]
br = []
for f in glob.glob('results/*knn=True.json'):
    r = json.load(open(f))
    if r['use_region']: br.append(dict(dataset=r['dataset'], setting=r['setting'], **r['branch_micro_auroc'], stacked=r['test']['micro_auroc']))
N['branches'] = pd.DataFrame(br).groupby(['dataset', 'setting']).mean().round(3).reset_index().to_dict('records')
json.dump(N, open('numbers.json', 'w'), indent=1)

# hybrid framework components (branch_eval.py)
import os
if os.path.exists('hybrid_components_summary.csv'):
    H = pd.read_csv('hybrid_components_summary.csv'); N['hybrid'] = {}
    for _, r in H.iterrows():
        N['hybrid'].setdefault(key(r.dataset, r.setting), {})[r.component] = {m.replace('_mean', ''): f3(r[m]) for m in H.columns if m.endswith('_mean')} | {m.replace('_std', '_sd'): f3(r[m]) for m in H.columns if m.endswith('_std')}
    HT = pd.read_csv('hybrid_components_tests.csv')
    N['hybrid_tests'] = [{**r, 'diff': f'{r["diff"]:+.3f}', 'p': ('< 0.001' if r['p'] < 0.001 else f'{r["p"]:.3f}') if r['p'] == r['p'] else 'n/a'} for r in HT.to_dict('records')]
    json.dump(N, open('numbers.json', 'w'), indent=1)
