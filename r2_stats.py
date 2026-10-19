"""Paired differences between the framework and the strongest standalone component (XGBoost), with
naive and Nadeau-Bengio corrected (dependent resamples) 95% intervals and effect size d_z."""
import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv('hybrid_components_runs.csv')
FR = {'A_random': (0.7, 0.2), 'B_gene_disjoint': (0.7, 0.2)}   # train / test fractions
rows = []
for (ds, st), g in d.groupby(['dataset', 'setting']):
    for met in ['macro_auroc', 'macro_auprc', 'micro_f1', 'within_gene_auroc', 'hit@1', 'bal_f1', 'bal_auroc']:
        a = g[g.component == 'stack'].sort_values('seed')[met].values; b = g[g.component == 'xgb'].sort_values('seed')[met].values
        if len(a) != len(b) or len(a) < 2: continue
        x = a - b; k = len(x); m, s = x.mean(), x.std(ddof=1)
        tcrit = stats.t.ppf(0.975, k - 1)
        se_naive = s / np.sqrt(k); se_nb = s * np.sqrt(1 / k + FR[st][1] / FR[st][0])
        p_naive = 2 * stats.t.sf(abs(m / se_naive), k - 1) if s > 0 else np.nan
        p_nb = 2 * stats.t.sf(abs(m / se_nb), k - 1) if s > 0 else np.nan
        rows.append(dict(dataset=ds, setting=st, metric=met, framework=a.mean(), xgboost=b.mean(), diff=m,
                         ci_naive=f'[{m - tcrit * se_naive:.3f}, {m + tcrit * se_naive:.3f}]', p_naive=p_naive,
                         ci_corrected=f'[{m - tcrit * se_nb:.3f}, {m + tcrit * se_nb:.3f}]', p_corrected=p_nb, d_z=m / s if s > 0 else np.nan))
R = pd.DataFrame(rows); R.to_csv('r2_framework_vs_xgb.csv', index=False)
pd.set_option('display.width', 250)
print(R.round(3).to_string(index=False))

# framework versus the gene-only baseline (same splits; saved predictions, r2_subset_eval.csv)
S = pd.read_csv('r2_subset_eval.csv'); S = S[S.subset == 'all test']
rows = []
for (ds, st), g in S.groupby(['dataset', 'setting']):
    for met in ['macro_auroc', 'macro_auprc', 'micro_f1']:
        a = g[g.model == 'model'].sort_values('seed')[met].values; b = g[g.model == 'gene_only'].sort_values('seed')[met].values
        x = a - b; k = len(x); m, s = x.mean(), x.std(ddof=1); tcrit = stats.t.ppf(0.975, k - 1)
        se_nb = s * np.sqrt(1 / k + FR[st][1] / FR[st][0])
        rows.append(dict(dataset=ds, setting=st, metric=met, framework=a.mean(), gene_only=b.mean(), diff=m,
                         ci_corrected=f'[{m - tcrit * se_nb:.3f}, {m + tcrit * se_nb:.3f}]',
                         p_corrected=2 * stats.t.sf(abs(m / se_nb), k - 1) if s > 0 else np.nan, d_z=m / s if s > 0 else np.nan))
G = pd.DataFrame(rows); G.to_csv('r2_framework_vs_geneonly.csv', index=False)
print(G.round(3).to_string(index=False))
