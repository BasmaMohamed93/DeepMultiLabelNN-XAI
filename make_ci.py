"""95% confidence intervals (t-interval over the five seeds) for every value of Table 10,
computed from the saved result files in results/ -> final_runs.csv and ci.csv (same computation as extras.py, section 1)."""
import json, glob, os
import numpy as np, pandas as pd
from scipy import stats
os.chdir(os.path.dirname(os.path.abspath(__file__)))
runs = []
for ds in ['all', 'high_evidence', 'no_tramadol', 'pharmgkb_only']:
    for st in ['A_random', 'B_gene_disjoint']:
        for s in range(5):
            r = json.load(open(f'results/{ds}_{st}_{s}_region=True_rank=False_knn=True.json'))
            t, b = r['test'], r['balanced']
            runs.append(dict(dataset=ds, setting=st, seed=s, subset_acc=t['subset_acc'], hamming_acc=t['hamming_acc'], precision=t['precision'],
                             recall=t['recall'], micro_f1=t['f1'], sample_f1=t['sample_f1'], macro_auroc=t['macro_auroc'], micro_auroc=t['micro_auroc'],
                             macro_auprc=t['macro_auprc'], within_gene_auroc=t['within_gene_auroc'],
                             **{'hit@1': t['hit@1'], 'hit@5': t['hit@5'], 'hit@10': t['hit@10']},
                             bal_accuracy=b['accuracy'], bal_precision=b['precision'], bal_recall=b['recall'], bal_f1=b['f1'],
                             bal_auroc=b['auroc'], bal_auprc=b['auprc']))
R = pd.DataFrame(runs); R.to_csv('final_runs.csv', index=False)
ci = []
for (ds, st), g in R.groupby(['dataset', 'setting']):
    for m in R.columns[3:]:
        x = g[m].astype(float); mu, sd = x.mean(), x.std(ddof=1); h = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))
        ci.append(dict(dataset=ds, setting=st, metric=m, mean=mu, sd=sd, ci_low=mu - h, ci_high=mu + h))
pd.DataFrame(ci).round(4).to_csv('ci.csv', index=False)
print(pd.DataFrame(ci).round(4).query("metric in ['macro_auroc','micro_f1','bal_f1']").to_string(index=False))
