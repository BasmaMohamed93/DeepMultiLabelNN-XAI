import os, json, glob, re, pandas as pd, numpy as np
rows = []
for f in glob.glob('results/*_region=True_rank=False_knn=True.json') + glob.glob('results_r2/*.json'):
    r = json.load(open(f)); tag = f.split('/')[-1][:-5]
    m = re.match(r'(.+?)_(A_random|B_gene_disjoint|C_locus_disjoint)_(\d)_region=True_rank=False_knn=True(.*)', tag)
    if not m: continue
    if f.startswith('results_r2') and m.group(4) == '' and os.path.exists(f.replace('results_r2', 'results')): continue
    t, g = r['test'], r['gene_only']
    rows.append(dict(dataset=m.group(1), setting=m.group(2), seed=int(m.group(3)), variant=m.group(4).strip('_') or 'reference',
                     macro_auroc=t['macro_auroc'], macro_auprc=t['macro_auprc'], micro_f1=t['f1'], within_gene=t['within_gene_auroc'],
                     link_f1=r['balanced']['f1'], gene_only_auroc=g['macro_auroc'], gene_only_f1=g['f1']))
D = pd.DataFrame(rows); D.to_csv('r2_all_runs.csv', index=False)
f = lambda s: f'{s.mean():.3f} ± {s.std(ddof=1):.3f}' if len(s) > 1 else f'{s.mean():.3f}'
S = D.groupby(['dataset', 'variant', 'setting']).agg(n=('seed', 'size'), **{k: (k, f) for k in ['macro_auroc', 'macro_auprc', 'micro_f1', 'within_gene', 'gene_only_auroc', 'gene_only_f1']})
pd.set_option('display.width', 250); print(S.to_string())
