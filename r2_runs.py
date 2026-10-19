"""Additional runs for the second revision (source confounding)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model
os.makedirs('results_r2', exist_ok=True)
jobs = [('all', 'A_random', 0, 'full')]
for feat in ('nofreq', 'nomiss'):
    for st in ('A_random', 'B_gene_disjoint'):
        for sd in range(5):
            jobs.append(('all', st, sd, feat))
for st in ('A_random', 'B_gene_disjoint'):
    for sd in range(5):
        jobs.append(('no_tramadol', st, sd, 'nofreq'))
for ds, st, sd, feat in jobs:
    model.run(ds, st, sd, out_dir='results_r2', feat=feat)
