import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model
jobs = [('all', 'C_locus_disjoint', sd, 'full') for sd in range(5)]
jobs += [('high_evidence_strict', st, sd, 'full') for st in ('A_random', 'B_gene_disjoint') for sd in range(5)]
jobs += [('high_evidence_no_tramadol', st, sd, 'full') for st in ('A_random', 'B_gene_disjoint') for sd in range(5)]
jobs += [('high_evidence', st, sd, 'nofreq') for st in ('A_random', 'B_gene_disjoint') for sd in range(5)]
for ds, st, sd, feat in jobs:
    model.run(ds, st, sd, out_dir='results_r2', feat=feat)
for frac in (0.1, 0.2):
    for st in ('A_random', 'B_gene_disjoint'):
        for sd in range(5):
            model.run('all', st, sd, out_dir='results_r2', mask_frac=frac)
