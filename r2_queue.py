"""Second-revision runs, in priority order (finished runs are skipped)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model
A, B = 'A_random', 'B_gene_disjoint'
J = [('all', st, sd, dict(feat='nofreq')) for st in (A, B) for sd in range(5)]
J += [('all', 'C_locus_disjoint', sd, {}) for sd in range(5)]
J += [('high_evidence_strict', st, sd, {}) for st in (A, B) for sd in range(5)]
J += [('high_evidence_no_tramadol', st, sd, {}) for st in (A, B) for sd in range(5)]
J += [('all', st, sd, dict(mask_frac=0.2)) for st in (A, B) for sd in range(5)]
J += [('no_tramadol', st, sd, dict(feat='nofreq')) for st in (A, B) for sd in range(5)]
J += [('high_evidence', st, sd, dict(feat='nofreq')) for st in (A, B) for sd in range(5)]
J += [('all', st, sd, dict(mask_frac=0.1)) for st in (A, B) for sd in range(5)]
for ds, st, sd, kw in J:
    model.run(ds, st, sd, out_dir='results_r2', **kw)
