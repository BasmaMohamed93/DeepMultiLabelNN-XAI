import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model
for sd in range(5): model.run('all', 'D_temporal', sd, out_dir='results_r2')
for st in ('A_random', 'B_gene_disjoint'):
    for sd in range(5): model.run('all', st, sd, out_dir='results_r2', pu_weight=0.25)
