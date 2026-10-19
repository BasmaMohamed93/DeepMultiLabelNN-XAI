"""Reproduce every DeepMultiLabelNN-XAI number reported in the manuscript.

Usage (from this folder):
    python reproduce.py            # rebuild the labels, train all runs and run every analysis (several hours on 2 CPU cores), then print the tables
    python reproduce.py --tables   # only print the tables from existing results/*.json

Steps: (1) variant-level labels from the public ClinVar and PharmGKB files (labels.py); (2) 4 datasets (all, high_evidence,
pharmgkb_only, no_tramadol) x 2 settings (known genes = random variant split; unseen genes = gene-disjoint split) x 5 seeds,
plus the stacking-only model and the ablations on the main dataset; (3) the analyses of the second revision (results_r2/:
source confounding, assumed negative class, multi-allelic variants, locus-disjoint and temporal hold-out, paired comparisons,
calibration, negative-sampling ratios, explanation stability, drug-retention rule, 95% CIs and Figures 5, S1-S2).
Finished runs are skipped, so an interrupted run can be restarted with the same command.
"""
import sys, os, glob, json, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

if '--tables' not in sys.argv:
    subprocess.run([sys.executable, 'labels.py'], check=True)                                # labels from the public files
    import model
    os.makedirs('results_components', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    for ds in ['all', 'high_evidence', 'pharmgkb_only', 'no_tramadol']:
        for st in ['A_random', 'B_gene_disjoint']:
            for sd in range(5):
                model.run(ds, st, sd)                                                          # proposed model
                model.run(ds, st, sd, out_dir='results_components')                           # same model, per-component scores saved
                if ds in ('all', 'high_evidence'):
                    model.run(ds, st, sd, use_region=False, use_rank=False, use_knn=False)   # stacking only
                if ds == 'all':
                    model.run(ds, st, sd, use_region=True, use_knn=False)                    # ablation: no kNN prior
                    model.run(ds, st, sd, use_region=False, use_knn=True)                    # ablation: no regional prior
    for script in ['summary.py', 'confound_by_source.py', 'extras.py', 'labelprop_baseline.py', 'gene_only_table.py', 'hybrid_components.py', 'additional_analyses.py', 'tcav_analysis.py', 'make_numbers.py', 'fig_pipeline.py', 'fig_results.py', 'fig_compare.py', 'fig_components.py']:
        subprocess.run([sys.executable, script], check=True)
    # second revision: additional runs (results_r2/) and analyses
    for script in ['r2_queue.py', 'r2_queue2.py', 'r2_summary.py', 'r2_analyses.py', 'r2_source_auc.py', 'r2_alleles.py',
                   'r2_allele_eval.py', 'r2_misc.py', 'r2_stats.py', 'r2_metrics_extra.py', 'r2_temporal_eval.py', 'r2_curated.py',
                   'r2_xai_stability.py', 'r2_drug_rule.py', 'r2_split_counts.py', 'make_ci.py', 'fig_r2.py']:
        subprocess.run([sys.executable, script], check=True)

f = lambda x: f'{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}'
NAME = {'all': 'All evidence', 'high_evidence': 'High-confidence evidence', 'pharmgkb_only': 'PharmGKB only', 'no_tramadol': 'No tramadol'}
SET = {'A_random': 'known genes', 'B_gene_disjoint': 'unseen genes'}
t1, t2 = [], []
for ds in NAME:
    for st in SET:
        R = [json.load(open(p)) for p in glob.glob(f'results/{ds}_{st}_*_region=True_rank=False_knn=True.json')]
        if not R: continue
        b = lambda k: [r['balanced'][k] for r in R]; t = lambda k: [r['test'][k] for r in R]
        t1.append([NAME[ds], SET[st], len(R), f(b('accuracy')), f(b('precision')), f(b('recall')), f(b('f1')), f(b('auroc'))])
        t2.append([NAME[ds], SET[st], len(R), f(t('subset_acc')), f(t('precision')), f(t('recall')), f(t('f1')), f(t('macro_auroc')), f(t('within_gene_auroc'))])
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print('\nTable A. Balanced link-prediction protocol (each positive variant-drug pair vs one random negative pair; 10 draws; threshold from validation)')
print(pd.DataFrame(t1, columns=['Dataset', 'Split', 'Seeds', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC']).to_string(index=False))
print('\nTable B. Multi-label protocol (all 60 drugs scored for every test variant; decoding rule from validation)')
print(pd.DataFrame(t2, columns=['Dataset', 'Split', 'Seeds', 'Subset acc.', 'Precision', 'Recall', 'Micro F1', 'Macro AUROC', 'Within-gene AUROC']).to_string(index=False))
