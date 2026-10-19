# Second revision (decision 2026-10-13) — re-analysis findings

Code: /home/claude/v3/code (r2_*.py). All runs 5 seeds; mean ± SD. Reference = submitted v3 model.

## Point 1 — source confounding
- Source classifier (ClinVar-only vs PharmGKB-only variants): AUROC 0.98 full features; 0.98 without missingness indicators (median imputation still reveals); 0.81 without population-frequency features. Frequency missingness: gnomAD genomes 71.9% (ClinVar) vs 1.7% (PharmGKB); 1000G AF 89.3% vs 4.8%.
- Tramadol: 1,584/3,904 variants (40.6%), 1,536 ClinVar-only.
- Propensity-matched subset (213 ClinVar + 213 PharmGKB variants, source AUROC 0.455): framework macro AUROC 0.936 ± 0.018 known / 0.780 ± 0.053 unseen (gene-only 0.867 / 0.500); AUPRC 0.647 / 0.385; micro-F1 0.725 / 0.334.
- No frequency features (also no missingness indicators): known 0.901 ± 0.011 (AUPRC 0.453, F1 0.643); unseen 0.686 ± 0.037 (AUPRC 0.186, F1 0.250).
- No tramadol: known 0.875 / unseen 0.699; no tramadol + no frequency: 0.867 / 0.672.
- High-confidence without tramadol: known 0.912 ± 0.015, unseen 0.763 ± 0.037.
- Source-stratified (main): ClinVar-only test variants AUROC 1.000 known / 0.880 unseen (≈11 drugs); PharmGKB-only 0.868 / 0.710 (gene-only 0.786 / 0.500).

## Point 2 — PU / alleles
- Hiding 20% of training+validation positives: known 0.919 ± 0.007 (ref 0.928), unseen 0.751 ± 0.049 (ref 0.764).
- SCAR bound: observed AUROC is a lower bound; theta = 1/2/5% unlabelled positives -> 0.932/0.937/0.951 (known), 0.767/0.769/0.778 (unseen).
- Multi-allelic rsIDs (>1 alternate allele in dbSNP): 1,656/3,904 (42.4%). VEP features describe one allele (first listed). Checkable coding SNVs: 172 match, 231 mismatch (5.9% of all variants), 1,253 non-coding uncheckable. Excluding mismatched: 0.929 / 0.766; biallelic only: 0.946 / 0.785.

## Point 3 — regional prior
- Unseen-gene test variants with a training variant within 1 kb / 10 kb / 100 kb / 1 Mb: 1.0% / 7.4% / 22.0% / 73.0% (known genes: 57.8 / 72.1 / 83.4 / 91.9%).
- Locus-disjoint split (no test variant within 1 Mb of training, no shared gene; 508 loci): 0.734 ± 0.076, AUPRC 0.128, F1 0.322 (gene-only 0.500).
- No-regional-prior ablation exists (Table 16).

## Point 4 — framework vs XGBoost alone (paired, Nadeau-Bengio corrected)
- Known genes: no difference (macro AUROC diff 0.000, corrected CI [-0.001, 0.002]).
- Unseen genes: macro AUROC +0.011 main (d_z 1.09, corrected p 0.19), +0.011 high-conf, +0.016 no_tramadol (p 0.022), +0.009 PharmGKB-only (p 0.064); positive in 4/4 datasets.

## Point 5
- ClinVar records in high-confidence set: 1,638/1,698 "no assertion criteria provided"; 55 expert panel; 5 multiple submitters. High-confidence set = ClinVar (no review filter) + PharmGKB levels 1A–2B + variant annotations.
- Strict set (ClinVar >=1 star + PharmGKB 1A–2B/VA; 691 variants, 312 genes, 16 drugs): known 0.896 ± 0.012, unseen 0.722 ± 0.030.
- PharmGKB 1A/1B check (131 pairs, known genes): top-1 57.3% vs gene-only 55.0% with fair tie-breaking (75.6% only when gene-only ties are resolved in its favour); top-3 99.2% vs 95.0%. Unseen genes (246 pairs): top-1 19.1% vs 0%, top-3 43.5% vs 17.1%.
- Note: shipped tcav_analysis.py curated check used the two-branch ensemble; recomputed with final model in r2_curated.py.

## Point 6
- Within-gene AUROC groups (gene-drug with both classes): main known 181 (172–189); unseen 184 (150–219); high-conf known 34, unseen 31.
- split_positive_counts.csv: per dataset/setting/seed/drug train/val/test positives (for Zenodo).
- Need: user's submitted manuscript for citation check; DGIdb release; Ensembl release used by VEP REST.

## Additional analyses (requested by Reviewer 3)
- Temporal hold-out: 187 pairs first curated in PharmGKB clinical annotations from 2021 (159 variants, 33 drugs) hidden; trained on earlier evidence. Rank among undocumented drugs (tie-fair): top-1 32.7% vs gene-only 32.1%, top-3 42.4 vs 40.4, top-5 48.0 vs 42.9, top-10 59.0 vs 48.3; median rank 6 vs 16.5; AUROC held-out vs undocumented pairs 0.838 ± 0.005 vs 0.776.
- PU weighting (unlabelled pairs weight 0.25 in XGBoost branches): known 0.921 ± 0.007 (F1 0.669), unseen 0.763 ± 0.044 (F1 0.335). Masking 10%: known 0.923.
- Negative-sampling ratios (main, known genes; threshold from 1:1 validation): AUROC 0.956 at all ratios; AUPRC 0.961 / 0.880 / 0.830 / 0.707 (1:1 / 1:5 / 1:10 / all); F1 0.872 / 0.702 / 0.568 / 0.246. SD across 10 negative draws of 1:1 F1 = 0.004.
- Precision@k / recall@k (main, known): k=1 0.704/0.646; k=3 0.300/0.759; k=5 0.194/0.796; k=10 0.106/0.846. Unseen: k=1 0.367/0.331; k=10 0.082/0.628.
- Micro AUPRC: 0.707 known, 0.314 unseen (high-conf 0.927 / 0.611). Calibration slope / intercept: 1.03 / 0.08 known, 0.92 / -0.35 unseen.
- TreeSHAP stability (5 seeds): regional prior 29.1 ± 2.0%, population frequency 29.2 ± 1.4%, gene prior 14.0 ± 2.5%, kNN prior 6.0 ± 1.2%; top-4 groups identical in all seeds.
- TCAV stability: 420 tests per seed; 168–217 significant per seed; 79 significant in >=4/5 seeds, 21 in all 5; direction consistent in 66% of significant tests; no-replacement vs resampling agreement 77.8% (LoF 79.7%). Of the 23 starred cells in Figure 5b (seed 0), only 7 are robust (>=4/5 seeds, same direction) -> figure/Table S9 should use multi-seed consensus.
