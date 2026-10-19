# DeepMultiLabelNN-XAI

Explainable hybrid deep learning-machine learning framework for variant-level pharmacogenomic drug-association prediction.
A residual neural network (deep-learning component) and gradient-boosted trees (machine-learning component) are coupled through the
network's embedding and a stacking layer fitted on validation data; knowledge priors and graph-propagated (regional genomic and
feature-space nearest-neighbour) priors are computed out-of-fold. Evaluation uses known-gene and gene-disjoint splits over five seeds.

## Quick start

    pip install -r requirements.txt
    python reproduce.py --tables   # print the main result tables from the included result files (seconds)
    python reproduce.py            # retrain every run and regenerate all tables and figures (several hours on 2 CPU cores)

## Data
| File | Content |
|---|---|
| label_records_raw.csv | Variant-level evidence records: ClinVar drug_response + PharmGKB/ClinPGx clinical (levels 1A-3) and variant annotations (built by labels.py from the public releases) |
| vep.csv.gz | Ensembl VEP annotation of every rsID |
| interactions.csv | DGIdb gene-drug interactions (PharmGKB- and FDA-sourced rows are removed inside pairwise.py) |

## Raw input files
| File | Source |
|---|---|
| clinvar_pgx.vcf.gz | ClinVar drug_response records (GRCh37 VCF, filtered to pharmacogenomic records, file date 2026-09-23); ClinVar data are in the public domain |
| pharmgkb_clinical_annotations.tsv.gz, pharmgkb_clinical_ann_alleles.tsv.gz | PharmGKB/ClinPGx clinical annotations (clinical_annotations.tsv, clinical_ann_alleles.tsv; created 2025-07-05) |
| pharmgkb_var_drug_ann.tsv.gz | PharmGKB/ClinPGx variant annotations (var_drug_ann.tsv; created 2026-09-05) |

PharmGKB/ClinPGx files are redistributed unchanged (gzip-compressed) under the Creative Commons Attribution-ShareAlike 4.0
licence (LICENSE_PharmGKB.txt); please cite ClinPGx (https://www.clinpgx.org/page/citingClinpgx).
labels.py builds label_records_raw.csv and clinvar_pgx_records.csv from these files, and r2_alleles.py builds allele_status.json.

## Code
| File | Purpose |
|---|---|
| build_dataset.py | Datasets (all, high_evidence, pharmgkb_only, no_tramadol) and the 181 VEP features |
| splits.py, evalkit.py | Iterative multi-label stratified split, gene-disjoint split, metrics, within-gene AUROC |
| resnet.py, nn_training.py | Residual neural network (NumPy) and its training loop |
| pairwise.py | Pairwise (variant, drug) instances, gene prior, DGIdb and similarity priors, decoding |
| model.py | Hybrid framework: regional and nearest-neighbour priors, XGBoost, hybrid and residual-network branches, stacking, both evaluation protocols, gene-only baseline |
| hybrid_components.py | The framework against each component alone (residual network, XGBoost, hybrid branch) |
| additional_analyses.py | Architecture ablation of the residual network, random-forest baseline, label-vocabulary check, parameter count |
| summary.py | Means, SDs and paired t-tests |
| confound_by_source.py | Results split by label source (ClinVar vs PharmGKB) |
| extras.py | Per-drug tables, 95% CIs, calibration, Hit@k, curated-evidence check, split statistics, TreeSHAP |
| labelprop_baseline.py | Graph label-propagation baseline |
| gene_only_table.py | Full metric set of the gene-only baseline |
| make_numbers.py | Collects every reported number into numbers.json |
| tcav_analysis.py | TCAV on the residual network (all 420 concept-drug tests, Supplementary Table S9), split statistics and the PharmGKB level-1A/1B consistency check |
| fig_pipeline.py, fig_results.py, fig_compare.py, fig_components.py | Figures (written to figures/) |

## Results folders
results/ (every run of the proposed model and ablations, with predictions of the proposed model), results_components/ (per-component
scores), results_pair/ (two-branch ensemble used for comparison).

All thresholds, early stopping, stacking weights and hyperparameters use only the validation partition; each test partition is scored once.

## Second revision (analyses requested in decision 2)
| File | Purpose |
|---|---|
| r2_runs.py, r2_runs2.py, r2_queue.py, r2_queue2.py | Retraining runs: without tramadol, without frequency features, masked positives, PU weighting, strict ClinVar set, locus-disjoint (C) and temporal (D) settings -> results_r2/ |
| r2_summary.py | Summary of all revision runs -> r2_all_runs.csv, r2_summary.txt |
| r2_analyses.py | Source-matched and source-stratified evaluation -> r2_subset_eval.csv |
| r2_source_auc.py | Source-classifier AUROC -> r2_source_auc.json |
| r2_alleles.py, r2_allele_eval.py | Multi-allelic rsIDs and allele-matched evaluation -> allele_status.json, r2_allele_eval.csv |
| r2_misc.py | Regional-prior coverage and within-gene group counts -> r2_misc.json |
| r2_stats.py | Paired differences with Nadeau-Bengio corrected intervals and d_z -> r2_framework_vs_xgb.csv, r2_framework_vs_geneonly.csv |
| r2_metrics_extra.py | Negative-sampling ratios, precision/recall@k, micro AUPRC, calibration slope/intercept |
| r2_temporal_eval.py | Temporal hold-out ranks -> r2_temporal_ranks.csv |
| r2_curated.py | PharmGKB level-1A/1B consistency check with tied ranks shared -> r2_curated_check.csv |
| r2_xai_stability.py | TreeSHAP and TCAV across five seeds -> r2_treeshap_seeds.csv, r2_tcav_seeds.csv |
| r2_drug_rule.py | Drug-retention rule recomputed on training data -> r2_drug_rule.csv |
| r2_split_counts.py | Per-split, per-drug positive counts -> split_positive_counts.csv |
| fig_r2.py | Figure 5 (multi-seed) and Supplementary Figures S1-S2 |
| make_ci.py | 95% confidence intervals for every value of Table 10 -> final_runs.csv, ci.csv |

Data files added: matched_variants.json, temporal_pairs.csv, clinvar_review_status.csv, allele_status.json.
Environment: Python 3.11, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, XGBoost 2.1.4 (Linux, two CPU cores, no GPU).

## Citation
[[add the article reference after publication]]
