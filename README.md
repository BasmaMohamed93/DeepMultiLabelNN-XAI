# DeepMultiLabelNN-XAI

Explainable hybrid deep learning-machine learning framework for variant-level pharmacogenomic drug-association prediction.
A residual neural network (deep-learning component) and gradient-boosted trees (machine-learning component) are coupled through the
network's embedding and a stacking layer fitted on validation data; knowledge priors and graph-propagated (regional/linkage and
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
| interactions.csv.xz | DGIdb gene-drug interactions (PharmGKB- and FDA-sourced rows are removed inside pairwise.py) |

label_records_raw.csv can be rebuilt with labels.py from the public releases placed in raw/: the ClinVar GRCh37 VCF restricted to
drug_response records (clinvar_pgx.vcf.gz) and the PharmGKB/ClinPGx files clinical_annotations.tsv and var_drug_ann.tsv.

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

## Citation
[[add the article reference after publication]]
