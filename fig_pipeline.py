import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(16, 17.2))
ax.set_xlim(0, 160); ax.set_ylim(0, 172); ax.axis('off')
C = {'data': ('#E8F1FA', '#2F6DB5'), 'proc': ('#EEF6EC', '#3C8C3A'), 'split': ('#FFF4E0', '#B06E0C'),
     'know': ('#F3ECF8', '#7A4BA8'), 'model': ('#FDECEC', '#B83B3B'), 'new': ('#FFF9D6', '#9A7B00'), 'eval': ('#EFEFEF', '#4D4D4D')}


def box(x, y, w, h, title, lines, kind, fs=9.0, tfs=10.3):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 1.8, title, ha='center', va='top', fontsize=tfs, fontweight='bold', color=ec)
    ax.text(x + 1.5, y + h - 5.6, '\n'.join(lines), ha='left', va='top', fontsize=fs, color='#1e1e1e', linespacing=1.38)


def arrow(x1, y1, x2, y2, color='#444444', lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14, color=color, lw=lw, shrinkA=1, shrinkB=1))


def band(y, label):
    ax.text(0.8, y, label, rotation=90, ha='left', va='center', fontsize=10, fontweight='bold', color='#666666')


ax.text(80, 171.5, 'DeepMultiLabelNN-XAI: variant-level pharmacogenomic drug-association prediction', ha='center', va='top', fontsize=14.5, fontweight='bold')

# 1 sources ---------------------------------------------------------------------------------------------------------
band(155, '1  Data sources')
box(5, 147, 34, 18, 'ClinVar (GRCh37 VCF)', ['CLNSIG = drug_response', 'drug parsed from CLNDN', '2,065 pharmacogenomic records'], 'data')
box(43, 147, 36, 18, 'PharmGKB / ClinPGx', ['Clinical annotations, LoE 1A-3', '(level 4 removed)', 'Significant variant annotations'], 'data')
box(83, 147, 36, 18, 'Ensembl VEP (REST)', ['6,703 rsIDs annotated', 'canonical transcript,', 'most-severe allele per rsID'], 'data')
box(123, 147, 33, 18, 'DGIdb interactions', ['gene-drug pharmacology', 'PharmGKB & FDA rows', 'REMOVED (no label overlap)'], 'data')

# 2 labels & features ------------------------------------------------------------------------------------------------
band(131, '2  Labels & features')
box(5, 121, 74, 21, 'Variant-level labels (keyed by rsID)', [
    '13,414 evidence records -> 9,145 unique (variant, drug) pairs',
    'positive = documented variant-specific evidence; negative = none documented',
    'drug kept if >= 30 variants from >= 3 genes (fixed before modelling)',
    'main: 3,904 variants x 60 drugs, 988 genes, cardinality 1.29'], 'proc', fs=8.8)
box(83, 121, 73, 21, 'Variant features: 181 VEP-derived (fit on training part only)', [
    'consequence (22 SO terms), impact, biotype, variant class',
    'SIFT, PolyPhen, CADD, BLOSUM62, amino-acid change, distance',
    'allele frequency: 1000G, gnomAD exome/genome (+ AFR/EAS/NFE)',
    'NOT features: gene symbol, chromosome, position'], 'proc', fs=8.8)
arrow(22, 147, 30, 142); arrow(61, 147, 52, 142); arrow(101, 147, 112, 142)

# 3 evaluation design -------------------------------------------------------------------------------------------------
band(107, '3  Evaluation design')
box(5, 98, 47, 19, 'Known genes (setting A)', ['random 70/10/20 variant split,', 'iterative multi-label stratification', '82% of test variants share a train gene', '+ within-gene AUROC'], 'split', fs=8.7)
box(56, 98, 45, 19, 'Unseen genes (setting B)', ['gene-disjoint 70/10/20 split', 'stratified on gene label vectors', '0 test genes seen in training'], 'split', fs=8.7)
box(105, 98, 51, 19, '4 datasets x 5 seeds', ['All evidence (main)   High-confidence (no LoE 3)', 'PharmGKB-only (no source confound)', 'No-tramadol (sensitivity)', 'tuning / stacking / thresholds on VALIDATION only'], 'split', fs=8.5)
arrow(42, 121, 28, 117); arrow(80, 121, 79, 117); arrow(119, 121, 130, 117)

# 4 model --------------------------------------------------------------------------------------------------------------
band(66, '4  Model')
box(5, 60, 43, 33, 'Knowledge priors (out-of-fold)', ['Gene prior: training label rate of', '  the drug in the variant\'s gene',
    'DGIdb gene-drug: 6 pair features', 'Similar-gene prior: top-10 genes', '  by DGIdb drug-profile Jaccard',
    'Similar-drug prior: target-set', '  Jaccard x gene label rates'], 'know', fs=8.5)
box(5, 32, 43, 25, 'Graph-propagated priors (out-of-fold)', ['Regional / LD prior: label rate of', '  training variants within 1 kb, 10 kb,',
    '  100 kb, 1 Mb (same position excl.)', 'Feature-space kNN prior: k = 10, 30', '  and same-gene kernel on VEP features'], 'new', fs=8.5)
box(53, 78, 50, 15, 'Pairwise instances', ['one row per (variant, drug): 3,904 x 60', 'variant features + drug one-hot +', 'all priors + drug prevalence'], 'model', fs=8.6)
box(53, 55, 15, 18, 'Branch 1', ['XGBoost', 'binary,', 'pairwise'], 'model', fs=8.4, tfs=9.6)
box(70.5, 55, 15, 18, 'Branch 2', ['hybrid:', 'NN 64-d', 'embedding', '-> XGBoost'], 'model', fs=8.4, tfs=9.6)
box(88, 55, 15, 18, 'Branch 3', ['residual', 'NN, multi-', 'label (60', 'sigmoids)'], 'model', fs=8.4, tfs=9.6)
box(108, 55, 48, 38, 'Residual NN (DeepMultiLabelNN)', ['Linear(181, 256) - BN - GELU stem',
    '2 residual blocks:', '  [Linear-BN-GELU-Dropout-Linear] + skip', 'Linear(256, 64) - BN - GELU embedding',
    'Linear(64, 60) + sigmoid', 'weighted BCE + label smoothing 0.05', 'AdamW, cosine annealing, early stop (val)', 'used as Branch 3 and, via its 64-d', 'embedding, inside Branch 2'], 'model', fs=8.6)
box(53, 32, 50, 18, 'Stacking + decoding', ['logistic regression on branch logits,', 'fitted on VALIDATION (GroupKFold by variant)',
    'decoding: per-drug or global threshold,', '  optional forced top-1 (validation F1)'], 'new', fs=8.4)
box(108, 32, 48, 18, 'Output', ['calibrated probability for each of', 'the 60 drugs + ranked drug list', 'for every variant'], 'model', fs=8.6)
ax.plot([48, 50.5], [76, 76], color='#444444', lw=1.5); ax.plot([48, 50.5, 50.5], [44, 44, 84], color='#9A7B00', lw=1.5); arrow(50.5, 84, 53, 84, color='#9A7B00'); arrow(78, 98, 78, 93)
arrow(66, 78, 60.5, 73); arrow(78, 78, 78, 73); arrow(90, 78, 95.5, 73)
arrow(108, 64, 103, 64, color='#B83B3B')
arrow(60.5, 55, 68, 50); arrow(78, 55, 78, 50); arrow(95.5, 55, 88, 50)
arrow(103, 41, 108, 41)

# 5 evaluation -------------------------------------------------------------------------------------------------------
band(14, '5  Evaluation')
box(3, 1, 37, 27, 'Two test protocols', ['Multi-label: all 60 drugs per variant', '  subset acc., micro P/R/F1,', '  macro AUROC/AUPRC, within-gene',
    'Link prediction (1:1): each true pair', '  vs 1 random negative pair (10 draws)', '  accuracy, P, R, F1, AUROC'], 'eval', fs=8.3)
box(43, 1, 36, 27, 'Baselines (same splits)', ['label frequency; gene-only', 'logistic regression; classifier chain', 'multi-output XGBoost, MLP,', '  residual NN, hybrid',
    'two-branch ensemble; stacking only', 'paired t-tests, 95% t-CIs'], 'eval', fs=8.3)
box(82, 1, 36, 27, 'Leakage & confound checks', ['gene overlap per split', 'label-source classifier (AUROC 0.97)', 'results split by label source', 'PharmGKB-only / no-tramadol',
    'ablations of every prior'], 'eval', fs=8.3)
box(121, 1, 36, 27, 'Explainability', ['TreeSHAP (exact) on XGBoost branch,', '  15 feature groups', 'TCAV on residual-NN block 2:', '  7 concepts, 30 + 30 CAVs,', '  Welch t-test, BH-FDR',
    'PharmGKB 1A/1B ranking check'], 'eval', fs=8.3)
arrow(132, 32, 132, 28)
fig.savefig('figures/Figure1_pipeline.png', dpi=300, bbox_inches='tight')
fig.savefig('figures/Figure1_pipeline_preview.png', dpi=80, bbox_inches='tight')
print('ok')
