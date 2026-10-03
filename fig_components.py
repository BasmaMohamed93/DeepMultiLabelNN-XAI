"""Figure 2: hybrid framework vs XGBoost alone vs residual network alone (link-prediction protocol, mean ± SD over five splits)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#8a8986', 'axes.linewidth': 0.8,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b', 'figure.facecolor': 'white'})
GRID, TXT, TXT2 = '#e4e3df', '#0b0b0b', '#52514e'
S = pd.read_csv('hybrid_components_summary.csv')
MODELS = [('nn', 'Residual NN alone (DL)', '#1baf7a'), ('xgb', 'XGBoost alone (ML)', '#eb6834'), ('stack', 'Hybrid framework', '#2a78d6')]
METRICS = [('bal_accuracy', 'Accuracy'), ('bal_precision', 'Precision'), ('bal_recall', 'Recall'), ('bal_f1', 'F1-score'), ('bal_auroc', 'AUROC')]
PANELS = [('high_evidence', 'A_random', 'a  High-confidence evidence, new variants in known genes'),
          ('high_evidence', 'B_gene_disjoint', 'b  High-confidence evidence, unseen genes'),
          ('all', 'A_random', 'c  All evidence, new variants in known genes'),
          ('all', 'B_gene_disjoint', 'd  All evidence, unseen genes')]
fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharey=True)
x = np.arange(len(METRICS)); w = 0.26
for ax, (ds, st, title) in zip(axes.ravel(), PANELS):
    for i, (k, lab, col) in enumerate(MODELS):
        r = S[(S.dataset == ds) & (S.setting == st) & (S.component == k)].iloc[0]
        mu = np.array([r[m + '_mean'] for m, _ in METRICS]); sd = np.array([r[m + '_std'] for m, _ in METRICS])
        pos = x + (i - 1) * w
        ax.bar(pos, mu, w - 0.03, color=col, label=lab, zorder=3)
        ax.errorbar(pos, mu, yerr=sd, fmt='none', ecolor='#3a3a38', elinewidth=0.8, capsize=1.8, zorder=4)
        for p_, v in zip(pos, mu):
            ax.text(p_, 0.03, f'{v:.3f}', ha='center', va='bottom', rotation=90, fontsize=7.4,
                    color='white', fontweight='bold' if k == 'stack' else 'normal', zorder=5)
    ax.axhline(0.9, color='#b0aea8', lw=0.8, ls=':', zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in METRICS], fontsize=9.5)
    ax.set_ylim(0, 1.05); ax.set_title(title, loc='left', fontweight='bold', fontsize=10.5, color=TXT)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for ax in axes[:, 0]: ax.set_ylabel('Score (link-prediction protocol)')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig('figures/Figure2_hybrid_vs_components.png', dpi=300, bbox_inches='tight')
print('saved')
