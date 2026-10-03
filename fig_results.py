import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#8a8986', 'axes.linewidth': 0.8,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b', 'figure.facecolor': 'white'})
BLUE, AQUA, ORANGE, GRID, TXT2 = '#2a78d6', '#1baf7a', '#eb6834', '#e4e3df', '#52514e'
OUT = 'figures/'
D = pd.read_csv('all_runs.csv')
models = [('proposed', 'DeepMultiLabelNN-XAI (proposed)', BLUE), ('two_branch_ensemble', 'Two-branch ensemble (no stacking)', AQUA), ('gene_only', 'Gene-only baseline', ORANGE)]
panels = [('all', 'A_random', 'All evidence\nknown genes'), ('all', 'B_gene_disjoint', 'All evidence\nunseen genes'),
          ('high_evidence', 'A_random', 'High-confidence\nknown genes'), ('high_evidence', 'B_gene_disjoint', 'High-confidence\nunseen genes'),
          ('pharmgkb_only', 'A_random', 'PharmGKB-only\nknown genes'), ('pharmgkb_only', 'B_gene_disjoint', 'PharmGKB-only\nunseen genes')]

# ---------------- Figure 2 ----------------
fig, axes = plt.subplots(3, 1, figsize=(13, 12.5))
for ax, (metric, title, ylim) in zip(axes, [('macro_auroc', 'a  Macro AUROC (multi-label protocol)', (0, 1.08)),
                                              ('micro_f1', 'b  Micro F1 (multi-label protocol)', (0, 1.0)),
                                              ('bal_f1', 'c  F1 (link-prediction protocol, 1 positive : 1 negative; not computed for the two-branch ensemble)', (0, 1.08))]):
    x = np.arange(len(panels)); w = 0.26
    ms = models if metric != 'bal_f1' else [models[0], models[2]]
    for i, (mk, lab, col) in enumerate(ms):
        vals, sds = [], []
        for ds, st, _ in panels:
            s = D[(D.dataset == ds) & (D.setting == st) & (D.model == mk)][metric].dropna()
            vals.append(s.mean() if len(s) else np.nan); sds.append(s.std() if len(s) else np.nan)
        pos = x + (i - (len(ms) - 1) / 2) * w
        ax.bar(pos, vals, w - 0.03, color=col, label=lab, zorder=3)
        ax.errorbar(pos, vals, yerr=sds, fmt='none', ecolor='#3a3a38', elinewidth=0.9, capsize=2, zorder=4)
        for p_, v, sd in zip(pos, vals, sds):
            if v == v: ax.text(p_, v + (sd if sd == sd else 0) + 0.012, f'{v:.2f}', ha='center', va='bottom', fontsize=7.6, color='#0b0b0b')
            else: ax.text(p_, 0.02, 'n/a', ha='center', va='bottom', fontsize=7, color=TXT2, rotation=90)
    if metric != 'micro_f1':
        ax.axhline(0.5, color='#8a8986', lw=1, ls='--', zorder=2); ax.text(-0.5, 0.51, 'chance', fontsize=8, color=TXT2, ha='left', va='bottom')
    if metric == 'bal_f1':
        ax.axhline(0.9, color='#b0aea8', lw=0.8, ls=':', zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([p[2] for p in panels], fontsize=9)
    ax.set_ylim(*ylim); ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT + 'Figure2_performance.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 3: Hit@k ----------------
C = pd.read_csv('topk_curve.csv')
fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharey=True)
for ax, (ds, st, ttl) in zip(axes.ravel(), [('all', 'A_random', 'a  All evidence, new variants in known genes'), ('all', 'B_gene_disjoint', 'b  All evidence, variants in unseen genes'),
                                              ('high_evidence', 'A_random', 'c  High-confidence, known genes'), ('high_evidence', 'B_gene_disjoint', 'd  High-confidence, unseen genes')]):
    for mk, lab, col in [('proposed', 'DeepMultiLabelNN-XAI', BLUE), ('gene_only', 'Gene-only baseline', ORANGE)]:
        s = C[(C.dataset == ds) & (C.setting == st) & (C.model == mk)].groupby('k').hit
        mu, sd = s.mean(), s.std()
        ax.plot(mu.index, mu.values, color=col, lw=2, marker='o', ms=4, label=lab, zorder=3)
        ax.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        for k in (1, 5, 10):
            ax.annotate(f'{mu[k]:.2f}', (k, mu[k]), textcoords='offset points', xytext=(0, 7 if mk == 'proposed' else -13), ha='center', fontsize=8, color='#0b0b0b')
    nd = 60 if ds == 'all' else 20
    ax.set_xlabel(f'k (top-ranked drugs out of {nd})'); ax.set_xticks([1, 3, 5, 10, 15, 20]); ax.set_ylim(0, 1.05)
    ax.set_title(ttl, loc='left', fontweight='bold', fontsize=10.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
axes[0, 0].set_ylabel('Hit@k'); axes[1, 0].set_ylabel('Hit@k')
axes[0, 1].legend(frameon=False, loc='lower right', fontsize=9)
fig.tight_layout(); fig.savefig(OUT + 'Figure3_topk.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 4: TreeSHAP  + TCAV ----------------
S = pd.read_csv('treeshap_groups.csv').sort_values('share_of_mean_abs_SHAP')
T = pd.read_csv('final_tcav.csv')
drugs = ['fluorouracil', 'warfarin', 'clopidogrel', 'methotrexate', 'tramadol', 'duloxetine']
concepts = ['loss_of_function', 'damaging_missense', 'missense', 'high_CADD_ge20', 'rare_gnomAD_lt1pct', 'regulatory_UTR_upstream', 'intronic']
clab = ['Loss-of-function', 'Damaging missense', 'Missense', 'CADD >= 20', 'gnomAD AF < 1%', 'UTR / upstream', 'Intronic']
fig = plt.figure(figsize=(14.5, 5.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.6)
ax = fig.add_subplot(gs[0])
cols = ['#9A7B00' if ('Regional' in g or 'kNN' in g) else BLUE for g in S.feature_group]
ax.barh(S.feature_group, S.share_of_mean_abs_SHAP * 100, color=cols, height=0.7, zorder=3)
for y, v in enumerate(S.share_of_mean_abs_SHAP * 100): ax.text(v + 0.5, y, f'{v:.1f}%', va='center', fontsize=8.5)
ax.set_xlabel('Share of mean |SHAP| (%)'); ax.set_title('a  TreeSHAP feature-group attribution\n    (XGBoost branch; ochre = graph-propagated priors)', loc='left', fontweight='bold', fontsize=11)
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlim(0, 36)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax2 = fig.add_subplot(gs[1])
Mx = np.full((len(drugs), len(concepts)), np.nan); Sig = np.zeros_like(Mx, bool)
for i, d in enumerate(drugs):
    for j, c in enumerate(concepts):
        r = T[(T.drug == d) & (T.concept == c)]
        if len(r): Mx[i, j] = r.tcav.iloc[0]; Sig[i, j] = r.p_bh.iloc[0] < 0.05
cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
im = ax2.imshow(Mx, cmap=cmap, vmin=0, vmax=1, aspect='auto')
for i in range(len(drugs)):
    for j in range(len(concepts)):
        ax2.text(j, i, f'{Mx[i, j]:.2f}' + ('*' if Sig[i, j] else ''), ha='center', va='center', fontsize=8.5, color='#0b0b0b')
ax2.set_xticks(range(len(concepts))); ax2.set_xticklabels(clab, rotation=35, ha='right', fontsize=9)
ax2.set_yticks(range(len(drugs))); ax2.set_yticklabels([d.capitalize() for d in drugs], fontsize=9.5)
ax2.set_xticks(np.arange(-.5, len(concepts)), minor=True); ax2.set_yticks(np.arange(-.5, len(drugs)), minor=True)
ax2.grid(which='minor', color='white', lw=2); ax2.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02); cb.set_label('TCAV score (share of positive variants whose\ndrug score increases along the concept)', fontsize=8)
ax2.set_title('b  TCAV scores (residual-NN branch)\n    * BH-adjusted p < 0.05 vs 30 random CAVs', loc='left', fontweight='bold', fontsize=11)
fig.savefig(OUT + 'Figure4_explanations.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('ok')
