"""Figures for the second revision: Figure 5 (TreeSHAP and TCAV over five seeds), Supplementary Fig. S1 (precision-recall curves),
Supplementary Fig. S2 (calibration curves)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, average_precision_score
from build_dataset import build
from splits import random_split, gene_split
from pairwise import gene_prior
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#8a8986', 'axes.linewidth': 0.8,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b', 'figure.facecolor': 'white'})
BLUE, ORANGE, GRID = '#2a78d6', '#eb6834', '#e4e3df'; OUT = 'figures/'; os.makedirs(OUT, exist_ok=True)
# ---------------- Figure 5 ----------------
SH = pd.read_csv('r2_treeshap_seeds.csv'); S = SH.groupby('group').share.agg(['mean', 'std']).sort_values('mean')
T = pd.read_csv('r2_tcav_seeds.csv'); T = T[T['mode'] == 'resample'].copy(); T['up'] = (T.p_bh < 0.05) & (T.t > 0); T['dn'] = (T.p_bh < 0.05) & (T.t < 0)
G = T.groupby(['drug', 'concept']).agg(tcav=('tcav', 'mean'), up=('up', 'sum'), dn=('dn', 'sum')).reset_index()
G['robust'] = (G.up >= 4) | (G.dn >= 4); G.to_csv('r2_tcav_consensus.csv', index=False)
drugs = ['fluorouracil', 'warfarin', 'clopidogrel', 'methotrexate', 'tramadol', 'duloxetine']
concepts = ['loss_of_function', 'damaging_missense', 'missense', 'high_CADD_ge20', 'rare_gnomAD_lt1pct', 'regulatory_UTR_upstream', 'intronic']
clab = ['Loss-of-function', 'Damaging missense', 'Missense', 'CADD >= 20', 'gnomAD AF < 1%', 'UTR / upstream', 'Intronic']
fig = plt.figure(figsize=(14.5, 5.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.6)
ax = fig.add_subplot(gs[0])
cols = ['#9A7B00' if ('Regional' in g or 'kNN' in g) else BLUE for g in S.index]
ax.barh(S.index, S['mean'] * 100, xerr=S['std'] * 100, color=cols, height=0.7, zorder=3, error_kw=dict(ecolor='#333', lw=0.8, capsize=2))
for y, (v, e) in enumerate(zip(S['mean'] * 100, S['std'] * 100)): ax.text(v + e + 0.6, y, f'{v:.1f}%', va='center', fontsize=8.5)
ax.set_xlabel('Share of mean |SHAP| (%), mean ± SD over five seeds')
ax.set_title('a  TreeSHAP feature-group attribution\n    (XGBoost branch; ochre = graph-propagated priors)', loc='left', fontweight='bold', fontsize=11)
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlim(0, 37)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax2 = fig.add_subplot(gs[1]); Mx = np.full((len(drugs), len(concepts)), np.nan); Sig = np.zeros_like(Mx, bool)
for i, d in enumerate(drugs):
    for j, c in enumerate(concepts):
        r = G[(G.drug == d) & (G.concept == c)]
        if len(r): Mx[i, j] = r.tcav.iloc[0]; Sig[i, j] = bool(r.robust.iloc[0])
cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
im = ax2.imshow(Mx, cmap=cmap, vmin=0, vmax=1, aspect='auto')
for i in range(len(drugs)):
    for j in range(len(concepts)): ax2.text(j, i, f'{Mx[i, j]:.2f}' + ('*' if Sig[i, j] else ''), ha='center', va='center', fontsize=8.5)
ax2.set_xticks(range(len(concepts))); ax2.set_xticklabels(clab, rotation=35, ha='right', fontsize=9)
ax2.set_yticks(range(len(drugs))); ax2.set_yticklabels([d.capitalize() for d in drugs], fontsize=9.5)
ax2.set_xticks(np.arange(-.5, len(concepts)), minor=True); ax2.set_yticks(np.arange(-.5, len(drugs)), minor=True)
ax2.grid(which='minor', color='white', lw=2); ax2.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02); cb.set_label('Mean TCAV score over five seeds', fontsize=8)
ax2.set_title('b  TCAV scores (residual-network branch, five seeds)\n    * significant in >= 4 of 5 seeds, consistent direction', loc='left', fontweight='bold', fontsize=11)
fig.savefig(OUT + 'Figure5_explanations.png', dpi=300, bbox_inches='tight'); plt.close(fig)
# ---------------- Supplementary Fig. S1 and S2 ----------------
V, Y, D, _ = build(); Gn = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
P = {}; 
for st in ('A_random', 'B_gene_disjoint'):
    ys, ps, gs_ = [], [], []
    for sd in range(5):
        z = np.load(f'results/all_{st}_{sd}_region=True_rank=False_knn=True.npz'); te = z['te']
        tr, va, _ = random_split(Y, sd) if st == 'A_random' else gene_split(Gn, Y, sd)
        ys.append(Y[te]); ps.append(z['Ps']); gs_.append(gene_prior(Gn, Y, tr, va, te, sd)[2])
    P[st] = (np.vstack(ys), np.vstack(ps), np.vstack(gs_))
sel = ['warfarin', 'clopidogrel', 'fluorouracil', 'methotrexate', 'tacrolimus', 'tramadol']
sel = [d for d in sel if d in D]
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
Yt, Pt, Gt = P['A_random']
for ax, d in zip(axes.ravel(), sel):
    j = D.index(d)
    for sc, col, lab in ((Pt[:, j], BLUE, 'DeepMultiLabelNN-XAI'), (Gt[:, j], ORANGE, 'Gene-only baseline')):
        pr, rc, _ = precision_recall_curve(Yt[:, j], sc); ax.step(rc, pr, where='post', color=col, lw=1.6, label=f'{lab} (AP {average_precision_score(Yt[:, j], sc):.2f})')
    ax.axhline(Yt[:, j].mean(), color='#888', ls='--', lw=0.8)
    ax.set_title(d.capitalize(), loc='left', fontweight='bold'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc='lower left'); ax.grid(color=GRID, lw=0.6)

fig.tight_layout(); fig.savefig(OUT + 'FigureS1_precision_recall.png', dpi=300, bbox_inches='tight'); plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, st, ttl in zip(axes, ('A_random', 'B_gene_disjoint'), ('a  New variants in known genes', 'b  Unseen genes (gene-disjoint)')):
    Yt, Pt, Gt = P[st]; y = Yt.ravel()
    for sc, col, lab in ((Pt.ravel(), BLUE, 'DeepMultiLabelNN-XAI'), (Gt.ravel(), ORANGE, 'Gene-only baseline')):
        q = np.quantile(sc, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, sc, side='right') - 1, 0, 9)
        mp = [sc[b == k].mean() for k in range(10) if (b == k).any()]; ob = [y[b == k].mean() for k in range(10) if (b == k).any()]
        ax.plot(mp, ob, 'o-', color=col, lw=1.4, ms=4, label=lab)
    m = max(ax.get_xlim()[1], ax.get_ylim()[1]); ax.plot([0, m], [0, m], color='#888', ls='--', lw=0.8)
    ax.set_xscale('symlog', linthresh=0.01); ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xlabel('Mean predicted probability (decile bins)'); ax.set_ylabel('Observed frequency'); ax.set_title(ttl, loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=8); ax.grid(color=GRID, lw=0.6)

fig.tight_layout(); fig.savefig(OUT + 'FigureS2_calibration.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('figures done')
