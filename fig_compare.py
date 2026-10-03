import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.edgecolor':'#8a8986','xtick.color':'#52514e','ytick.color':'#333333'})
rows=[('Label frequency',0.500,0.500,0.355,0.173),('Gene-only',0.816,0.500,0.607,0.173),('Logistic regression',0.761,0.668,0.128,0.084),
('Classifier chain',0.737,0.658,0.139,0.092),('Non-residual MLP',0.771,0.674,0.134,0.079),('Residual NN',0.760,0.671,0.121,0.079),
('XGBoost',0.805,0.695,0.322,0.146),('Hybrid ResNN->XGB',0.780,0.678,0.296,0.150),('XGBoost + gene prior',0.894,0.661,0.559,0.103),
('Hybrid + gene prior',0.862,0.635,0.547,0.123),('Pairwise XGB + prior',0.897,0.655,0.616,0.206),('Pairwise XGB + DGIdb + prior',0.908,0.701,0.629,0.288),
('Pairwise hybrid + DGIdb + prior',0.901,0.696,0.617,0.244),('+ similarity priors',0.912,0.725,0.631,0.226),('Two-branch ensemble (no stacking)',0.913,0.723,0.621,0.281)]
import json; N=json.load(open('numbers.json')); A=N['all|A_random']; B=N['all|B_gene_disjoint']; fl=float
rows[0]=('Label frequency',0.5,0.5,fl(A['label_freq']['f1']),fl(B['label_freq']['f1'])); rows[1]=('Gene-only',fl(A['gene_only']['macro_auroc']),0.5,fl(A['gene_only']['micro_f1']),fl(B['gene_only']['micro_f1']))
rows.insert(2,('Graph label propagation',fl(A['labelprop']['macro_auroc']),fl(B['labelprop']['macro_auroc']),fl(A['labelprop']['micro_f1']),fl(B['labelprop']['micro_f1'])))
rows.append(('Three-branch stacking (no graph priors)',fl(A['stacking_no_graph_priors']['macro_auroc']),fl(B['stacking_no_graph_priors']['macro_auroc']),fl(A['stacking_no_graph_priors']['micro_f1']),fl(B['stacking_no_graph_priors']['micro_f1'])))
rows.append(('DeepMultiLabelNN-XAI (proposed)',fl(A['proposed']['macro_auroc']['mean']),fl(B['proposed']['macro_auroc']['mean']),fl(A['proposed']['micro_f1']['mean']),fl(B['proposed']['micro_f1']['mean'])))
names=[r[0] for r in rows][::-1]; y=np.arange(len(rows))
fig,axes=plt.subplots(1,2,figsize=(13,7.4),sharey=True)
for ax,(i,j,t,xl) in zip(axes,[(1,2,'a  Macro AUROC',(0.45,0.95)),(3,4,'b  Micro F1',(0,0.7))]):
    k=[r[i] for r in rows][::-1]; u=[r[j] for r in rows][::-1]
    for yy,a,b in zip(y,k,u): ax.plot([a,b],[yy,yy],color='#cfcdc8',lw=2,zorder=1)
    ax.scatter(k,y,s=46,color='#2a78d6',label='Known genes',zorder=3); ax.scatter(u,y,s=46,color='#eb6834',marker='D',label='Unseen genes',zorder=3)
    for yy,a in zip(y,k): ax.text(a+0.008,yy+0.18,f'{a:.3f}',fontsize=7.2,color='#0b0b0b')
    for yy,b in zip(y,u): ax.text(b+0.008,yy-0.42,f'{b:.3f}',fontsize=7.2,color='#52514e')
    ax.set_xlim(*xl); ax.set_title(t,loc='left',fontweight='bold',fontsize=11); ax.xaxis.grid(True,color='#e4e3df'); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.axhspan(-0.5,0.5,color='#EEF3FA',zorder=0)
axes[0].set_yticks(y); axes[0].set_yticklabels(names)
axes[0].get_yticklabels()[0].set_fontweight('bold')
h,l=axes[1].get_legend_handles_labels(); fig.legend(h,l,frameon=False,loc='lower center',ncol=2,bbox_to_anchor=(0.55,-0.03))
fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig('figures/FigureS1_approach_comparison.png',dpi=300,bbox_inches='tight')
