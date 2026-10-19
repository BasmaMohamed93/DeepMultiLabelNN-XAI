"""Seed stability of TreeSHAP feature-group shares and TCAV results (main dataset, known genes, seeds 0-4),
and a TCAV sensitivity analysis without resampling (concept and random sets of equal size, drawn without replacement)."""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb
from scipy import stats
from scipy.stats import false_discovery_control
from sklearn.linear_model import LogisticRegression
import pairwise as PW
from build_dataset import build, VepFeatures
from splits import random_split
from nn_training import nn_fit
from resnet import gelu, gelu_grad
from model import regional_prior, knn_prior, WINDOWS

V, Y, drugs, _ = build(); G = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object); nd = len(drugs)
chrom = V.seq_region_name.astype(str).values; pos = V.start.astype(float).values

def grp(n):
    if n.startswith('drug='): return 'Drug identity'
    if n.startswith('DGIdb'): return 'DGIdb gene-drug knowledge'
    if n.startswith('sim_gene'): return 'Similar-gene prior (DGIdb)'
    if n.startswith('sim_drug'): return 'Similar-drug prior (DGIdb targets)'
    if n.startswith('regional'): return 'Regional genomic prior'
    if n.startswith('knn'): return 'Feature-space kNN prior'
    if n == 'gene_prior': return 'Gene prior (training labels)'
    if n == 'drug_prevalence': return 'Drug prevalence'
    for k, v in [('gnomad', 'Population frequency'), (':af', 'Population frequency'), ('sift', 'SIFT/PolyPhen'), ('polyphen', 'SIFT/PolyPhen'),
                 ('cadd', 'CADD'), ('blosum', 'BLOSUM62'), ('aa_', 'Amino-acid change'), ('distance', 'Distance to gene')]:
        if k in n.lower(): return v
    return 'Consequence / impact / class'

SH, TC = [], []
for seed in range(5):
    tr, va, te = random_split(Y, seed)
    fb = VepFeatures().fit(V.iloc[tr]); (Xt, vnames), (Xv, _), (Xs, _) = (fb.transform(V.iloc[i]) for i in (tr, va, te))
    # ---- TreeSHAP of the XGBoost branch
    K = PW.pair_features(G, drugs, *PW.DG)
    S = np.zeros((len(G), nd, 2 + 2 * len(WINDOWS)), np.float32); fold = np.random.default_rng(seed + 7).integers(0, 5, len(tr))
    for k in range(5):
        q, rr = tr[fold == k], tr[fold != k]
        S[q, :, :2] = PW.similarity_features(G[q], drugs, Y, rr, *PW.DS_SETS, G); S[q, :, 2:] = regional_prior(chrom, pos, Y, q, rr)
    rest = np.r_[va, te]; S[rest, :, :2] = PW.similarity_features(G[rest], drugs, Y, tr, *PW.DS_SETS, G); S[rest, :, 2:] = regional_prior(chrom, pos, Y, rest, tr)
    Xall = np.zeros((len(G), Xt.shape[1]), np.float32); Xall[tr], Xall[va], Xall[te] = Xt, Xv, Xs
    Kn = np.zeros((len(G), nd, 3), np.float32)
    for k in range(5):
        q, rr = tr[fold == k], tr[fold != k]; Kn[q] = knn_prior(Xall, G, Y, q, rr)
    Kn[rest] = knn_prior(Xall, G, Y, rest, tr); K = np.concatenate([K, S, Kn], 2)
    Gt, Gv, Gs, p0 = PW.gene_prior(G, Y, tr, va, te, seed)
    At, Av, As = PW.to_pairs(Xt, K[tr], Gt, p0, nd), PW.to_pairs(Xv, K[va], Gv, p0, nd), PW.to_pairs(Xs, K[te], Gs, p0, nd)
    best = None
    for depth in (4, 6):
        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.04, max_depth=depth, min_child_weight=2, subsample=0.8, colsample_bytree=0.6, tree_method='hist',
                              max_bin=128, early_stopping_rounds=60, eval_metric='logloss', n_jobs=2, random_state=seed).fit(At, Y[tr].ravel(), eval_set=[(Av, Y[va].ravel())], verbose=False)
        if best is None or m.best_score < best[0]: best = (m.best_score, m)
    m = best[1]
    fnames = vnames + [f'drug={d}' for d in drugs] + ['DGIdb_interaction', 'DGIdb_n_sources', 'DGIdb_score', 'DGIdb_inhibitor', 'DGIdb_gene_degree', 'DGIdb_drug_degree',
                                                      'sim_gene_prior', 'sim_drug_prior'] + [f'regional_{t}_{int(w)}' for w in WINDOWS for t in ('rate', 'logcount')] + \
             ['knn10_prior', 'knn30_prior', 'knn_same_gene_prior', 'gene_prior', 'drug_prevalence']
    sub = np.random.default_rng(0).choice(len(As), 30000, replace=False)
    cb = m.get_booster().predict(xgb.DMatrix(As[sub]), pred_contribs=True)[:, :-1]
    a = pd.Series(np.abs(cb).mean(0), index=fnames).groupby(grp).sum(); a = a / a.sum()
    for g_, v_ in a.items(): SH.append(dict(seed=seed, group=g_, share=v_))
    # ---- TCAV on the residual network
    nn, _ = nn_fit(Xt, Y[tr], Xv, Y[va], seed)
    Lr = nn.body.layers; nb = nn.n_blocks
    def act(X):
        h = X
        for l in Lr[:3 + nb]: h = l.fwd(h, False)
        return h
    def grads(H):
        lin, bn = Lr[3 + nb], Lr[4 + nb]
        z = bn.fwd(H @ lin.p['W'] + lin.p['b'], False); s = bn.p['gamma'] * bn.inv
        _, t = gelu(z); d = gelu_grad(z, t) * s
        return np.einsum('nk,hk,kl->nhl', d, lin.p['W'], nn.head.layers[1].p['W'], optimize=True)
    Vt = V.iloc[tr]; csq = Vt.most_severe_consequence.astype(str)
    C = {'loss_of_function': csq.isin(['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost']).values,
         'missense': (csq == 'missense_variant').values,
         'damaging_missense': (Vt.sift_prediction.astype(str).str.startswith('deleterious') & Vt.polyphen_prediction.astype(str).str.contains('damaging')).values,
         'high_CADD_ge20': (pd.to_numeric(Vt.cadd_phred, errors='coerce') >= 20).values,
         'rare_gnomAD_lt1pct': (pd.to_numeric(Vt.gnomadg, errors='coerce').fillna(0) < 0.01).values,
         'regulatory_UTR_upstream': csq.isin(['3_prime_UTR_variant', '5_prime_UTR_variant', 'upstream_gene_variant']).values,
         'intronic': (csq == 'intron_variant').values}
    Ht, Gr = act(Xt), grads(act(Xs)); rng = np.random.default_rng(seed)
    def cav(a_, b_):
        w = LogisticRegression(C=0.1, max_iter=500).fit(np.vstack([Ht[a_], Ht[b_]]), np.r_[np.ones(len(a_)), np.zeros(len(b_))]).coef_[0]
        return w / np.linalg.norm(w)
    def score(v):
        dd = np.einsum('nhl,h->nl', Gr, v)
        return np.array([(dd[Y[te][:, j] == 1, j] > 0).mean() if Y[te][:, j].sum() else np.nan for j in range(nd)])
    for mode in ('resample', 'no_replacement'):
        rows_mode = []
        for c, mask in C.items():
            pool = np.where(mask)[0]
            nex = 150 if mode == 'resample' else min(150, len(pool) // 2 * 2 if len(pool) < 150 else 150)
            if mode == 'no_replacement': nex = min(150, len(pool))
            rand = np.array([score(cav(rng.choice(len(Ht), nex, replace=False), rng.choice(len(Ht), nex, replace=False))) for _ in range(30)])
            sc = np.array([score(cav(rng.choice(pool, nex, replace=(mode == 'resample' and len(pool) < nex)), rng.choice(len(Ht), nex, replace=False))) for _ in range(30)])
            for j, d in enumerate(drugs):
                if np.isnan(sc[:, j]).all(): continue
                t, p = stats.ttest_ind(sc[:, j], rand[:, j], equal_var=False)
                rows_mode.append(dict(seed=seed, mode=mode, concept=c, n_examples=nex, drug=d, tcav=sc[:, j].mean(), t=t, p=p))
        R = pd.DataFrame(rows_mode); R['p_bh'] = false_discovery_control(R.p.fillna(1).values); TC.append(R)
    print('seed', seed, 'done', flush=True)
SH = pd.DataFrame(SH); TC = pd.concat(TC)
SH.to_csv('r2_treeshap_seeds.csv', index=False); TC.to_csv('r2_tcav_seeds.csv', index=False)
SH['rank'] = SH.groupby('seed').share.rank(ascending=False)
print(SH.groupby('group').agg(mean_share=('share', 'mean'), sd_share=('share', 'std'), mean_rank=('rank', 'mean'), min_rank=('rank', 'min'), max_rank=('rank', 'max')).sort_values('mean_rank').round(3).to_string())
TC['sig'] = TC.p_bh < 0.05
print(TC.groupby(['mode', 'seed']).sig.sum().unstack().to_string())
