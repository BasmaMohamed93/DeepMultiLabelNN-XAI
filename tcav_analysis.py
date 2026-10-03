"""Supporting analyses for the final model :
1) split statistics, 2) curated-evidence (PharmGKB level 1A/1B) sanity check, 3) blend weights,
4) TreeSHAP feature-group attribution of the pairwise XGBoost (all drugs, random split, seed 0),
5) TCAV on the Residual-NN component (variant features), 6) comparison table of every approach."""
import sys, os, json, glob, hashlib, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb
from scipy import stats
from scipy.stats import false_discovery_control
from sklearn.linear_model import LogisticRegression
import pairwise as PW
from build_dataset import build, VepFeatures
from splits import gene_split, random_split
from nn_training import nn_fit
from resnet import gelu, gelu_grad

OUT = {}
KW = {'all': {}, 'no_tramadol': {'exclude_drugs': ('tramadol',)}, 'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}}
DATA = {}
for ds, kw in KW.items():
    V, Y, drugs, flow = build(**kw)
    DATA[ds] = (V, Y, drugs, np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object), flow)

# 1) split statistics ---------------------------------------------------------------------------------------------
rows = []
for ds, (V, Y, drugs, G, flow) in DATA.items():
    for st in ['A_random', 'B_gene_disjoint']:
        for s in range(5):
            tr, va, te = random_split(Y, s) if st == 'A_random' else gene_split(G, Y, s)
            rows.append(dict(dataset=ds, setting=st, seed=s, n_train=len(tr), n_val=len(va), n_test=len(te),
                             genes_train=len(set(G[tr])), genes_test=len(set(G[te])), test_genes_in_train=len(set(G[te]) & set(G[tr])),
                             test_variants_gene_in_train=float(np.isin(G[te], G[tr]).mean()),
                             drugs_without_test_pos=int((Y[te].sum(0) == 0).sum()), test_positive_pairs=int(Y[te].sum())))
SPL = pd.DataFrame(rows); SPL.to_csv('final_split_stats.csv', index=False)
print(SPL.groupby(['dataset', 'setting'])[['n_train', 'n_val', 'n_test', 'genes_test', 'test_genes_in_train', 'test_variants_gene_in_train', 'drugs_without_test_pos']].mean().round(2))

# 2) curated evidence sanity check ------------------------------------------------------------------------------------
L = pd.read_csv('label_records_raw.csv')
hi = L[(L.source == 'PharmGKB_CA') & L.level.astype(str).isin(['1A', '1B'])][['variant', 'drug']].drop_duplicates()
cur = []
for ds in ['all', 'pharmgkb_only']:
    V, Y, drugs, G, _ = DATA[ds]; vidx = {v: i for i, v in enumerate(V.index)}
    pairs = [(vidx[v], drugs.index(d)) for v, d in zip(hi.variant, hi.drug) if v in vidx and d in drugs]
    for s in range(5):
        f = f'results_pair/{ds}_A_random_{s}_dgidb=True_prior=True_sim=True_ens=True.npz'
        gfile = 'preds_v2/' + hashlib.md5(f'{ds}|A_random|{s}|gene_only|functional+consequence+frequency|'.encode()).hexdigest()[:12] + '.npz'
        for name, fn in [('final_ensemble', f), ('gene_only', gfile)]:
            if not os.path.exists(fn): continue
            p = np.load(fn); te = list(p['te']); pos = {v: k for k, v in enumerate(te)}
            ranks = []
            for i, j in pairs:
                if i in pos:
                    row = p['Ps'][pos[i]]; ranks.append(int((row > row[j]).sum()) + 1)
            ranks = np.array(ranks)
            cur.append(dict(dataset=ds, seed=s, model=name, n_pairs=len(ranks), top1=float((ranks <= 1).mean()),
                            top3=float((ranks <= 3).mean()), top5=float((ranks <= 5).mean()), median_rank=float(np.median(ranks))))
CUR = pd.DataFrame(cur); CUR.to_csv('final_curated_check.csv', index=False)
print(CUR.groupby(['dataset', 'model'])[['n_pairs', 'top1', 'top3', 'top5', 'median_rank']].mean().round(3))

# 3) blend weights -----------------------------------------------------------------------------------------------------
BW = pd.DataFrame([json.load(open(f)) for f in glob.glob('results_pair/*ens=True.json')])[['dataset', 'setting', 'seed', 'blend_weight_xgb', 'decoding', 'force_top1']]
BW.to_csv('final_blend_weights.csv', index=False); print(BW.groupby(['dataset', 'setting']).blend_weight_xgb.describe()[['mean', 'min', 'max']])

# 4) TreeSHAP on the pairwise XGBoost (all, A, seed 0) -----------------------------------------------------------------
V, Y, drugs, G, _ = DATA['all']; seed = 0; nd = len(drugs)
tr, va, te = random_split(Y, seed)
fb = VepFeatures().fit(V.iloc[tr]); (Xt, vnames), (Xv, _), (Xs, _) = (fb.transform(V.iloc[i]) for i in (tr, va, te))
K = PW.pair_features(G, drugs, *PW.DG)
S = np.zeros((len(G), nd, 2), np.float32); fold = np.random.default_rng(seed + 7).integers(0, 5, len(tr))
for k in range(5):
    S[tr[fold == k]] = PW.similarity_features(G[tr[fold == k]], drugs, Y, tr[fold != k], *PW.DS_SETS, G)
rest = np.r_[va, te]; S[rest] = PW.similarity_features(G[rest], drugs, Y, tr, *PW.DS_SETS, G)
K = np.concatenate([K, S], 2)
Gt, Gv, Gs, p0 = PW.gene_prior(G, Y, tr, va, te, seed)
At, Av, As = PW.to_pairs(Xt, K[tr], Gt, p0, nd), PW.to_pairs(Xv, K[va], Gv, p0, nd), PW.to_pairs(Xs, K[te], Gs, p0, nd)
m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=4, min_child_weight=2, subsample=0.8, colsample_bytree=0.6,
                      tree_method='hist', max_bin=128, early_stopping_rounds=50, eval_metric='logloss', n_jobs=2, random_state=seed)
m.fit(At, Y[tr].ravel(), eval_set=[(Av, Y[va].ravel())], verbose=False)
fnames = vnames + [f'drug={d}' for d in drugs] + ['DGIdb_interaction', 'DGIdb_n_sources', 'DGIdb_score', 'DGIdb_inhibitor',
                                                   'DGIdb_gene_degree', 'DGIdb_drug_degree', 'sim_gene_prior', 'sim_drug_prior', 'gene_prior', 'drug_prevalence']


def grp(n):
    if n.startswith('drug='): return 'Drug identity'
    if n.startswith('DGIdb'): return 'DGIdb gene-drug knowledge'
    if n.startswith('sim_gene'): return 'Similar-gene prior (DGIdb)'
    if n.startswith('sim_drug'): return 'Similar-drug prior (DGIdb targets)'
    if n == 'gene_prior': return 'Gene prior (training labels)'
    if n == 'drug_prevalence': return 'Drug prevalence'
    for k, v in [('gnomad', 'Population frequency'), (':af', 'Population frequency'), ('sift', 'SIFT/PolyPhen'), ('polyphen', 'SIFT/PolyPhen'),
                 ('cadd', 'CADD'), ('blosum', 'BLOSUM62'), ('aa_', 'Amino-acid change'), ('distance', 'Distance to gene')]:
        if k in n.lower(): return v
    return 'Consequence / impact / class'


rng = np.random.default_rng(0); sub = rng.choice(len(As), 30000, replace=False)
cb = m.get_booster().predict(xgb.DMatrix(As[sub]), pred_contribs=True)[:, :-1]
a = pd.Series(np.abs(cb).mean(0), index=fnames).groupby(grp).sum().sort_values(ascending=False)
SH = (a / a.sum()).rename('share_of_mean_abs_SHAP').reset_index().rename(columns={'index': 'feature_group'})
SH.to_csv('final_treeshap_groups.csv', index=False); print(SH)
top = pd.Series(np.abs(cb).mean(0), index=fnames).sort_values(ascending=False).head(20)
top.rename('mean_abs_shap').to_csv('final_treeshap_top20.csv'); print(top.head(12))

# 5) TCAV on the Residual-NN component (variant features only, as used inside the hybrid) ---------------------------
nn, hpn = nn_fit(Xt, Y[tr], Xv, Y[va], seed)
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
Ht, Gr = act(Xt), grads(act(Xs)); NREP, NEX = 30, 150
def cav(a_, b_):
    w = LogisticRegression(C=0.1, max_iter=500).fit(np.vstack([Ht[a_], Ht[b_]]), np.r_[np.ones(len(a_)), np.zeros(len(b_))]).coef_[0]
    return w / np.linalg.norm(w)
def score(v):
    dd = np.einsum('nhl,h->nl', Gr, v)
    return np.array([(dd[Y[te][:, j] == 1, j] > 0).mean() if Y[te][:, j].sum() else np.nan for j in range(nd)])
rand = np.array([score(cav(rng.choice(len(Ht), NEX, replace=False), rng.choice(len(Ht), NEX, replace=False))) for _ in range(NREP)])
rows = []
for c, mask in C.items():
    pool = np.where(mask)[0]
    sc = np.array([score(cav(rng.choice(pool, NEX, replace=len(pool) < NEX), rng.choice(len(Ht), NEX, replace=False))) for _ in range(NREP)])
    for j, d in enumerate(drugs):
        if np.isnan(sc[:, j]).all(): continue
        t, p = stats.ttest_ind(sc[:, j], rand[:, j], equal_var=False)
        rows.append(dict(concept=c, n_concept_train=int(mask.sum()), drug=d, tcav=sc[:, j].mean(), tcav_sd=sc[:, j].std(), random=rand[:, j].mean(), t=t, p=p))
T = pd.DataFrame(rows); T['p_bh'] = false_discovery_control(T.p.fillna(1).values); T.to_csv('final_tcav.csv', index=False)
TS = T.groupby('concept').agg(n_concept_train=('n_concept_train', 'first'), n_drugs=('drug', 'size'), sig_up=('p_bh', lambda s: int(((s < .05) & (T.loc[s.index, 't'] > 0)).sum())),
                              sig_down=('p_bh', lambda s: int(((s < .05) & (T.loc[s.index, 't'] < 0)).sum())), mean_tcav=('tcav', 'mean')).reset_index()
TS.to_csv('final_tcav_summary.csv', index=False); print(TS)
print('TCAV examples'); print(T[T.drug.isin(['fluorouracil', 'warfarin', 'clopidogrel', 'tramadol', 'duloxetine'])].sort_values('p_bh').head(15).round(4).to_string())
