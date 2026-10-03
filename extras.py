"""Supporting analyses for the model (from saved test predictions in results/):
per-drug table, 95% CIs, calibration, Hit@k curves (vs gene-only), curated-evidence check,
split statistics for the high-evidence dataset, label-frequency baseline, TreeSHAP of the XGBoost branch."""
import sys, os, json, glob, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from build_dataset import build, VepFeatures
from splits import gene_split, random_split
import pairwise as PW
from pairwise import decode, dec_metrics, choose_decoding
from evalkit import metrics
from model import regional_prior, knn_prior, balanced_eval, balanced_threshold, WINDOWS

KW = {'all': {}, 'high_evidence': {'drop_levels': ('3',)}, 'no_tramadol': {'exclude_drugs': ('tramadol',)},
      'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}}
DATA = {}
for ds, kw in KW.items():
    V, Y, drugs, flow = build(**kw)
    DATA[ds] = (V, Y, drugs, np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object), flow)


def load(ds, st, s):
    f = f'results/{ds}_{st}_{s}_region=True_rank=False_knn=True'
    return np.load(f + '.npz'), json.load(open(f + '.json'))


def gene_scores(ds, st, s):
    V, Y, drugs, G, _ = DATA[ds]
    tr, va, te = random_split(Y, s) if st == 'A_random' else gene_split(G, Y, s)
    Gt, Gv, Gs, p0 = PW.gene_prior(G, Y, tr, va, te, s)
    return tr, va, te, Gv, Gs, p0


def ece(y, p, bins=10):
    e = 0.0; idx = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


# 1) per-run extras: calibration, Hit@k curve, label-frequency baseline, CI inputs ---------------------------------------
cal, curve, lf, runs = [], [], [], []
for ds in KW:
    V, Y, drugs, G, _ = DATA[ds]
    for st in ['A_random', 'B_gene_disjoint']:
        for s in range(5):
            d, r = load(ds, st, s); te, Ps = d['te'], d['Ps']; Ys = Y[te]
            tr, va, te2, Gv, Gs, p0 = gene_scores(ds, st, s); assert (te2 == te).all()
            for name, P in [('proposed', Ps), ('gene_only', Gs)]:
                cal.append(dict(dataset=ds, setting=st, seed=s, model=name, brier=float(np.mean((P - Ys) ** 2)), ece=ece(Ys.ravel(), P.ravel())))
                o = np.argsort(-P, 1)
                for k in range(1, 21):
                    top = np.zeros_like(Ys, bool); np.put_along_axis(top, o[:, :k], True, 1)
                    curve.append(dict(dataset=ds, setting=st, seed=s, model=name, k=k, hit=float(((top & (Ys > 0)).sum(1) > 0).mean())))
            # label-frequency baseline: every variant scored with training prevalence p0
            Pv0 = np.tile(p0, (len(va), 1)); Ps0 = np.tile(p0, (len(te), 1))
            n0, th0, ft0 = choose_decoding(Y[va], Pv0); m0 = dec_metrics(Ys, decode(Ps0, th0, ft0))
            b0 = balanced_eval(Ys, Ps0 + 1e-9 * np.random.default_rng(s).random(Ps0.shape), balanced_threshold(Y[va], Pv0, np.random.default_rng(s + 100)), np.random.default_rng(s + 200))
            lf.append(dict(dataset=ds, setting=st, seed=s, **{k: m0[k] for k in ['subset_acc', 'precision', 'recall', 'f1']}, bal_accuracy=b0['accuracy'],
                           bal_precision=b0['precision'], bal_recall=b0['recall'], bal_f1=b0['f1'], bal_auroc=b0['auroc']))
            t, b = r['test'], r['balanced']
            runs.append(dict(dataset=ds, setting=st, seed=s, subset_acc=t['subset_acc'], hamming_acc=t['hamming_acc'], precision=t['precision'], recall=t['recall'],
                             micro_f1=t['f1'], sample_f1=t['sample_f1'], macro_auroc=t['macro_auroc'], micro_auroc=t['micro_auroc'], macro_auprc=t['macro_auprc'],
                             within_gene_auroc=t['within_gene_auroc'], **{'hit@1': t['hit@1'], 'hit@5': t['hit@5'], 'hit@10': t['hit@10']},
                             bal_accuracy=b['accuracy'], bal_precision=b['precision'], bal_recall=b['recall'], bal_f1=b['f1'], bal_auroc=b['auroc'], bal_auprc=b['auprc']))
pd.DataFrame(cal).to_csv('calibration.csv', index=False)
pd.DataFrame(curve).to_csv('topk_curve.csv', index=False)
LF = pd.DataFrame(lf); LF.to_csv('label_frequency.csv', index=False)
R = pd.DataFrame(runs); R.to_csv('final_runs.csv', index=False)
ci = []
for (ds, st), g in R.groupby(['dataset', 'setting']):
    for m in R.columns[3:]:
        x = g[m].astype(float); mu, sd = x.mean(), x.std(ddof=1); h = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))
        ci.append(dict(dataset=ds, setting=st, metric=m, mean=mu, sd=sd, ci_low=mu - h, ci_high=mu + h))
pd.DataFrame(ci).round(4).to_csv('ci.csv', index=False)
print(pd.DataFrame(cal).groupby(['dataset', 'setting', 'model'])[['brier', 'ece']].mean().round(4))
print(LF.groupby(['dataset', 'setting']).mean(numeric_only=True).drop(columns='seed').round(3))

# 2) per-drug table (main + high-evidence, known genes) -------------------------------------------------------------
for ds in ['all', 'high_evidence']:
    V, Y, drugs, G, _ = DATA[ds]; rows = []
    for s in range(5):
        d, r = load(ds, 'A_random', s); te, Ps, th = d['te'], d['Ps'], d['th']; B = decode(Ps, th, r['force_top1'])
        _, _, _, _, Gs, _ = gene_scores(ds, 'A_random', s)
        for j, dr in enumerate(drugs):
            y = Y[te, j]
            if y.sum() == 0: continue
            tp, fp, fn = (B[:, j] & (y == 1)).sum(), (B[:, j] & (y == 0)).sum(), (~B[:, j] & (y == 1)).sum()
            pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            rows.append(dict(drug=dr, seed=s, test_pos=int(y.sum()), precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-12),
                             auroc=roc_auc_score(y, Ps[:, j]), auprc=average_precision_score(y, Ps[:, j]), gene_auroc=roc_auc_score(y, Gs[:, j])))
    D = pd.DataFrame(rows).groupby('drug').mean(numeric_only=True).drop(columns='seed')
    D.insert(0, 'variants', [int(Y[:, drugs.index(x)].sum()) for x in D.index]); D.insert(1, 'prevalence', D.variants / len(Y))
    D.insert(2, 'genes', [len(set(G[Y[:, drugs.index(x)] == 1])) for x in D.index])
    D = D.sort_values('variants', ascending=False).round(4); D.to_csv(f'per_drug_{ds}.csv')
    print(ds, 'drugs', len(D), 'AUROC>=0.8', (D.auroc >= 0.8).sum(), '>=0.9', (D.auroc >= 0.9).sum(), 'median', D.auroc.median(), '> gene-only', (D.auroc > D.gene_auroc).sum())

# 3) curated-evidence check (PharmGKB 1A/1B) ------------------------------------------------------------------------
L = pd.read_csv('label_records_raw.csv')
hi = L[(L.source == 'PharmGKB_CA') & L.level.astype(str).isin(['1A', '1B'])][['variant', 'drug']].drop_duplicates()
cur = []
for ds in ['all', 'high_evidence']:
    V, Y, drugs, G, _ = DATA[ds]; vidx = {v: i for i, v in enumerate(V.index)}
    pairs = [(vidx[v], drugs.index(dd)) for v, dd in zip(hi.variant, hi.drug) if v in vidx and dd in drugs]
    for s in range(5):
        d, r = load(ds, 'A_random', s); te = list(d['te']); pos = {v: k for k, v in enumerate(te)}
        _, _, _, _, Gs, _ = gene_scores(ds, 'A_random', s)
        for name, P in [('proposed', d['Ps']), ('gene_only', Gs)]:
            rk = np.array([int((P[pos[i]] > P[pos[i]][j]).sum()) + 1 for i, j in pairs if i in pos])
            cur.append(dict(dataset=ds, seed=s, model=name, n_pairs=len(rk), top1=(rk <= 1).mean(), top3=(rk <= 3).mean(), top5=(rk <= 5).mean(), median_rank=np.median(rk)))
CUR = pd.DataFrame(cur); CUR.to_csv('curated_check.csv', index=False)
print(CUR.groupby(['dataset', 'model']).mean(numeric_only=True).drop(columns='seed').round(3))

# 4) split statistics for every dataset ------------------------------------------------------------------------------
rows = []
for ds in KW:
    V, Y, drugs, G, flow = DATA[ds]
    for st in ['A_random', 'B_gene_disjoint']:
        for s in range(5):
            tr, va, te = random_split(Y, s) if st == 'A_random' else gene_split(G, Y, s)
            rows.append(dict(dataset=ds, setting=st, seed=s, n_variants=len(Y), n_genes=len(set(G)), n_drugs=len(drugs), n_train=len(tr), n_val=len(va), n_test=len(te),
                             genes_test=len(set(G[te])), test_genes_in_train=len(set(G[te]) & set(G[tr])), test_variants_gene_in_train=float(np.isin(G[te], G[tr]).mean()),
                             drugs_without_test_pos=int((Y[te].sum(0) == 0).sum()), test_positive_pairs=int(Y[te].sum()),
                             clinvar_share=float(V.label_source.str.contains('ClinVar').mean()), cardinality=float(Y.sum(1).mean())))
SP = pd.DataFrame(rows); SP.to_csv('split_stats.csv', index=False)
print(SP.groupby(['dataset', 'setting']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())

# 5) TreeSHAP of the pairwise XGBoost branch (main dataset, known genes, seed 0) --------------------------------------
V, Y, drugs, G, _ = DATA['all']; seed = 0; nd = len(drugs)
chrom = V.seq_region_name.astype(str).values; pos = V.start.astype(float).values
tr, va, te = random_split(Y, seed)
fb = VepFeatures().fit(V.iloc[tr]); (Xt, vnames), (Xv, _), (Xs, _) = (fb.transform(V.iloc[i]) for i in (tr, va, te))
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
Kn[rest] = knn_prior(Xall, G, Y, rest, tr)
K = np.concatenate([K, S, Kn], 2)
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
assert len(fnames) == At.shape[1], (len(fnames), At.shape[1])


def grp(n):
    if n.startswith('drug='): return 'Drug identity'
    if n.startswith('DGIdb'): return 'DGIdb gene-drug knowledge'
    if n.startswith('sim_gene'): return 'Similar-gene prior (DGIdb)'
    if n.startswith('sim_drug'): return 'Similar-drug prior (DGIdb targets)'
    if n.startswith('regional'): return 'Regional (LD) prior'
    if n.startswith('knn'): return 'Feature-space kNN prior'
    if n == 'gene_prior': return 'Gene prior (training labels)'
    if n == 'drug_prevalence': return 'Drug prevalence'
    for k, v in [('gnomad', 'Population frequency'), (':af', 'Population frequency'), ('sift', 'SIFT/PolyPhen'), ('polyphen', 'SIFT/PolyPhen'),
                 ('cadd', 'CADD'), ('blosum', 'BLOSUM62'), ('aa_', 'Amino-acid change'), ('distance', 'Distance to gene')]:
        if k in n.lower(): return v
    return 'Consequence / impact / class'


sub = np.random.default_rng(0).choice(len(As), 30000, replace=False)
cb = m.get_booster().predict(xgb.DMatrix(As[sub]), pred_contribs=True)[:, :-1]
a = pd.Series(np.abs(cb).mean(0), index=fnames).groupby(grp).sum().sort_values(ascending=False)
SH = (a / a.sum()).rename('share_of_mean_abs_SHAP').reset_index().rename(columns={'index': 'feature_group'})
SH.to_csv('treeshap_groups.csv', index=False); print(SH)
pd.Series(np.abs(cb).mean(0), index=fnames).sort_values(ascending=False).head(20).rename('mean_abs_shap').to_csv('treeshap_top20.csv')
print('test micro AUROC of this branch', roc_auc_score(Y[te].ravel(), m.predict_proba(As)[:, 1]))
