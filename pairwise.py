"""Pairwise (variant, drug) model with independent gene-drug pharmacology knowledge from DGIdb.

Leakage controls
- DGIdb rows whose source is PharmGKB or FDA are removed (those overlap the label sources).
- Gene prior (training-label target encoding) is out-of-fold inside the training partition; for unseen genes it
  falls back to the drug's training prevalence.
- Every threshold / decoding rule is chosen on the validation partition; the test partition is scored once.
"""
import sys, os, re, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from build_dataset import build, VepFeatures
from splits import gene_split, random_split, within_gene_auroc
from evalkit import metrics, tune_thresholds

CLASS = {'opioids': ['morphine', 'methadone', 'tramadol', 'codeine', 'fentanyl', 'oxycodone', 'hydrocodone', 'buprenorphine'],
         'platinum compounds': ['cisplatin', 'carboplatin', 'oxaliplatin'],
         'antipsychotics': ['risperidone', 'olanzapine', 'clozapine', 'haloperidol', 'aripiprazole', 'quetiapine', 'paliperidone'],
         'hmg coa reductase inhibitors': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'fluvastatin', 'lovastatin'],
         'tumor necrosis factor alpha (tnf-alpha) inhibitors': ['infliximab', 'adalimumab', 'etanercept', 'certolizumab pegol', 'golimumab'],
         'heroin': ['morphine', 'diamorphine']}
EXCLUDED_SOURCES = ('PharmGKB', 'FDA')


def dgidb_table():
    d = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interactions.csv.xz'), low_memory=False).dropna(subset=['gene_name', 'drug_name'])
    d = d[~d.interaction_source_db_name.isin(EXCLUDED_SOURCES)].copy()
    d['drug'] = d.drug_name.str.lower().str.strip()
    d2 = d.copy(); d2['drug'] = d.drug_claim_name.astype(str).str.lower().str.strip()
    d = pd.concat([d, d2]).drop_duplicates(['gene_name', 'drug', 'interaction_source_db_name'])
    d['inhib'] = d.interaction_type.astype(str).str.contains('inhibitor|antagonist|blocker', na=False)
    g = d.groupby(['gene_name', 'drug']).agg(n_src=('interaction_source_db_name', 'nunique'),
                                            max_score=('interaction_score', 'max'), inhib=('inhib', 'max'))
    gene_deg = d.groupby('gene_name').drug.nunique()
    drug_deg = d.groupby('drug').gene_name.nunique()
    return g, gene_deg, drug_deg


def pair_features(genes, drugs, G, gene_deg, drug_deg):
    """(n_variants, n_drugs, 6) independent knowledge features."""
    F = np.zeros((len(genes), len(drugs), 6), np.float32)
    idx = G.index
    lut = {k: v for k, v in zip(idx, G.values)}
    for j, dr in enumerate(drugs):
        names = CLASS.get(dr, [dr])
        ddeg = sum(drug_deg.get(n, 0) for n in names)
        for i, g in enumerate(genes):
            best = None
            for n in names:
                v = lut.get((g, n))
                if v is not None and (best is None or v[0] > best[0]): best = v
            if best is not None:
                F[i, j, 0] = 1; F[i, j, 1] = best[0]; F[i, j, 2] = np.nan_to_num(best[1]); F[i, j, 3] = float(best[2])
            F[i, j, 4] = np.log1p(gene_deg.get(g, 0)); F[i, j, 5] = np.log1p(ddeg)
    return F


def dgidb_sets():
    d = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interactions.csv.xz'), low_memory=False).dropna(subset=['gene_name', 'drug_name'])
    d = d[~d.interaction_source_db_name.isin(EXCLUDED_SOURCES)]
    d['drug'] = d.drug_name.str.lower().str.strip()
    return d.groupby('gene_name').drug.apply(set).to_dict(), d.groupby('drug').gene_name.apply(set).to_dict()


def jacc(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def similarity_features(qgenes, drugs, Y, tr, gene2drugs, drug2genes, allgenes, topk=10):
    """(n, nd, 2): [gene-similarity prior, drug-similarity prior], built from TRAINING labels only.
    gene-sim prior: labels of the top-k training genes most similar (DGIdb drug-profile Jaccard) to the variant's gene,
                    excluding the gene itself (so it is informative for unseen genes and not self-leaking for seen ones).
    drug-sim prior: the variant gene's training label rates for OTHER drugs, weighted by drug-drug target similarity."""
    gtr = allgenes[tr]; Yt = Y[tr]
    tg = {g: Yt[gtr == g].mean(0) for g in np.unique(gtr)}
    tgs = list(tg); TG = np.array([tg[g] for g in tgs])
    cache = {}
    def gprior(g):
        if g in cache: return cache[g]
        s = gene2drugs.get(g, set())
        sims = np.array([jacc(s, gene2drugs.get(h, set())) if h != g else 0.0 for h in tgs])
        if sims.max() <= 0: out = np.full(Y.shape[1], np.nan)
        else:
            idx = np.argsort(-sims)[:topk]; w = sims[idx]; out = (w[:, None] * TG[idx]).sum(0) / w.sum()
        cache[g] = out; return out
    nd = len(drugs)
    tsets = [set().union(*[drug2genes.get(n, set()) for n in CLASS.get(dr, [dr])]) for dr in drugs]
    DS = np.array([[jacc(tsets[a], tsets[b]) if a != b else 0.0 for b in range(nd)] for a in range(nd)])
    DSn = DS / np.maximum(DS.sum(1, keepdims=True), 1e-9)
    F = np.zeros((len(qgenes), nd, 2), np.float32)
    for i, g in enumerate(qgenes):
        F[i, :, 0] = gprior(g)
        own = tg.get(g)
        F[i, :, 1] = DSn @ own if own is not None else np.nan
    return F


def gene_prior(genes, Y, tr, va, te, seed):
    gtr = genes[tr]; Yt = Y[tr]; p0 = Yt.mean(0); fold = np.random.default_rng(seed).integers(0, 5, len(tr))
    Gt = np.zeros_like(Yt)
    for k in range(5):
        mi, mo = fold != k, fold == k
        tab = {g: Yt[mi][gtr[mi] == g].mean(0) for g in np.unique(gtr[mi])}
        Gt[mo] = [tab.get(g, p0) for g in gtr[mo]]
    tab = {g: Yt[gtr == g].mean(0) for g in np.unique(gtr)}
    return Gt, np.array([tab.get(g, p0) for g in genes[va]]), np.array([tab.get(g, p0) for g in genes[te]]), p0


def to_pairs(Xv, K, GP, p0, nd):
    n = len(Xv)
    D = np.tile(np.eye(nd, dtype=np.float32), (n, 1))
    return np.hstack([np.repeat(Xv, nd, 0), D, K.reshape(n * nd, -1), GP.reshape(-1, 1), np.tile(p0, n).reshape(-1, 1)]).astype(np.float32)


def decode(P, th, force_top1=True):
    B = P >= th
    if force_top1:
        B[np.arange(len(P)), P.argmax(1)] = True
    return B


def dec_metrics(Y, B):
    Yb = Y.astype(bool); tp, fp, fn = (B & Yb).sum(), (B & ~Yb).sum(), (~B & Yb).sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(subset_acc=float((B == Yb).all(1).mean()), hamming_acc=float((B == Yb).mean()), precision=float(p), recall=float(r),
                f1=float(2 * p * r / max(p + r, 1e-12)),
                sample_f1=float(np.mean([2 * (b & y).sum() / max(b.sum() + y.sum(), 1) for b, y in zip(B, Yb)])))


def choose_decoding(Yv, Pv):
    """Per-drug F1 thresholds vs a global threshold, each with/without forced top-1; pick best micro-F1 on VALIDATION."""
    best = None
    th_pd = tune_thresholds(Yv, Pv)
    cands = [('per_drug', th_pd)] + [(f'global_{t:.2f}', np.full(Yv.shape[1], t)) for t in np.linspace(0.05, 0.9, 18)]
    for name, th in cands:
        for ft in (False, True):
            m = dec_metrics(Yv, decode(Pv, th, ft))
            if best is None or m['f1'] > best[0]:
                best = (m['f1'], name, th, ft)
    return best[1], best[2], best[3]


def run(dataset, setting, seed, use_dgidb=True, use_prior=True, use_emb=False, use_sim=False, ensemble=False, use_var=True):
    tag = f'{dataset}|{setting}|{seed}|dgidb={use_dgidb}|prior={use_prior}' + ('' if use_var else '|novar=True') + ('|emb=True' if use_emb else '') + ('|sim=True' if use_sim else '') + ('|ens=True' if ensemble else '')
    fn = 'results_pair/' + tag.replace('|', '_') + '.json'
    if os.path.exists(fn): return json.load(open(fn))
    t0 = time.time()
    kw = {'all': {}, 'no_tramadol': {'exclude_drugs': ('tramadol',)}, 'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}}[dataset]
    V, Y, drugs, flow = build(**kw)
    genes = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
    tr, va, te = random_split(Y, seed) if setting == 'A_random' else gene_split(genes, Y, seed)
    fb = VepFeatures().fit(V.iloc[tr]); Xt, Xv_, Xs = (fb.transform(V.iloc[i])[0] for i in (tr, va, te))
    nd = len(drugs)
    if not use_var:
        Xt, Xv_, Xs = (np.zeros((len(i), 0), np.float32) for i in (tr, va, te))
    K = pair_features(genes, drugs, *DG) if use_dgidb else np.zeros((len(genes), nd, 0), np.float32)
    if use_sim:
        # training rows: out-of-fold (5 folds) so a training variant never sees its own label through the similarity priors
        S = np.zeros((len(genes), nd, 2), np.float32)
        fold = np.random.default_rng(seed + 7).integers(0, 5, len(tr))
        for k in range(5):
            S[tr[fold == k]] = similarity_features(genes[tr[fold == k]], drugs, Y, tr[fold != k], *DS_SETS, genes)
        rest = np.r_[va, te]; S[rest] = similarity_features(genes[rest], drugs, Y, tr, *DS_SETS, genes)
        K = np.concatenate([K, S], 2)
    Gt, Gv, Gs, p0 = gene_prior(genes, Y, tr, va, te, seed)
    if not use_prior:
        Gt, Gv, Gs = (np.tile(p0, (len(i), 1)) for i in (tr, va, te))
    if use_emb:   # hybrid: Residual-NN variant embedding (trained multi-label on train, early-stopped on val) appended
        from nn_training import nn_fit
        nn, _ = nn_fit(Xt, Y[tr], Xv_, Y[va], seed)
        Xt, Xv_, Xs = np.hstack([nn.embed(Xt), Xt]), np.hstack([nn.embed(Xv_), Xv_]), np.hstack([nn.embed(Xs), Xs])
    At, Av, As = to_pairs(Xt, K[tr], Gt, p0, nd), to_pairs(Xv_, K[va], Gv, p0, nd), to_pairs(Xs, K[te], Gs, p0, nd)
    yt, yv = Y[tr].ravel(), Y[va].ravel()
    best = None
    for depth in (4, 6):
        m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=depth, min_child_weight=2, subsample=0.8,
                              colsample_bytree=0.6, tree_method='hist', max_bin=128, early_stopping_rounds=50,
                              eval_metric='logloss', n_jobs=2, random_state=seed)
        m.fit(At, yt, eval_set=[(Av, yv)], verbose=False)
        if best is None or m.best_score < best[0]: best = (m.best_score, m, depth)
    m = best[1]
    Pv = m.predict_proba(Av)[:, 1].reshape(len(va), nd); Ps = m.predict_proba(As)[:, 1].reshape(len(te), nd)
    blend = None
    if ensemble:
        from nn_training import nn_fit
        nn, _ = nn_fit(Xt, Y[tr], Xv_, Y[va], seed)
        E = lambda X, Xo: np.hstack([nn.embed(Xo), Xo])
        Bt = to_pairs(E(None, Xt), K[tr], Gt, p0, nd); Bv = to_pairs(E(None, Xv_), K[va], Gv, p0, nd); Bs = to_pairs(E(None, Xs), K[te], Gs, p0, nd)
        m2 = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=best[2], min_child_weight=2, subsample=0.8,
                               colsample_bytree=0.6, tree_method='hist', max_bin=128, early_stopping_rounds=50,
                               eval_metric='logloss', n_jobs=2, random_state=seed + 1).fit(Bt, yt, eval_set=[(Bv, yv)], verbose=False)
        Pv2 = m2.predict_proba(Bv)[:, 1].reshape(len(va), nd); Ps2 = m2.predict_proba(Bs)[:, 1].reshape(len(te), nd)
        ll = lambda P: -np.mean(Y[va] * np.log(P + 1e-7) + (1 - Y[va]) * np.log(1 - P + 1e-7))
        blend = min(np.linspace(0, 1, 11), key=lambda w: ll(w * Pv + (1 - w) * Pv2))
        Pv, Ps = blend * Pv + (1 - blend) * Pv2, blend * Ps + (1 - blend) * Ps2
    name, th, ft = choose_decoding(Y[va], Pv)
    Ys = Y[te]; B = decode(Ps, th, ft)
    res = dec_metrics(Ys, B)
    M = metrics(Ys, Ps, th)
    res.update(macro_auroc=M['macro_auroc'], micro_auroc=M['micro_auroc'], macro_auprc=M['macro_auprc'])
    res['within_gene_auroc'], _ = within_gene_auroc(Ys, Ps, genes[te])
    o = np.argsort(-Ps, 1)
    for k in (1, 5, 10):
        top = np.zeros_like(Ys, bool); np.put_along_axis(top, o[:, :k], True, 1)
        res[f'hit@{k}'] = float(((top & (Ys > 0)).sum(1) > 0).mean())
    out = dict(dataset=dataset, setting=setting, seed=seed, use_dgidb=use_dgidb, use_prior=use_prior, use_emb=use_emb, use_sim=use_sim, ensemble=ensemble, blend_weight_xgb=blend, decoding=name, force_top1=ft,
               depth=best[2], n_trees=int(m.best_iteration) + 1, test=res, seconds=round(time.time() - t0, 1))
    np.savez_compressed(fn.replace('.json', '.npz'), Ps=Ps.astype(np.float32), te=te, th=th)
    json.dump(out, open(fn, 'w'), default=float)
    print(f"{tag:55s} {out['seconds']:6.0f}s AUROC={res['macro_auroc']:.3f} WG={res['within_gene_auroc']:.3f} "
          f"P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f} acc={res['subset_acc']:.3f} hit@1={res['hit@1']:.3f} [{name},{ft}]", flush=True)
    return out


os.makedirs('results_pair', exist_ok=True)
DG = dgidb_table()
DS_SETS = dgidb_sets()
if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'quick'
    if stage == 'v3quick':
        for ds in ['all', 'pharmgkb_only']:
            for st in ['A_random', 'B_gene_disjoint']:
                run(ds, st, 0, True, True, use_sim=True)
                run(ds, st, 0, True, True, use_sim=True, ensemble=True)
    if stage == 'dgidb_only':
        for ds in ['all', 'pharmgkb_only']:
            for st in ['A_random', 'B_gene_disjoint']:
                for sd in range(5):
                    run(ds, st, sd, True, False, use_var=False)
        sys.exit()
    if stage == 'v3full':
        for ds in ['all', 'pharmgkb_only', 'no_tramadol']:
            for st in ['A_random', 'B_gene_disjoint']:
                for sd in range(5):
                    run(ds, st, sd, True, True, use_sim=True)
                    run(ds, st, sd, True, True, use_sim=True, ensemble=True)
        sys.exit()
    if stage == 'quick':
        for st in ['A_random', 'B_gene_disjoint']:
            for cfg in [(False, True), (True, True)]:
                run('all', st, 0, *cfg)
    else:
        for ds in ['all', 'pharmgkb_only', 'no_tramadol']:
            for st in ['A_random', 'B_gene_disjoint']:
                for s in range(5):
                    for cfg in [(True, True), (False, True), (True, False)]:
                        run(ds, st, s, *cfg)
    if stage in ('full', 'hybrid'):
        for ds in ['all', 'pharmgkb_only', 'no_tramadol']:
            for st in ['A_random', 'B_gene_disjoint']:
                for s in range(5):
                    run(ds, st, s, True, True, use_emb=True)
