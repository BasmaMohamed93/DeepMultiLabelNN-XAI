"""DeepMultiLabelNN-XAI model: regional (LD-aware) priors + LambdaMART ranking branch + residual-NN branch, stacked on validation.
Adds a second evaluation protocol used in prior link-prediction work: every positive (variant, drug) pair
against one randomly sampled negative pair (1:1), threshold chosen on the validation partition."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from pairwise import (build, VepFeatures, gene_split, random_split, within_gene_auroc, metrics, pair_features, similarity_features,
                      gene_prior, to_pairs, decode, dec_metrics, choose_decoding, DG, DS_SETS)

WINDOWS = (1e3, 1e4, 1e5, 1e6)


def regional_prior(chrom, pos, Y, q, r):
    """For query rows q: mean label vector of reference rows r on the same chromosome within each window
    (same-position rows excluded), plus log(1+count). Shape (len(q), nd, 2*len(WINDOWS))."""
    nd = Y.shape[1]; out = np.zeros((len(q), nd, 2 * len(WINDOWS)), np.float32)
    for c in np.unique(chrom[q]):
        qi = np.where(chrom[q] == c)[0]; ri = r[chrom[r] == c]
        if len(ri) == 0: continue
        D = np.abs(pos[q[qi]][:, None] - pos[ri][None, :])
        for k, w in enumerate(WINDOWS):
            M = ((D <= w) & (D > 0)).astype(np.float32); n = M.sum(1, keepdims=True)
            out[qi, :, 2 * k] = (M @ Y[ri]) / np.maximum(n, 1)
            out[qi, :, 2 * k + 1] = np.log1p(n)
    return out


def knn_prior(X, genes, Y, q, r, ks=(10, 30)):
    """Label mean of the k nearest reference variants in standardized VEP-feature space (global) and within the same gene."""
    nd = Y.shape[1]; out = np.zeros((len(q), nd, len(ks) + 1), np.float32)
    mu, sd = X[r].mean(0), X[r].std(0) + 1e-6; Z = (X - mu) / sd
    D = ((Z[q] ** 2).sum(1)[:, None] + (Z[r] ** 2).sum(1)[None, :] - 2 * Z[q] @ Z[r].T)
    o = np.argsort(D, 1)
    for j, k in enumerate(ks):
        out[:, :, j] = Y[r][o[:, :k]].mean(1)
    for i, qi in enumerate(q):
        same = r[genes[r] == genes[qi]]
        if len(same):
            d = D[i][genes[r] == genes[qi]]; w = np.exp(-d / (np.median(d) + 1e-6)); out[i, :, -1] = (w @ Y[same]) / w.sum()
    return out


def xgb_bin(At, yt, Av, yv, seed, depth):
    return xgb.XGBClassifier(n_estimators=2000, learning_rate=0.04, max_depth=depth, min_child_weight=2, subsample=0.8,
                             colsample_bytree=0.6, tree_method='hist', max_bin=128, early_stopping_rounds=60,
                             eval_metric='logloss', n_jobs=2, random_state=seed).fit(At, yt, eval_set=[(Av, yv)], verbose=False)


def xgb_rank(At, yt, qt, Av, yv, qv, seed):
    return xgb.XGBRanker(objective='rank:ndcg', lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=10,
                         n_estimators=1500, learning_rate=0.05, max_depth=6, min_child_weight=2, subsample=0.8,
                         colsample_bytree=0.6, tree_method='hist', max_bin=128, early_stopping_rounds=60,
                         eval_metric='ndcg@10', n_jobs=2, random_state=seed).fit(At, yt, qid=qt, eval_set=[(Av, yv)], eval_qid=[qv], verbose=False)


def balanced_eval(Y, P, th, rng, reps=10):
    y, p = Y.ravel(), P.ravel(); pos = np.where(y == 1)[0]; negs = np.where(y == 0)[0]; R = []
    for _ in range(reps):
        i = np.r_[pos, rng.choice(negs, len(pos), replace=False)]; yy, pp = y[i], p[i]; b = pp >= th
        tp, fp, fn = (b & (yy == 1)).sum(), (b & (yy == 0)).sum(), (~b & (yy == 1)).sum()
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        R.append(dict(accuracy=(b == (yy == 1)).mean(), precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-12),
                      auroc=roc_auc_score(yy, pp), auprc=average_precision_score(yy, pp)))
    return {k: float(np.mean([r[k] for r in R])) for k in R[0]}


def balanced_threshold(Y, P, rng, reps=10):
    y, p = Y.ravel(), P.ravel(); pos = np.where(y == 1)[0]; negs = np.where(y == 0)[0]
    grid = np.quantile(p[pos], np.linspace(0.01, 0.6, 60)); sc = np.zeros(len(grid))
    for _ in range(reps):
        i = np.r_[pos, rng.choice(negs, len(pos), replace=False)]; yy, pp = y[i], p[i]
        for g, t in enumerate(grid):
            b = pp >= t; tp = (b & (yy == 1)).sum(); fp = (b & (yy == 0)).sum(); fn = (~b & (yy == 1)).sum()
            sc[g] += 2 * tp / max(2 * tp + fp + fn, 1)
    return float(grid[sc.argmax()])


def run(dataset, setting, seed, use_region=True, use_rank=False, use_knn=True, out_dir='results'):
    tag = f'{dataset}_{setting}_{seed}_region={use_region}_rank={use_rank}' + ('_knn=True' if use_knn else '')
    fn = f'{out_dir}/{tag}.json'
    if os.path.exists(fn): return json.load(open(fn))
    t0 = time.time()
    kw = {'all': {}, 'no_tramadol': {'exclude_drugs': ('tramadol',)}, 'pharmgkb_only': {'sources': ('PharmGKB_CA', 'PharmGKB_VA')}, 'high_evidence': {'drop_levels': ('3',)}}[dataset]
    V, Y, drugs, _ = build(**kw)
    genes = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
    chrom = V.seq_region_name.astype(str).values; pos = V.start.astype(float).values
    tr, va, te = random_split(Y, seed) if setting == 'A_random' else gene_split(genes, Y, seed)
    fb = VepFeatures().fit(V.iloc[tr]); Xt, Xv_, Xs = (fb.transform(V.iloc[i])[0] for i in (tr, va, te))
    nd = len(drugs)
    K = pair_features(genes, drugs, *DG)
    S = np.zeros((len(genes), nd, 2 + (2 * len(WINDOWS) if use_region else 0)), np.float32)
    fold = np.random.default_rng(seed + 7).integers(0, 5, len(tr))
    for k in range(5):   # out-of-fold for training rows
        q, r = tr[fold == k], tr[fold != k]
        S[q, :, :2] = similarity_features(genes[q], drugs, Y, r, *DS_SETS, genes)
        if use_region: S[q, :, 2:] = regional_prior(chrom, pos, Y, q, r)
    rest = np.r_[va, te]; S[rest, :, :2] = similarity_features(genes[rest], drugs, Y, tr, *DS_SETS, genes)
    if use_region: S[rest, :, 2:] = regional_prior(chrom, pos, Y, rest, tr)
    if use_knn:
        Xall = np.zeros((len(genes), Xt.shape[1]), np.float32); Xall[tr], Xall[va], Xall[te] = Xt, Xv_, Xs
        Kn = np.zeros((len(genes), nd, 3), np.float32)
        for k in range(5):
            q, r = tr[fold == k], tr[fold != k]; Kn[q] = knn_prior(Xall, genes, Y, q, r)
        Kn[rest] = knn_prior(Xall, genes, Y, rest, tr)
        S = np.concatenate([S, Kn], 2)
    K = np.concatenate([K, S], 2)
    Gt, Gv, Gs, p0 = gene_prior(genes, Y, tr, va, te, seed)
    At, Av, As = to_pairs(Xt, K[tr], Gt, p0, nd), to_pairs(Xv_, K[va], Gv, p0, nd), to_pairs(Xs, K[te], Gs, p0, nd)
    yt, yv = Y[tr].ravel(), Y[va].ravel()
    best = None
    for depth in (4, 6):
        m = xgb_bin(At, yt, Av, yv, seed, depth)
        if best is None or m.best_score < best[0]: best = (m.best_score, m, depth)
    m1 = best[1]
    Pv = {'xgb': m1.predict_proba(Av)[:, 1]}; Ps = {'xgb': m1.predict_proba(As)[:, 1]}
    from nn_training import nn_fit
    nn, _ = nn_fit(Xt, Y[tr], Xv_, Y[va], seed)
    E = lambda X: np.hstack([nn.embed(X), X])
    Bt, Bv, Bs = to_pairs(E(Xt), K[tr], Gt, p0, nd), to_pairs(E(Xv_), K[va], Gv, p0, nd), to_pairs(E(Xs), K[te], Gs, p0, nd)
    m2 = xgb_bin(Bt, yt, Bv, yv, seed + 1, best[2])
    Pv['hyb'] = m2.predict_proba(Bv)[:, 1]; Ps['hyb'] = m2.predict_proba(Bs)[:, 1]
    Pv['nn'] = np.asarray(nn.predict_proba(Xv_)).ravel(); Ps['nn'] = np.asarray(nn.predict_proba(Xs)).ravel()
    if use_rank:
        qt, qv = np.repeat(np.arange(len(tr)), nd), np.repeat(np.arange(len(va)), nd)
        m3 = xgb_rank(At, yt, qt, Av, yv, qv, seed + 2)
        Pv['rank'] = m3.predict(Av); Ps['rank'] = m3.predict(As)
    # stacking on the validation partition (logistic regression on logits / scores)
    keys = list(Pv)
    f = lambda D: np.column_stack([np.log(np.clip(D[k], 1e-6, 1 - 1e-6) / (1 - np.clip(D[k], 1e-6, 1 - 1e-6))) if k != 'rank' else D[k] for k in keys])
    Fv, Fs = f(Pv), f(Ps)
    st = LogisticRegression(C=1.0, max_iter=2000)
    grp = np.repeat(np.arange(len(va)), nd) % 5
    from sklearn.model_selection import GroupKFold
    Pv_st = cross_val_predict(st, Fv, yv, cv=GroupKFold(5), groups=np.repeat(np.arange(len(va)), nd), method='predict_proba')[:, 1]
    st.fit(Fv, yv); Ps_st = st.predict_proba(Fs)[:, 1]
    PV, PS = Pv_st.reshape(len(va), nd), Ps_st.reshape(len(te), nd)
    Yv, Ys = Y[va], Y[te]
    name, th, ft = choose_decoding(Yv, PV); B = decode(PS, th, ft)
    res = dec_metrics(Ys, B); Mt = metrics(Ys, PS, th)
    res.update(macro_auroc=Mt['macro_auroc'], micro_auroc=Mt['micro_auroc'], macro_auprc=Mt['macro_auprc'])
    res['within_gene_auroc'], _ = within_gene_auroc(Ys, PS, genes[te])
    o = np.argsort(-PS, 1)
    for k in (1, 5, 10):
        top = np.zeros_like(Ys, bool); np.put_along_axis(top, o[:, :k], True, 1)
        res[f'hit@{k}'] = float(((top & (Ys > 0)).sum(1) > 0).mean())
    tb = balanced_threshold(Yv, PV, np.random.default_rng(seed + 100))
    bal = balanced_eval(Ys, PS, tb, np.random.default_rng(seed + 200))
    # gene-only baseline with the identical decoding / balanced protocol
    n2, th2, ft2 = choose_decoding(Yv, Gv); g = dec_metrics(Ys, decode(Gs, th2, ft2)); Mg = metrics(Ys, Gs, th2)
    g.update(macro_auroc=Mg['macro_auroc'], micro_auroc=Mg['micro_auroc'])
    g['balanced'] = balanced_eval(Ys, Gs, balanced_threshold(Yv, Gv, np.random.default_rng(seed + 100)), np.random.default_rng(seed + 200))
    branch_auroc = {k: float(roc_auc_score(Ys.ravel(), Ps[k])) for k in keys}
    out = dict(dataset=dataset, setting=setting, seed=seed, use_region=use_region, use_rank=use_rank, use_knn=use_knn, decoding=name, force_top1=ft,
               stack_coef=dict(zip(keys, st.coef_[0].round(3).tolist())), branch_micro_auroc=branch_auroc,
               test=res, balanced=bal, gene_only=g, balanced_threshold=tb, seconds=round(time.time() - t0, 1))
    np.savez_compressed(fn.replace('.json', '.npz'), Ps=PS.astype(np.float32), Pv=PV.astype(np.float32), te=te, va=va, th=th)
    json.dump(out, open(fn, 'w'), default=float)
    if out_dir != 'results':   # per-branch validation/test scores of the hybrid framework
        np.savez_compressed(fn.replace('.json', '_branches.npz'), **{f'Pv_{k}': Pv[k].reshape(len(va), nd).astype(np.float32) for k in keys},
                            **{f'Ps_{k}': Ps[k].reshape(len(te), nd).astype(np.float32) for k in keys}, te=te, va=va)
    print(f"{tag:50s} {out['seconds']:5.0f}s AUROC={res['macro_auroc']:.3f} microAUROC={res['micro_auroc']:.3f} WG={res['within_gene_auroc']:.3f} "
          f"F1={res['f1']:.3f} hit@1={res['hit@1']:.3f} | BAL acc={bal['accuracy']:.3f} P={bal['precision']:.3f} R={bal['recall']:.3f} F1={bal['f1']:.3f} AUROC={bal['auroc']:.3f}", flush=True)
    return out


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    stage = sys.argv[1]
    if stage == 'hiev':
        for st in ['A_random', 'B_gene_disjoint']:
            run('high_evidence', st, 0)
    if stage == 'knn':
        for st in ['A_random', 'B_gene_disjoint']:
            run('all', st, 0)
    if stage == 'quick':
        for st in ['A_random', 'B_gene_disjoint']:
            run('all', st, 0)
            run('all', st, 0, use_region=False, use_rank=False)
    if stage == 'full':
        for ds in ['all', 'high_evidence', 'pharmgkb_only', 'no_tramadol']:
            for st in ['A_random', 'B_gene_disjoint']:
                for sd in range(5):
                    run(ds, st, sd)
                    if ds in ('all', 'high_evidence'): run(ds, st, sd, use_region=False, use_rank=False, use_knn=False)
    if stage == 'branches':
        os.makedirs('results_components', exist_ok=True)
        for ds in sys.argv[2].split(','):
            for st in ['A_random', 'B_gene_disjoint']:
                for sd in range(5):
                    run(ds, st, sd, out_dir='results_components')
    if stage == 'ablate':
        for st in ['A_random', 'B_gene_disjoint']:
            for sd in range(5):
                run('all', st, sd, use_region=True, use_knn=False)
                run('all', st, sd, use_region=False, use_knn=True)
