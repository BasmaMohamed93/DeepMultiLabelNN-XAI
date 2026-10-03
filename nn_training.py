"""Variant-level revision experiments. Cached per job in results_v2/.
Settings: A = random variant split (genes may repeat; within-gene AUROC = setting C), B = gene-disjoint split.
Tuning (small grids), early stopping and per-drug thresholds use VALIDATION only; test scored once per seed."""
import sys, os, json, time, hashlib, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from evalkit import tune_thresholds, metrics, per_label
from resnet import ResMLP
from build_dataset import build, VepFeatures
from splits import gene_split, random_split, within_gene_auroc, top1

os.makedirs('results_v2', exist_ok=True); os.makedirs('preds_v2', exist_ok=True)
DATA = {}


def data(name):
    if name not in DATA:
        V, Y, drugs, flow = build(exclude_drugs=('tramadol',) if name == 'no_tramadol' else (),
                                  sources=('PharmGKB_CA', 'PharmGKB_VA') if name == 'pharmgkb_only' else None)
        genes = np.array([g if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
        DATA[name] = (V, Y, drugs, flow, genes)
    return DATA[name]


def xgb_fit(Xt, Yt, Xv, Yv, seed):
    best = None
    for depth, mcw in [(3, 1), (5, 3)]:
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=depth, min_child_weight=mcw,
                              subsample=0.8, colsample_bytree=0.6, tree_method='hist', early_stopping_rounds=30,
                              eval_metric='logloss', n_jobs=2, random_state=seed)
        m.fit(Xt, Yt, eval_set=[(Xv, Yv)], verbose=False)
        if best is None or m.best_score < best[0]:
            best = (m.best_score, m, {'max_depth': depth, 'min_child_weight': mcw, 'n_trees': int(m.best_iteration) + 1})
    return best[1], best[2]


def nn_fit(Xt, Yt, Xv, Yv, seed, **kw):
    grid = [{}] if kw.pop('no_grid', False) else [{'dropout': 0.2}, {'dropout': 0.4}]
    best = None
    for g in grid:
        p = {'epochs': 150, 'patience': 15, 'batch': 128, **kw, **g}
        m = ResMLP(Xt.shape[1], Yt.shape[1], seed=seed, **p).fit(Xt, Yt, Xv, Yv)
        if best is None or min(m.history) < best[0]:
            best = (min(m.history), m, {**p, 'best_epoch': m.best_epoch})
    return best[1], best[2]


def lr_prob(Xt, y, Xs_list, C=0.1):
    if y.min() == y.max():
        return [np.full(len(X), y.mean()) for X in Xs_list]
    m = LogisticRegression(C=C, max_iter=500, class_weight='balanced').fit(Xt, y)
    return [m.predict_proba(X)[:, 1] for X in Xs_list]


def run(dataset, setting, seed, model, groups=('functional', 'consequence', 'frequency'), tag='', **kw):
    key = f'{dataset}|{setting}|{seed}|{model}|{"+".join(groups)}|{tag}'
    fn = 'results_v2/' + hashlib.md5(key.encode()).hexdigest()[:12] + '.json'
    if os.path.exists(fn):
        return json.load(open(fn))
    t0 = time.time()
    V, Y, drugs, flow, genes = data(dataset)
    tr, va, te = random_split(Y, seed) if setting == 'A_random' else gene_split(genes, Y, seed)
    fb = VepFeatures(groups).fit(V.iloc[tr])
    (Xt, names), (Xv, _), (Xs, _) = (fb.transform(V.iloc[i]) for i in (tr, va, te))
    Yt, Yv, Ys = Y[tr], Y[va], Y[te]
    hp = {}
    if model.endswith('_geneprior'):
        # gene prior = target encoding from TRAINING labels only; out-of-fold (5 folds) inside train to avoid self-leakage
        gtr = genes[tr]; prior0 = Yt.mean(0); fold = np.random.default_rng(seed).integers(0, 5, len(tr))
        Gt = np.zeros_like(Yt)
        for k in range(5):
            m_in, m_out = fold != k, fold == k
            tab = {g: Yt[m_in][gtr[m_in] == g].mean(0) for g in np.unique(gtr[m_in])}
            Gt[m_out] = [tab.get(g, prior0) for g in gtr[m_out]]
        tab = {g: Yt[gtr == g].mean(0) for g in np.unique(gtr)}
        Gv = np.array([tab.get(g, prior0) for g in genes[va]]); Gs = np.array([tab.get(g, prior0) for g in genes[te]])
        Xt, Xv, Xs = np.hstack([Xt, Gt]), np.hstack([Xv, Gv]), np.hstack([Xs, Gs])
        model = model[:-len('_geneprior')]
    prior = Yt.mean(0)
    if model == 'label_frequency':
        Pv, Ps = np.tile(prior, (len(va), 1)), np.tile(prior, (len(te), 1))
    elif model == 'gene_only':
        tab = {g: Yt[genes[tr] == g].mean(0) for g in np.unique(genes[tr])}
        Pv = np.array([tab.get(g, prior) for g in genes[va]]); Ps = np.array([tab.get(g, prior) for g in genes[te]])
    elif model == 'logreg_br':
        Pv, Ps = np.zeros(Yv.shape), np.zeros(Ys.shape)
        for j in range(Y.shape[1]):
            Pv[:, j], Ps[:, j] = lr_prob(Xt, Yt[:, j], [Xv, Xs])
        hp = {'C': 0.1, 'class_weight': 'balanced'}
    elif model == 'classifier_chain':
        order = np.random.default_rng(seed).permutation(Y.shape[1])
        At, Av, As = Xt, Xv, Xs
        Pv, Ps = np.zeros(Yv.shape), np.zeros(Ys.shape)
        for j in order:
            pt, pv, ps = lr_prob(At, Yt[:, j], [At, Av, As])
            Pv[:, j], Ps[:, j] = pv, ps
            At, Av, As = np.c_[At, Yt[:, j]], np.c_[Av, pv], np.c_[As, ps]   # teacher forcing on train
        hp = {'base': 'LogReg C=0.1 balanced', 'order': 'random'}
    elif model == 'xgboost':
        m, hp = xgb_fit(Xt, Yt, Xv, Yv, seed); Pv, Ps = m.predict_proba(Xv), m.predict_proba(Xs)
    elif model in ('resnn', 'mlp_nonresidual'):
        if model == 'mlp_nonresidual':
            kw = {'residual': False, 'act': 'relu', **kw}
        m, hp = nn_fit(Xt, Yt, Xv, Yv, seed, **kw); Pv, Ps = m.predict_proba(Xv), m.predict_proba(Xs)
    elif model == 'hybrid_oof':
        # out-of-fold embeddings for training rows (NN never embeds rows it was trained on), full-train NN for val/test
        nn, hpn = nn_fit(Xt, Yt, Xv, Yv, seed, **kw)
        E = nn.emb; Et = np.zeros((len(Xt), E), np.float32)
        fold = np.random.default_rng(seed + 11).integers(0, 5, len(Xt))
        for k in range(5):
            mk = fold != k
            nk, _ = nn_fit(Xt[mk], Yt[mk], Xv, Yv, seed + k, no_grid=True, dropout=hpn['dropout'])
            Et[~mk] = nk.embed(Xt[~mk])
        Et, Ev, Es = np.hstack([Et, Xt]), np.hstack([nn.embed(Xv), Xv]), np.hstack([nn.embed(Xs), Xs])
        m, hpx = xgb_fit(Et, Yt, Ev, Yv, seed); Pv, Ps = m.predict_proba(Ev), m.predict_proba(Es)
        hp = {'nn': hpn, 'xgb': hpx, 'embedding': 'out-of-fold (5)'}
    elif model.startswith('hybrid'):
        nn, hpn = nn_fit(Xt, Yt, Xv, Yv, seed, **kw)
        Et, Ev, Es = nn.embed(Xt), nn.embed(Xv), nn.embed(Xs)
        if model == 'hybrid_emb_raw':
            Et, Ev, Es = np.hstack([Et, Xt]), np.hstack([Ev, Xv]), np.hstack([Es, Xs])
        m, hpx = xgb_fit(Et, Yt, Ev, Yv, seed); Pv, Ps = m.predict_proba(Ev), m.predict_proba(Es)
        hp = {'nn': hpn, 'xgb': hpx}
    model = key.split('|')[3]
    th = tune_thresholds(Yv, Pv)
    M = metrics(Ys, Ps, th)
    B = Ps >= th; Yb = Ys.astype(bool)
    tp, fp, fn_ = (B & Yb).sum(0), (B & ~Yb).sum(0), (~B & Yb).sum(0)
    has = Yb.sum(0) > 0
    M.update(micro_precision=float(tp.sum() / max(tp.sum() + fp.sum(), 1)), micro_recall=float(tp.sum() / max(tp.sum() + fn_.sum(), 1)),
             macro_precision=float((tp / np.maximum(tp + fp, 1))[has].mean()), macro_recall=float((tp / np.maximum(tp + fn_, 1))[has].mean()),
             top1_acc=top1(Ys, Ps))
    M['within_gene_auroc'], M['n_within_groups'] = within_gene_auroc(Ys, Ps, genes[te])
    res = {'dataset': dataset, 'setting': setting, 'seed': seed, 'model': model, 'groups': list(groups), 'tag': tag,
           'n_features': Xt.shape[1], 'hp': hp, 'test': M, 'per_label': per_label(Ys, Ps, th, drugs),
           'test_variants_gene_in_train': float(np.isin(genes[te], genes[tr]).mean()),
           'n_train': len(tr), 'n_val': len(va), 'n_test': len(te), 'seconds': round(time.time() - t0, 1)}
    np.savez_compressed('preds_v2/' + os.path.basename(fn)[:-5] + '.npz', Ps=Ps.astype(np.float32), te=te, th=th)
    json.dump(res, open(fn, 'w'), default=float)
    print(f'{key:75s} {res["seconds"]:6.1f}s macroAUROC={M["macro_auroc"]:.3f} withinGene={M["within_gene_auroc"]:.3f} '
          f'microF1={M["micro_f1"]:.3f} subset={M["subset_acc"]:.3f}', flush=True)
    return res


MODELS = ['label_frequency', 'gene_only', 'logreg_br', 'classifier_chain', 'xgboost', 'mlp_nonresidual', 'resnn',
          'hybrid_emb', 'hybrid_emb_raw', 'xgboost_geneprior', 'resnn_geneprior', 'hybrid_emb_raw_geneprior']

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'main':
        for ds in ['all', 'no_tramadol']:
            for st in ['A_random', 'B_gene_disjoint']:
                for s in range(5):
                    for m in MODELS:
                        run(ds, st, s, m)
    elif stage == 'pgkb':
        for st in ['A_random', 'B_gene_disjoint']:
            for s in range(5):
                for m in MODELS:
                    run('pharmgkb_only', st, s, m)
    elif stage == 'ablation':
        for st in ['A_random', 'B_gene_disjoint']:          # PharmGKB-only (no source confound) first
            for s in range(5):
                for m in ['label_frequency', 'gene_only', 'logreg_br', 'xgboost', 'resnn', 'hybrid_emb_raw',
                          'xgboost_geneprior', 'hybrid_emb_raw_geneprior']:
                    run('pharmgkb_only', st, s, m)
        for st in ['A_random', 'B_gene_disjoint']:
            for s in range(3):
                for tag, kw in [('no_skip', {'residual': False}), ('no_bn', {'bn': False}), ('relu', {'act': 'relu'}),
                                ('no_smoothing', {'smoothing': 0.0}), ('no_class_weight', {'class_weight': False}),
                                ('1_block', {'n_blocks': 1}), ('3_blocks', {'n_blocks': 3})]:
                    run('all', st, s, 'resnn', tag=tag, **kw)
                for g in [('functional',), ('consequence',), ('frequency',), ('functional', 'consequence')]:
                    run('all', st, s, 'xgboost', groups=g)
                    run('all', st, s, 'hybrid_emb_raw', groups=g)
