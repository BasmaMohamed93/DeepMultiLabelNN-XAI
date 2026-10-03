import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from evalkit import iterative_stratify, tune_thresholds, metrics
from build_dataset import build, VepFeatures


def gene_split(genes, Y, seed, fr=(0.7, 0.1, 0.2)):
    ug, inv = np.unique(genes, return_inverse=True)
    G = np.zeros((len(ug), Y.shape[1]))
    np.maximum.at(G, inv, Y)          # union of a gene's variant labels
    f = iterative_stratify(G, fr, np.random.default_rng(seed))[inv]
    return [np.where(f == k)[0] for k in range(3)]


def random_split(Y, seed, fr=(0.7, 0.1, 0.2)):
    f = iterative_stratify(Y, fr, np.random.default_rng(seed))
    return [np.where(f == k)[0] for k in range(3)]


def within_gene_auroc(Y, P, genes):
    """Mean AUROC computed inside (gene, drug) groups that contain both classes; weight = group size."""
    aucs, w = [], []
    for g in np.unique(genes):
        m = genes == g
        if m.sum() < 2:
            continue
        for j in range(Y.shape[1]):
            y = Y[m, j]
            if 0 < y.sum() < len(y):
                aucs.append(roc_auc_score(y, P[m, j])); w.append(len(y))
    return (float(np.average(aucs, weights=w)) if aucs else np.nan), len(aucs)


def top1(Y, P):
    return float(Y[np.arange(len(Y)), P.argmax(1)].mean())


def evaluate(Ys, Ps, th, genes):
    m = metrics(Ys, Ps, th)
    wg, ng = within_gene_auroc(Ys, Ps, genes)
    return {k: m[k] for k in ['subset_acc', 'hamming_acc', 'micro_f1', 'macro_f1', 'macro_auroc', 'macro_auprc']} | \
           {'top1_acc': top1(Ys, Ps), 'within_gene_auroc': wg, 'n_within_groups': ng}


def run(exclude=(), seed=0):
    V, Y, drugs, flow = build(exclude_drugs=exclude)
    genes = np.array([str(g) if isinstance(g, str) else 'NA' for g in V.gene], dtype=object)
    out = {'flow': flow}
    for setting in ['A_random', 'B_gene_disjoint']:
        tr, va, te = random_split(Y, seed) if setting == 'A_random' else gene_split(genes, Y, seed)
        fb = VepFeatures().fit(V.iloc[tr])
        Xt, Xv, Xs = (fb.transform(V.iloc[i])[0] for i in (tr, va, te))
        res = {'test_genes_seen_in_train': float(np.isin(genes[te], genes[tr]).mean())}
        prior = Y[tr].mean(0)
        tab = {}
        for g in np.unique(genes[tr]):
            tab[g] = Y[tr][genes[tr] == g].mean(0)
        preds = {
            'label_frequency': (np.tile(prior, (len(va), 1)), np.tile(prior, (len(te), 1))),
            'gene_only': (np.array([tab.get(g, prior) for g in genes[va]]), np.array([tab.get(g, prior) for g in genes[te]])),
        }
        m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.6,
                              tree_method='hist', early_stopping_rounds=30, eval_metric='logloss', n_jobs=2, random_state=seed)
        m.fit(Xt, Y[tr], eval_set=[(Xv, Y[va])], verbose=False)
        preds['xgboost_variant_features'] = (m.predict_proba(Xv), m.predict_proba(Xs))
        for k, (pv, ps) in preds.items():
            res[k] = evaluate(Y[te], ps, tune_thresholds(Y[va], pv), genes[te])
        out[setting] = res
    return out


if __name__ == '__main__':
    R = {'all_drugs': run(), 'without_tramadol': run(exclude=('tramadol',))}
    json.dump(R, open('gonogo.json', 'w'), indent=1, default=float)
    for k, r in R.items():
        print('\n######', k, r['flow']['variants_final'], 'variants', r['flow']['drugs_final'], 'drugs', r['flow']['genes_final'], 'genes')
        for s in ['A_random', 'B_gene_disjoint']:
            print('--', s, 'test variants whose gene is in train:', round(r[s]['test_genes_seen_in_train'], 3))
            print(pd.DataFrame({m: r[s][m] for m in ['label_frequency', 'gene_only', 'xgboost_variant_features']}).T.round(3).to_string())
