"""Splitting, threshold selection and multi-label metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def iterative_stratify(Y, fractions, rng):
    """Sechidis et al. (2011) iterative stratification. Returns fold id per row."""
    Y = Y.astype(bool)
    n, L = Y.shape
    fractions = np.asarray(fractions, float)
    desired = np.outer(fractions, Y.sum(0)).astype(float)  # folds x labels
    cap = fractions * n
    fold = -np.ones(n, int)
    remaining = np.ones(n, bool)
    while remaining.any():
        counts = Y[remaining].sum(0)
        if counts.max() == 0:
            rows = np.where(remaining)[0]
            for r in rng.permutation(rows):
                k = np.argmax(cap)
                fold[r] = k
                cap[k] -= 1
            break
        counts = np.where(counts == 0, np.inf, counts)
        l = int(np.argmin(counts))
        rows = rng.permutation(np.where(remaining & Y[:, l])[0])
        for r in rows:
            d = desired[:, l]
            best = np.flatnonzero(d == d.max())
            if len(best) > 1:
                best = best[cap[best] == cap[best].max()]
            k = rng.choice(best)
            fold[r] = k
            remaining[r] = False
            desired[k] -= Y[r]
            cap[k] -= 1
    return fold


def random_split(Y, seed, fr=(0.7, 0.1, 0.2)):
    f = iterative_stratify(Y, fr, np.random.default_rng(seed))
    return [np.where(f == k)[0] for k in range(3)]


def gene_split(genes, Y, seed, fr=(0.7, 0.1, 0.2)):
    """Gene-disjoint split: iterative stratification applied to genes (label vector is per gene)."""
    genes = np.asarray(genes)
    ug, inv = np.unique(genes, return_inverse=True)
    G = np.zeros((len(ug), Y.shape[1]))
    G[inv] = Y  # identical labels within a gene
    f = iterative_stratify(G, fr, np.random.default_rng(seed))
    rowf = f[inv]
    return [np.where(rowf == k)[0] for k in range(3)]


def tune_thresholds(Yv, Pv, grid=np.linspace(0.05, 0.95, 19)):
    """Per-label threshold maximising F1 on VALIDATION data; labels without val positives get 0.5."""
    th = np.full(Yv.shape[1], 0.5)
    for j in range(Yv.shape[1]):
        if Yv[:, j].sum() == 0:
            continue
        best = -1
        for t in grid:
            p = Pv[:, j] >= t
            tp = (p & (Yv[:, j] == 1)).sum()
            f1 = 2 * tp / (p.sum() + Yv[:, j].sum() + 1e-9)
            if f1 > best:
                best, th[j] = f1, t
    return th


def metrics(Y, P, th):
    Y = Y.astype(bool)
    B = P >= th
    tp, fp, fn = (B & Y).sum(0), (B & ~Y).sum(0), (~B & Y).sum(0)
    has = Y.sum(0) > 0
    both = has & (Y.sum(0) < len(Y))
    f1_l = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    inter, union = (B & Y).sum(1), (B | Y).sum(1)
    out = {
        "subset_acc": float((B == Y).all(1).mean()),
        "hamming_acc": float((B == Y).mean()),
        "sample_acc_jaccard": float(np.where(union > 0, inter / np.maximum(union, 1), 1).mean()),
        "micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
        "macro_f1": float(f1_l[has].mean()),
        "micro_auroc": float(roc_auc_score(Y[:, both].ravel(), P[:, both].ravel())),
        "macro_auroc": float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in np.where(both)[0]])),
        "macro_auprc": float(np.mean([average_precision_score(Y[:, j], P[:, j]) for j in np.where(both)[0]])),
        "micro_auprc": float(average_precision_score(Y[:, both].ravel(), P[:, both].ravel())),
        "n_labels_evaluable": int(both.sum()),
    }
    return out


def per_label(Y, P, th, names):
    rows = []
    for j, nme in enumerate(names):
        y, p = Y[:, j].astype(bool), P[:, j]
        b = p >= th[j]
        tp, fp, fn = (b & y).sum(), (b & ~y).sum(), (~b & y).sum()
        ok = 0 < y.sum() < len(y)
        rows.append({"drug": nme, "prevalence": y.mean(), "n_pos": int(y.sum()),
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                     "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                     "auroc": roc_auc_score(y, p) if ok else np.nan,
                     "auprc": average_precision_score(y, p) if ok else np.nan, "threshold": th[j]})
    return rows
