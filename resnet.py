"""NumPy implementation of the residual multi-label network (DeepMultiLabelNN).

Architecture (exactly what produces every NN number in the revision):
  stem   : Linear(d,256) -> BatchNorm -> GELU
  blocks : n_blocks x [ Linear(256,256) -> BN -> GELU -> Dropout -> Linear(256,256) ] + identity skip
  embed  : Linear(256,64) -> BN -> GELU      (64-d embedding, fed to XGBoost in the hybrid)
  head   : Linear(64, L) -> sigmoid
Loss     : class-weighted BCE with label smoothing
Optimiser: AdamW + cosine annealing, early stopping on validation loss.
Switches allow the architecture ablations (no skip, no BN, ReLU, no smoothing, no weights).
"""
import numpy as np


def gelu(x):
    t = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1 + t), t


def gelu_grad(x, t):
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * 0.7978845608 * (1 + 3 * 0.044715 * x ** 2)


class Linear:
    def __init__(self, i, o, rng):
        self.p = {"W": rng.normal(0, np.sqrt(2.0 / i), (i, o)).astype(np.float32),
                  "b": np.zeros(o, np.float32)}
        self.decay = {"W": True, "b": False}

    def fwd(self, x, train):
        self.x = x
        return x @ self.p["W"] + self.p["b"]

    def bwd(self, g):
        self.g = {"W": self.x.T @ g, "b": g.sum(0)}
        return g @ self.p["W"].T


class BN:
    def __init__(self, n, on=True):
        self.on = on
        self.p = {"gamma": np.ones(n, np.float32), "beta": np.zeros(n, np.float32)} if on else {}
        self.decay = {"gamma": False, "beta": False}
        self.rm, self.rv = np.zeros(n, np.float32), np.ones(n, np.float32)

    def fwd(self, x, train):
        if not self.on:
            return x
        if train:
            mu, var = x.mean(0), x.var(0)
            self.rm = 0.9 * self.rm + 0.1 * mu
            self.rv = 0.9 * self.rv + 0.1 * var
        else:
            mu, var = self.rm, self.rv
        self.inv = 1.0 / np.sqrt(var + 1e-5)
        self.xh = (x - mu) * self.inv
        return self.p["gamma"] * self.xh + self.p["beta"]

    def bwd(self, g):
        if not self.on:
            return g
        self.g = {"gamma": (g * self.xh).sum(0), "beta": g.sum(0)}
        dxh = g * self.p["gamma"]
        n = g.shape[0]
        return (self.inv / n) * (n * dxh - dxh.sum(0) - self.xh * (dxh * self.xh).sum(0))


class Act:
    def __init__(self, kind="gelu"):
        self.kind, self.p = kind, {}

    def fwd(self, x, train):
        self.x = x
        if self.kind == "gelu":
            y, self.t = gelu(x)
            return y
        return np.maximum(x, 0)

    def bwd(self, g):
        if self.kind == "gelu":
            return g * gelu_grad(self.x, self.t)
        return g * (self.x > 0)


class Dropout:
    def __init__(self, rate, rng):
        self.rate, self.rng, self.p = rate, rng, {}

    def fwd(self, x, train):
        if not train or self.rate == 0:
            self.m = None
            return x
        self.m = (self.rng.random(x.shape) > self.rate).astype(np.float32) / (1 - self.rate)
        return x * self.m

    def bwd(self, g):
        return g if self.m is None else g * self.m


class Seq:
    def __init__(self, layers, skip=False):
        self.layers, self.skip, self.p = layers, skip, {}

    def fwd(self, x, train):
        h = x
        for l in self.layers:
            h = l.fwd(h, train)
        return x + h if self.skip else h

    def bwd(self, g):
        h = g
        for l in reversed(self.layers):
            h = l.bwd(h)
        return g + h if self.skip else h

    def params(self):
        for l in self.layers:
            if isinstance(l, Seq):
                yield from l.params()
            elif l.p:
                yield l


class ResMLP:
    def __init__(self, d, L, hidden=256, emb=64, n_blocks=2, dropout=0.3, residual=True, bn=True,
                 act="gelu", lr=1e-3, wd=1e-4, epochs=120, batch=256, patience=12,
                 smoothing=0.05, class_weight=True, seed=0, verbose=False, cosine=True, early_stop=True):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        rng = np.random.default_rng(seed)
        self.rng = rng
        blocks = [Seq([Linear(hidden, hidden, rng), BN(hidden, bn), Act(act), Dropout(dropout, rng),
                       Linear(hidden, hidden, rng)], skip=residual) for _ in range(n_blocks)]
        self.body = Seq([Linear(d, hidden, rng), BN(hidden, bn), Act(act), *blocks,
                         Linear(hidden, emb, rng), BN(emb, bn), Act(act)])
        self.head = Seq([Dropout(dropout / 2, rng), Linear(emb, L, rng)])
        self.mods = list(self.body.params()) + list(self.head.params())

    def _loss_grad(self, z, y, pw):
        ys = y * (1 - self.smoothing) + 0.5 * self.smoothing
        s = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        w = pw * ys + (1 - ys)
        loss = -(pw * ys * np.log(s + 1e-7) + (1 - ys) * np.log(1 - s + 1e-7)).mean()
        return loss, (w * s - pw * ys) / z.size

    def fit(self, X, Y, Xv, Yv):
        pos = Y.mean(0).clip(1e-3, None)
        self.pw = np.clip((1 - pos) / pos, 1, 10).astype(np.float32) if self.class_weight else np.ones(Y.shape[1], np.float32)
        opt = {}
        for m in self.mods:
            for k, v in m.p.items():
                opt[(id(m), k)] = [np.zeros_like(v), np.zeros_like(v)]
        best, bad, t, self.history = np.inf, 0, 0, []
        n = len(X)
        for ep in range(self.epochs):
            lr = 0.5 * self.lr * (1 + np.cos(np.pi * ep / self.epochs)) if self.cosine else self.lr  # cosine annealing
            idx = self.rng.permutation(n)
            for i in range(0, n - 1, self.batch):
                b = idx[i:i + self.batch]
                if len(b) < 2:
                    continue
                z = self.head.fwd(self.body.fwd(X[b], True), True)
                _, g = self._loss_grad(z, Y[b], self.pw)
                self.body.bwd(self.head.bwd(g))
                t += 1
                for m in self.mods:
                    for k in m.p:
                        gk = m.g[k]
                        mv = opt[(id(m), k)]
                        mv[0] = 0.9 * mv[0] + 0.1 * gk
                        mv[1] = 0.999 * mv[1] + 0.001 * gk * gk
                        mh, vh = mv[0] / (1 - 0.9 ** t), mv[1] / (1 - 0.999 ** t)
                        if m.decay.get(k, False):
                            m.p[k] -= lr * self.wd * m.p[k]  # decoupled weight decay (AdamW)
                        m.p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
            vl, _ = self._loss_grad(self.logits(Xv), Yv, self.pw)
            self.history.append(vl)
            if not self.early_stop:   # ablation: no early stopping, keep the last epoch
                self.best_epoch = ep + 1
                snap = [{k: v.copy() for k, v in m.p.items()} for m in self.mods]
                bn_snap = [(m.rm.copy(), m.rv.copy()) for m in self.mods if isinstance(m, BN)]
                continue
            if vl < best - 1e-4:
                best, bad, self.best_epoch = vl, 0, ep + 1
                snap = [{k: v.copy() for k, v in m.p.items()} for m in self.mods]
                bn_snap = [(m.rm.copy(), m.rv.copy()) for m in self.mods if isinstance(m, BN)]
            else:
                bad += 1
                if bad >= self.patience:
                    break
        for m, s in zip(self.mods, snap):
            m.p = s
        for m, (a, b) in zip([m for m in self.mods if isinstance(m, BN)], bn_snap):
            m.rm, m.rv = a, b
        return self

    def embed(self, X):
        return np.vstack([self.body.fwd(X[i:i + 4096], False) for i in range(0, len(X), 4096)])

    def logits(self, X):
        return self.head.fwd(self.embed(X), False)

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self.logits(X)))
