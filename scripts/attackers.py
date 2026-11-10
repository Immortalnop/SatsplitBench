"""Phase 4b attackers: deep residual MLP, LightGBM, RBF-kernel SVM (Nystroem).

One entry point, `run_attacker(name, ...)`, for three task types:
  binary      y (N,) in {0,1}        -> scores (N,)      selection metric: ROC-AUC
  multiclass  y (N,) in {0..K-1}     -> probs  (N, K)    selection metric: accuracy
  multilabel  y (N, L) in {0,1}      -> scores (N, L)    selection metric: macro average precision

Every hyperparameter is chosen on the validation set only; the returned test scores come from
the configuration with the best validation metric. Inputs are standardised on train.

macOS note: torch bundles its own OpenMP runtime and LightGBM uses Homebrew's libomp. If torch loads
first, LightGBM crashes ("OMP: Error #179"). LightGBM is therefore imported first, here; any script
using this module must import it before importing torch.
"""
import lightgbm  # noqa: F401  (must load before torch, see above)
import itertools
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ATTACKERS = ["deep_mlp", "lightgbm", "svm_rbf"]


def val_metric(task, y, s):
    if task == "binary":
        return roc_auc_score(y, s)
    if task == "multiclass":
        return accuracy_score(y, s.argmax(1))
    keep = (y.sum(0) > 0) & (y.sum(0) < len(y))
    return average_precision_score(y[:, keep], s[:, keep], average="macro")


def n_outputs(task, y):
    if task == "binary":
        return 1
    if task == "multiclass":
        return int(y.max()) + 1
    return y.shape[1]


# ---------------------------------------------------------------- deep residual MLP

class ResBlock(torch.nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(w), torch.nn.Linear(w, w * 2), torch.nn.GELU(),
                                       torch.nn.Dropout(p), torch.nn.Linear(w * 2, w))

    def forward(self, x):
        return x + self.net(x)


class DeepMLP(torch.nn.Module):
    def __init__(self, d_in, d_out, width=512, depth=3, p=0.1):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(d_in, width), torch.nn.Dropout(p))
        self.blocks = torch.nn.Sequential(*[ResBlock(width, p) for _ in range(depth)])
        self.out = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, d_out))

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


def _mlp_fit(xtr, ytr, xva, yva, task, lr, wd, p, seed, device, epochs=200, patience=12, batch=512):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    k = n_outputs(task, ytr)
    model = DeepMLP(xtr.shape[1], k, p=p).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    if task == "binary":
        pos = float(ytr.mean())
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32,
                                                                   device=device))
        tgt = lambda y: torch.as_tensor(y, dtype=torch.float32, device=device)[:, None]
    elif task == "multiclass":
        lossf = torch.nn.CrossEntropyLoss()
        tgt = lambda y: torch.as_tensor(y, dtype=torch.long, device=device)
    else:
        lossf = torch.nn.BCEWithLogitsLoss()
        tgt = lambda y: torch.as_tensor(y, dtype=torch.float32, device=device)
    xtr_t = torch.as_tensor(xtr, device=device)
    xva_t = torch.as_tensor(xva, device=device)

    def predict(x_t):
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(x_t[i:i + 8192]) for i in range(0, len(x_t), 8192)])
        if task == "binary":
            return torch.sigmoid(out[:, 0]).cpu().numpy()
        if task == "multiclass":
            return torch.softmax(out, 1).cpu().numpy()
        return torch.sigmoid(out).cpu().numpy()

    best, best_state, bad = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(len(xtr)), max(1, len(xtr) // batch)):
            opt.zero_grad()
            lossf(model(xtr_t[b]), tgt(ytr[b])).backward()
            opt.step()
        score = val_metric(task, yva, predict(xva_t))
        sched.step(-score)
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {n: t.detach().clone() for n, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best, predict


def deep_mlp(xtr, ytr, xva, yva, xte, task, seed, device):
    grid = list(itertools.product([1e-3, 3e-4], [1e-4, 1e-2], [0.1, 0.3]))
    best = None
    for lr, wd, p in grid:
        score, predict = _mlp_fit(xtr, ytr, xva, yva, task, lr, wd, p, seed, device)
        if best is None or score > best[0]:
            best = (score, {"lr": lr, "wd": wd, "dropout": p}, predict(torch.as_tensor(xte, device=device)),
                    predict(torch.as_tensor(xva, device=device)))
    return best


# ---------------------------------------------------------------- LightGBM

def lightgbm(xtr, ytr, xva, yva, xte, task, seed, device=None, grid=None):
    import lightgbm as lgb

    grid = grid or [{"num_leaves": 31, "colsample_bytree": 0.5}, {"num_leaves": 127, "colsample_bytree": 0.5},
                    {"num_leaves": 31, "colsample_bytree": 0.9}, {"num_leaves": 127, "colsample_bytree": 0.9}]
    if task == "multilabel":  # one model per label; smaller grid keeps the cost sane
        grid = grid[:2]

    def fit_one(y_tr, y_va, objective, params, num_class=None):
        # 63 histogram bins and lr 0.1 with early stopping: ~3-5x faster than defaults on 2048-dim inputs,
        # standard for high-dimensional dense features
        kw = dict(n_estimators=2000, learning_rate=0.1, max_bin=63, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1,
                  objective=objective, **params)
        if objective == "binary":
            kw["is_unbalance"] = True
        if objective == "multiclass":
            kw["num_class"] = num_class
        m = lgb.LGBMClassifier(**kw)
        m.fit(xtr, y_tr, eval_set=[(xva, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        return m

    best = None
    for params in grid:
        if task == "binary":
            m = fit_one(ytr, yva, "binary", params)
            sva, ste = m.predict_proba(xva)[:, 1], m.predict_proba(xte)[:, 1]
        elif task == "multiclass":
            m = fit_one(ytr, yva, "multiclass", params, num_class=int(ytr.max()) + 1)
            sva, ste = m.predict_proba(xva), m.predict_proba(xte)
        else:
            sva = np.zeros(yva.shape, np.float32)
            ste = np.zeros((len(xte), ytr.shape[1]), np.float32)
            for j in range(ytr.shape[1]):
                if ytr[:, j].min() == ytr[:, j].max():
                    continue
                m = fit_one(ytr[:, j], yva[:, j], "binary", params)
                sva[:, j], ste[:, j] = m.predict_proba(xva)[:, 1], m.predict_proba(xte)[:, 1]
        score = val_metric(task, yva, sva)
        if best is None or score > best[0]:
            best = (score, dict(params), ste, sva)
    return best


# ---------------------------------------------------------------- RBF-kernel SVM (Nystroem)

def svm_rbf(xtr, ytr, xva, yva, xte, task, seed, device=None, n_components=2000):
    from sklearn.kernel_approximation import Nystroem
    from sklearn.svm import LinearSVC

    d = xtr.shape[1]  # inputs are standardised, so gamma = 1/d is sklearn's "scale"
    best = None
    for gmul, c in itertools.product([0.5, 1.0, 2.0], [0.1, 1.0]):
        ny = Nystroem(kernel="rbf", gamma=gmul / d, n_components=min(n_components, len(xtr)), random_state=seed)
        ftr = ny.fit_transform(xtr)
        fva, fte = ny.transform(xva), ny.transform(xte)

        def svc(y):
            return LinearSVC(C=c, class_weight="balanced" if task != "multiclass" else None,
                             max_iter=5000, random_state=seed).fit(ftr, y)

        if task == "binary":
            m = svc(ytr)
            sva, ste = m.decision_function(fva), m.decision_function(fte)
        elif task == "multiclass":
            m = svc(ytr)
            sva, ste = m.decision_function(fva), m.decision_function(fte)
        else:
            sva = np.zeros(yva.shape, np.float32)
            ste = np.zeros((len(xte), ytr.shape[1]), np.float32)
            for j in range(ytr.shape[1]):
                if ytr[:, j].min() == ytr[:, j].max():
                    continue
                m = svc(ytr[:, j])
                sva[:, j], ste[:, j] = m.decision_function(fva), m.decision_function(fte)
        score = val_metric(task, yva, sva)
        if best is None or score > best[0]:
            best = (score, {"gamma": f"{gmul}/d", "C": c}, ste, sva)
    return best


# ---------------------------------------------------------------- entry point

def run_attacker(name, xtr, ytr, xva, yva, xte, task, seed=0, device="cpu"):
    """Returns dict(val_score, params, test_scores, val_scores, seconds)."""
    t0 = time.time()
    sc = StandardScaler().fit(xtr)
    xtr, xva, xte = (sc.transform(a).astype(np.float32) for a in (xtr, xva, xte))
    fn = {"deep_mlp": deep_mlp, "lightgbm": lightgbm, "svm_rbf": svm_rbf}[name]
    score, params, ste, sva = fn(xtr, ytr, xva, yva, xte, task, seed, device)
    return {"val_score": float(score), "params": params, "test_scores": np.asarray(ste, np.float32),
            "val_scores": np.asarray(sva, np.float32),
            "seconds": time.time() - t0}
