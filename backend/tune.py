import os, time, uuid, threading
import numpy as np
from typing import List
from fastapi import HTTPException, Header
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics as M
import pandas as pd

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:  # Optuna is optional: fall back to plain random search
    optuna = None

MAX_TRIALS = int(os.getenv("MAX_TUNE_TRIALS", 30))  # per model
MAX_SECONDS = int(os.getenv("MAX_TUNE_SECONDS", 90))  # per model
TUNE_ROWS = int(os.getenv("TUNE_MAX_ROWS", 10000))  # rows used inside the search
JOBS = {}
LOCAL = threading.local()  # per-job settings (the balance switch)
MAX_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", 2))


def check_slot(owner):
    running = [j for j in JOBS.values() if j["status"] == "running"]
    if any(j.get("owner") == owner for j in running):
        raise HTTPException(
            429,
            "You already have a job running. Wait for it to finish or cancel it first.",
        )
    if len(running) >= MAX_JOBS:
        raise HTTPException(
            429, "The server is busy with other jobs. Please try again in a moment."
        )


class TuneReq(BaseModel):
    dataset_id: str
    task: str
    target: str
    algos: List[str]
    trials: int = 20
    seconds: int = 30
    test_size: float = 0.2
    scale: bool = True
    drop_duplicates: bool = True
    balance: bool = False


class RandTrial:  # tiny stand-in for an Optuna trial, used when Optuna is not installed
    def __init__(s, rng):
        s.r = rng

    def suggest_int(s, n, lo, hi, step=1):
        return int(s.r.choice(np.arange(lo, hi + 1, step)))

    def suggest_float(s, n, lo, hi, log=False):
        return (
            float(np.exp(s.r.uniform(np.log(lo), np.log(hi))))
            if log
            else float(s.r.uniform(lo, hi))
        )

    def suggest_categorical(s, n, ch):
        return ch[int(s.r.randint(len(ch)))]


def space(name, t):
    """Search space per algorithm. Keys are Pipeline parameter names. Returns None if nothing to tune."""
    i = lambda k, lo, hi, **kw: t.suggest_int("m__" + k, lo, hi, **kw)
    f = lambda k, lo, hi, **kw: t.suggest_float("m__" + k, lo, hi, **kw)
    c = lambda k, ch: t.suggest_categorical("m__" + k, ch)
    if name == "Polynomial Regression":
        return {"m__polynomialfeatures__degree": i("polynomialfeatures__degree", 1, 4)}
    if name == "Ridge":
        return {"m__alpha": f("alpha", 1e-3, 1e3, log=True)}
    if name == "Lasso":
        return {"m__alpha": f("alpha", 1e-4, 10, log=True)}
    if name == "Logistic Regression":
        return {"m__C": f("C", 1e-3, 100, log=True)}
    if name == "Naive Bayes":
        return {"m__var_smoothing": f("var_smoothing", 1e-11, 1e-2, log=True)}
    if name == "KNN":
        return {
            "m__n_neighbors": i("n_neighbors", 1, 30),
            "m__weights": c("weights", ["uniform", "distance"]),
            "m__p": i("p", 1, 2),
        }
    if name in ("SVM", "SVR"):
        return {
            "m__C": f("C", 0.1, 100, log=True),
            "m__gamma": f("gamma", 1e-4, 1, log=True),
        }
    if name == "Decision Tree":
        return {
            "m__max_depth": i("max_depth", 2, 20),
            "m__min_samples_split": i("min_samples_split", 2, 20),
            "m__min_samples_leaf": i("min_samples_leaf", 1, 10),
        }
    if name in ("Random Forest", "Extra Trees"):
        return {
            "m__n_estimators": i("n_estimators", 50, 300, step=50),
            "m__max_depth": i("max_depth", 3, 20),
            "m__min_samples_leaf": i("min_samples_leaf", 1, 10),
        }
    if name == "Gradient Boosting":
        return {
            "m__n_estimators": i("n_estimators", 50, 250, step=50),
            "m__learning_rate": f("learning_rate", 0.01, 0.3, log=True),
            "m__max_depth": i("max_depth", 2, 6),
            "m__subsample": f("subsample", 0.6, 1.0),
        }
    if name == "AdaBoost":
        return {
            "m__n_estimators": i("n_estimators", 30, 300),
            "m__learning_rate": f("learning_rate", 0.01, 2, log=True),
        }
    return None


def search(name, score, n, secs, tick, stopped):
    t0, hist, best = time.time(), [], [-1e18, None]

    def record(p, v):
        if v > best[0]:
            best[0], best[1] = v, p
        hist.append(best[0])
        tick(len(hist), time.time() - t0)

    if optuna:
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )

        def obj(tr):
            p = space(name, tr)
            v = score(p)
            record(p, v)
            if stopped():
                study.stop()
            return v

        study.optimize(obj, n_trials=n, timeout=secs)
    else:
        rng = np.random.RandomState(42)
        while len(hist) < n and time.time() - t0 < secs and not stopped():
            p = space(name, RandTrial(rng))
            record(p, score(p))
    return best[1], best[0], hist


def feats_of(X, num, cat):
    """Input-form metadata for the Predict tab: a typical default for every column."""
    out = []
    for c in num:
        v = X[c].median()
        out.append(
            {"name": c, "type": "num", "default": float(v) if pd.notna(v) else 0.0}
        )
    for c in cat:
        s = X[c].dropna().astype(str)
        out.append(
            {
                "name": c,
                "type": "cat",
                "options": s.unique().tolist()[:50],
                "default": s.mode().iloc[0] if len(s) else "",
            }
        )
    return out


def register_tune(app, DATA, MODELS, REG, CLF):
    def make_pipe(name, clf, num, cat, scale):
        pre = ColumnTransformer(
            [
                (
                    "n",
                    Pipeline(
                        [
                            ("i", SimpleImputer(strategy="median")),
                            ("s", StandardScaler() if scale else "passthrough"),
                        ]
                    ),
                    num,
                ),
                (
                    "c",
                    Pipeline(
                        [
                            ("i", SimpleImputer(strategy="most_frequent")),
                            (
                                "o",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    cat,
                ),
            ]
        )
        est = (CLF if clf else REG)[name]()
        if clf and getattr(LOCAL, "balance", False) and "class_weight" in est.get_params():
            est.set_params(class_weight="balanced")
        return Pipeline([("pre", pre), ("m", est)])

    def run(jid, r, clf, Xtr, Xte, ytr, yte, num, cat, le, n, secs):
        LOCAL.balance = r.balance
        J = JOBS[jid]
        out = []
        scoring = "accuracy" if clf else "r2"
        test_score = (
            (lambda p: M.accuracy_score(yte, p.predict(Xte)))
            if clf
            else (lambda p: M.r2_score(yte, p.predict(Xte)))
        )
        try:
            for mi, name in enumerate(r.algos):
                if J["cancel"]:
                    break
                J.update(model=name, trial=0, message=f"Fitting the default {name}...")
                try:
                    t0 = time.time()
                    Xf, yf = Xtr, ytr
                    if name in ("SVM", "SVR") and len(Xtr) > 10000:
                        idx = np.random.RandomState(42).choice(
                            len(Xtr), 10000, replace=False
                        )
                        Xf, yf = Xtr.iloc[idx], ytr.iloc[idx]
                    base = make_pipe(name, clf, num, cat, r.scale)
                    base.fit(Xf, yf)
                    b_test = test_score(base)
                    if space(name, RandTrial(np.random.RandomState(0))) is None:
                        out.append(
                            {
                                "model": name,
                                "baseline": round(b_test, 4),
                                "note": "No hyperparameters to tune.",
                            }
                        )
                        continue
                    Xs, ys = Xtr, ytr
                    if len(Xtr) > TUNE_ROWS:
                        idx = np.random.RandomState(42).choice(
                            len(Xtr), TUNE_ROWS, replace=False
                        )
                        Xs, ys = Xtr.iloc[idx], ytr.iloc[idx]
                    folds = min(3, int(ys.value_counts().min())) if clf else 3
                    if folds < 2:
                        out.append(
                            {
                                "model": name,
                                "baseline": round(b_test, 4),
                                "note": "A class has too few rows to cross-validate.",
                            }
                        )
                        continue

                    def cv_score(params):
                        try:
                            p = make_pipe(name, clf, num, cat, r.scale)
                            p.set_params(**params)
                            return float(
                                cross_val_score(
                                    p, Xs, ys, cv=folds, scoring=scoring
                                ).mean()
                            )
                        except Exception:
                            return -1e9

                    cv_base = cv_score({})

                    def tick(k, el):
                        J.update(
                            trial=k,
                            trials=n,
                            pct=round(
                                100
                                * (mi + min(1.0, max(k / n, el / secs)))
                                / len(r.algos)
                            ),
                            message=f"Tuning {name}: trial {k} of {n}",
                        )

                    params, cv_best, hist = search(
                        name, cv_score, n, secs, tick, lambda: J["cancel"]
                    )
                    if params is None:
                        out.append(
                            {
                                "model": name,
                                "baseline": round(b_test, 4),
                                "note": "No trial finished within the time limit.",
                            }
                        )
                        continue
                    tuned = make_pipe(name, clf, num, cat, r.scale)
                    tuned.set_params(**params)
                    tuned.fit(Xf, yf)
                    MODELS[f"{r.dataset_id}:{name} (tuned)"] = {
                        "pipe": tuned,
                        "le": le,
                        "num": num,
                        "cat": cat,
                        "feats": feats_of(Xtr, num, cat),
                        "Xte": Xte,
                        "yte": yte,
                    }
                    out.append(
                        {
                            "model": name,
                            "baseline": round(b_test, 4),
                            "tuned": round(test_score(tuned), 4),
                            "cv_baseline": round(cv_base, 4),
                            "cv_best": round(cv_best, 4),
                            "trials": len(hist),
                            "best_params": {
                                k.replace("m__", ""): (
                                    float(f"{v:.4g}") if isinstance(v, float) else v
                                )
                                for k, v in params.items()
                            },
                            "history": [round(h, 4) for h in hist],
                            "seconds": round(time.time() - t0, 1),
                        }
                    )
                except Exception as e:
                    out.append({"model": name, "error": str(e)})
            J.update(
                status="cancelled" if J["cancel"] else "done",
                pct=100 if not J["cancel"] else J["pct"],
                message="Finished",
                result={
                    "task": r.task,
                    "metric": scoring,
                    "models": out,
                    "optimizer": "Optuna (TPE)" if optuna else "Random search",
                },
            )
        except Exception as e:
            J.update(status="error", error=str(e))

    @app.post("/api/tune")
    def tune_start(r: TuneReq, x_client_id: str = Header(default="anon")):
        check_slot(x_client_id)
        if r.dataset_id not in DATA:
            raise HTTPException(404, "Dataset not found. Please reload your data.")
        clf = r.task == "classification"
        bad = [a for a in r.algos if a not in (CLF if clf else REG)]
        if bad or not r.algos:
            raise HTTPException(400, f"Choose valid {r.task} algorithms first.")
        df = DATA[r.dataset_id].copy()
        if r.drop_duplicates:
            df = df.drop_duplicates()
        df = df.dropna(subset=[r.target])
        y = df[r.target]
        X = df.drop(columns=[r.target])
        if (
            clf
            and pd.api.types.is_numeric_dtype(y)
            and y.nunique() > 20
            and y.nunique() > 0.05 * len(y)
        ):
            raise HTTPException(
                400, f"'{r.target}' looks continuous. Choose Regression instead."
            )
        if not clf and not pd.api.types.is_numeric_dtype(y):
            raise HTTPException(
                400,
                f"'{r.target}' contains text or categories. Choose Classification instead.",
            )
        le = None
        if clf:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        num = X.select_dtypes("number").columns.tolist()
        cat = [c for c in X.columns if c not in num]
        strat = y if clf and y.value_counts().min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=r.test_size, random_state=42, stratify=strat
        )
        n, secs = max(5, min(r.trials, MAX_TRIALS)), max(5, min(r.seconds, MAX_SECONDS))
        jid = uuid.uuid4().hex
        JOBS[jid] = {
            "id": jid,
            "status": "running",
            "pct": 0,
            "message": "Starting...",
            "model": None,
            "trial": 0,
            "trials": n,
            "cancel": False,
            "result": None,
            "error": None,
            "owner": x_client_id,
        }
        for old in list(JOBS)[:-10]:
            JOBS.pop(old, None)
        threading.Thread(
            target=run,
            args=(jid, r, clf, Xtr, Xte, ytr, yte, num, cat, le, n, secs),
            daemon=True,
        ).start()
        return {"job_id": jid}

    @app.get("/api/job/{jid}")
    def job_status(jid: str):
        if jid not in JOBS:
            raise HTTPException(404, "Job not found. The server may have restarted.")
        return JOBS[jid]

    @app.post("/api/job/{jid}/cancel")
    def job_cancel(jid: str):
        if jid in JOBS:
            JOBS[jid]["cancel"] = True
        return {"ok": True}

    @app.get("/api/models/{dataset_id}")
    def list_models(dataset_id: str):
        p = dataset_id + ":"
        return [
            {
                "name": k[len(p) :],
                "task": "classification" if m["le"] is not None else "regression",
                "feats": m.get("feats", []),
            }
            for k, m in MODELS.items()
            if k.startswith(p)
        ]
