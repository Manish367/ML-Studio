import os, time, uuid, threading
import numpy as np, pandas as pd
from fastapi import HTTPException, Header
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics as M
import tune as T

AUTOML_ROWS = int(
    os.getenv("AUTOML_MAX_ROWS", 20000)
)  # training rows used for the quick first pass


class AutoReq(BaseModel):
    dataset_id: str
    task: str
    target: str
    trials: int = 15
    seconds: int = 20
    top_k: int = 3
    test_size: float = 0.2
    scale: bool = True
    drop_duplicates: bool = True
    balance: bool = False


def explain(clf, ranked, base_acc):
    w = ranked[0]
    lab = "accuracy" if clf else "R²"
    out = [
        f"Best model: {w['model']}, with {lab} {w['final']:.3f} on test data it never saw."
    ]
    if w["tuned"]:
        out.append(f"Tuning lifted it from {w['default']:.3f} to {w['final']:.3f}.")
    elif w.get("note"):
        out.append(
            "Its default settings were kept because tuning did not beat them in cross-validation."
        )
    if len(ranked) > 1:
        s = ranked[1]
        gap = w["final"] - s["final"]
        t = f"Runner-up: {s['model']} ({s['final']:.3f}), {gap:.3f} behind."
        if gap < 0.01:
            t += " The top models are very close, so a simpler or faster one is also a good choice."
        out.append(t)
    if clf and base_acc is not None:
        out.append(
            f"For comparison, always guessing the most common class scores {base_acc:.3f}."
        )
        if w["final"] <= base_acc + 0.02:
            out.append(
                "The best model barely beats guessing, so these columns may not explain the target well."
            )
    if not clf:
        out.append(
            "R² of 0 means no better than always predicting the average, and 1 is perfect."
        )
    if not clf and w["final"] < 0.3:
        out.append("The R² is low, so these columns explain little of the target.")
    if w["final"] > 0.995:
        out.append(
            "This score is unusually high. Check that no other column gives away the target."
        )
    return out


def register_automl(app, DATA, MODELS, REG, CLF):
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
        if clf and getattr(T.LOCAL, "balance", False) and "class_weight" in est.get_params():
            est.set_params(class_weight="balanced")
        return Pipeline([("pre", pre), ("m", est)])

    def run(jid, r, clf, Xtr, Xte, ytr, yte, num, cat, le, n, secs):
        T.LOCAL.balance = r.balance
        J = T.JOBS[jid]
        scoring = "accuracy" if clf else "r2"
        names = list(CLF if clf else REG)
        test_score = (
            (lambda p: M.accuracy_score(yte, p.predict(Xte)))
            if clf
            else (lambda p: M.r2_score(yte, p.predict(Xte)))
        )
        second = (
            (
                lambda p: M.f1_score(
                    yte, p.predict(Xte), average="weighted", zero_division=0
                )
            )
            if clf
            else (lambda p: float(np.sqrt(M.mean_squared_error(yte, p.predict(Xte)))))
        )
        Xa, ya = Xtr, ytr
        if len(Xtr) > AUTOML_ROWS:
            idx = np.random.RandomState(42).choice(len(Xtr), AUTOML_ROWS, replace=False)
            Xa, ya = Xtr.iloc[idx], ytr.iloc[idx]

        def fit_data(name):
            if name in ("SVM", "SVR") and len(Xa) > 10000:
                idx = np.random.RandomState(42).choice(len(Xa), 10000, replace=False)
                return Xa.iloc[idx], ya.iloc[idx]
            return Xa, ya

        rows = {}
        try:
            for i, name in enumerate(
                names
            ):  # phase 1: every algorithm with default settings
                if J["cancel"]:
                    break
                J.update(
                    message=f"Training default models ({i + 1} of {len(names)}): {name}",
                    pct=round(50 * i / len(names)),
                )
                try:
                    Xf, yf = fit_data(name)
                    t = time.time()
                    p = make_pipe(name, clf, num, cat, r.scale)
                    p.fit(Xf, yf)
                    tt = time.time() - t
                    s = test_score(p)
                    rows[name] = {
                        "model": name,
                        "default": s,
                        "final": s,
                        "secondary": second(p),
                        "tuned": False,
                        "train_time": round(tt, 3),
                        "trained_rows": len(Xf),
                        "pipe": p,
                    }
                except Exception as e:
                    rows[name] = {"model": name, "error": str(e)}
            ok = sorted(
                [v for v in rows.values() if "error" not in v],
                key=lambda v: (-v["default"], v["train_time"]),
            )
            if not ok and not J["cancel"]:
                raise RuntimeError("No model could be trained on this data.")
            top = ok[: max(1, min(r.top_k, 5))]
            Xs, ys = Xtr, ytr
            if len(Xtr) > T.TUNE_ROWS:
                idx = np.random.RandomState(42).choice(
                    len(Xtr), T.TUNE_ROWS, replace=False
                )
                Xs, ys = Xtr.iloc[idx], ytr.iloc[idx]
            folds = min(3, int(ys.value_counts().min())) if clf else 3
            for j, v in enumerate(top):  # phase 2: tune only the best few
                if J["cancel"] or folds < 2:
                    break
                name = v["model"]
                if T.space(name, T.RandTrial(np.random.RandomState(0))) is None:
                    continue
                try:

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
                            pct=round(
                                50
                                + 50 * (j + min(1.0, max(k / n, el / secs))) / len(top)
                            ),
                            message=f"Tuning the top {len(top)}: {name}, trial {k} of {n}",
                        )

                    params, cv_best, hist = T.search(
                        name, cv_score, n, secs, tick, lambda: J["cancel"]
                    )
                    v.update(
                        cv_base=round(cv_base, 4),
                        cv_best=round(cv_best, 4) if params else None,
                    )
                    if params is not None and cv_best > cv_base:
                        Xf, yf = fit_data(name)
                        tp = make_pipe(name, clf, num, cat, r.scale)
                        tp.set_params(**params)
                        tp.fit(Xf, yf)
                        v.update(
                            pipe=tp,
                            final=test_score(tp),
                            secondary=second(tp),
                            tuned=True,
                            best_params={
                                k.replace("m__", ""): (
                                    float(f"{x:.4g}") if isinstance(x, float) else x
                                )
                                for k, x in params.items()
                            },
                        )
                    else:
                        v["note"] = (
                            "Tuning did not beat the defaults in cross-validation, so defaults were kept."
                        )
                except Exception as e:
                    v["note"] = f"Tuning skipped: {e}"
            ranked = sorted(
                [v for v in rows.values() if "error" not in v],
                key=lambda v: (-v["final"], v["train_time"]),
            )
            res = None
            if ranked:
                w = ranked[0]
                MODELS[f"{r.dataset_id}:AutoML best ({w['model']})"] = {
                    "pipe": w["pipe"],
                    "le": le,
                    "num": num,
                    "cat": cat,
                    "feats": T.feats_of(Xtr, num, cat),
                    "Xte": Xte,
                    "yte": yte,
                }
                base_acc = (
                    float((yte == ytr.value_counts().idxmax()).mean()) if clf else None
                )
                clean = []
                for k, v in enumerate(ranked):
                    d = {
                        a: (round(b, 4) if isinstance(b, float) else b)
                        for a, b in v.items()
                        if a != "pipe"
                    }
                    d["rank"] = k + 1
                    clean.append(d)
                clean += [
                    {k: v for k, v in x.items()} for x in rows.values() if "error" in x
                ]
                res = {
                    "task": r.task,
                    "target": r.target,
                    "metric": scoring,
                    "secondary": "f1" if clf else "rmse",
                    "models": clean,
                    "winner": w["model"],
                    "summary": explain(
                        clf,
                        [
                            dict(x, final=x["final"], default=x["default"])
                            for x in ranked
                        ],
                        base_acc,
                    ),
                    "optimizer": "Optuna (TPE)" if T.optuna else "Random search",
                }
            J.update(
                status="cancelled" if J["cancel"] else "done",
                pct=100 if not J["cancel"] else J["pct"],
                message="Finished",
                result=res,
            )
        except Exception as e:
            J.update(status="error", error=str(e))

    @app.post("/api/automl")
    def automl_start(r: AutoReq, x_client_id: str = Header(default="anon")):
        T.check_slot(x_client_id)
        if r.dataset_id not in DATA:
            raise HTTPException(404, "Dataset not found. Please reload your data.")
        clf = r.task == "classification"
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
        n, secs = max(5, min(r.trials, T.MAX_TRIALS)), max(
            5, min(r.seconds, T.MAX_SECONDS)
        )
        jid = uuid.uuid4().hex
        T.JOBS[jid] = {
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
        for old in list(T.JOBS)[:-10]:
            T.JOBS.pop(old, None)
        threading.Thread(
            target=run,
            args=(jid, r, clf, Xtr, Xte, ytr, yte, num, cat, le, n, secs),
            daemon=True,
        ).start()
        return {"job_id": jid}
