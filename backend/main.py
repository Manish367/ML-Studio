import os
from dotenv import load_dotenv

load_dotenv()
import threading, time, uuid, numpy as np, pandas as pd
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn import datasets, metrics as M
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    PolynomialFeatures,
    LabelEncoder,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    IsolationForest,
)
from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier,
    LocalOutlierFactor,
)
from sklearn.svm import SVR, SVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from fe import register, k_scan, pick_k
from detect import detect
from sklearn.datasets import make_blobs
from tune import register_tune, feats_of
import tune as T
from explain import register_explain
from insights import register_insights
from automl import register_automl
from fastapi.responses import JSONResponse

app = FastAPI(title="ML Studio")


@app.exception_handler(KeyError)
async def missing_data(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Dataset not found. The server may have restarted or cleared old data. Please reload your dataset."
        },
    )


ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
DATA: Dict[str, pd.DataFrame] = {}
MODELS: Dict[str, dict] = {}

rs = lambda cls, **kw: (lambda: cls(random_state=42, **kw))
REG = {
    "Linear Regression": LinearRegression,
    "Polynomial Regression": lambda: make_pipeline(
        PolynomialFeatures(2), LinearRegression()
    ),
    "Ridge": rs(Ridge),
    "Lasso": rs(Lasso),
    "Decision Tree": rs(DecisionTreeRegressor),
    "Random Forest": rs(RandomForestRegressor, n_estimators=100, n_jobs=-1),
    "KNN": lambda: KNeighborsRegressor(n_jobs=-1),
    "SVR": SVR,
    "Gradient Boosting": rs(GradientBoostingRegressor),
}
CLF = {
    "Logistic Regression": rs(LogisticRegression, max_iter=2000),
    "KNN": lambda: KNeighborsClassifier(n_jobs=-1),
    "Decision Tree": rs(DecisionTreeClassifier),
    "Random Forest": rs(RandomForestClassifier, n_estimators=100, n_jobs=-1),
    "SVM": rs(SVC, probability=True),
    "Naive Bayes": GaussianNB,
    "Gradient Boosting": rs(GradientBoostingClassifier),
    "AdaBoost": rs(AdaBoostClassifier),
    "Extra Trees": rs(ExtraTreesClassifier, n_jobs=-1),
}
CLU = {
    "kmeans": lambda k: KMeans(k, n_init=10, random_state=42),
    "dbscan": lambda k: DBSCAN(eps=1.2, min_samples=5),
    "agglomerative": lambda k: AgglomerativeClustering(k),
    "gmm": lambda k: GaussianMixture(k, random_state=42),
    "meanshift": lambda k: MeanShift(),
    "isolation_forest": lambda k: IsolationForest(contamination=0.05, random_state=42),
    "lof": lambda k: LocalOutlierFactor(contamination=0.05),
    "oneclass_svm": lambda k: OneClassSVM(nu=0.05),
}


def _mall():
    rng = np.random.RandomState(7)
    X, _ = make_blobs(
        200,
        centers=[[25, 80], [55, 50], [85, 20], [25, 20], [88, 82]],
        cluster_std=7,
        random_state=7,
    )
    return pd.DataFrame(
        {
            "Gender": rng.choice(["Male", "Female"], 200),
            "Age": rng.randint(18, 65, 200),
            "Annual Income (k$)": X[:, 0].clip(5, 140).round(),
            "Spending Score (1-100)": X[:, 1].clip(1, 99).round(),
        }
    )


def _fraud():
    rng = np.random.RandomState(11)
    n = 5000
    y = (rng.rand(n) < 0.015).astype(int)
    f = y == 1
    return pd.DataFrame(
        {
            "Amount": np.where(
                f, rng.lognormal(5.2, 0.9, n), rng.lognormal(3.6, 0.9, n)
            ).round(2),
            "Hour": np.where(
                f, rng.choice([0, 1, 2, 3, 4, 22, 23], n), rng.randint(6, 23, n)
            ),
            "DistanceFromHome_km": np.where(
                f, rng.exponential(120, n), rng.exponential(12, n)
            ).round(1),
            "Transactions_24h": np.where(f, rng.poisson(6, n), rng.poisson(2, n)),
            "ForeignMerchant": np.where(
                f, rng.rand(n) < 0.6, rng.rand(n) < 0.05
            ).astype(int),
            "Class": y,
        }
    )


def _wholesale():
    rng = np.random.RandomState(5)
    prof = [
        [9.3, 8.0, 8.3, 7.8, 5.5, 6.9],
        [8.0, 9.0, 9.5, 6.6, 8.6, 6.7],
        [7.4, 6.8, 7.2, 6.6, 5.6, 5.9],
    ]
    cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    parts = [
        pd.DataFrame(np.exp(rng.normal(mu, 0.6, (150, 6))).round(), columns=cols)
        for mu in prof
    ]
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=1)
        .reset_index(drop=True)
    )


_f = lambda fn, drop=None: (
    lambda: (
        fn(as_frame=True).frame.drop(columns=drop) if drop else fn(as_frame=True).frame
    )
)
SAMPLES = {  # id: (label, task, loader, description)
    "iris": (
        "Iris flowers",
        "classification",
        _f(datasets.load_iris),
        "150 flowers, predict the species",
    ),
    "wine": (
        "Wine types",
        "classification",
        _f(datasets.load_wine),
        "178 wines, predict the cultivar",
    ),
    "breast_cancer": (
        "Breast cancer",
        "classification",
        _f(datasets.load_breast_cancer),
        "569 tumours, benign or malignant",
    ),
    "diabetes": (
        "Diabetes progression",
        "regression",
        _f(datasets.load_diabetes),
        "442 patients, predict disease progression",
    ),
    "iris_unlabeled": (
        "Iris (no labels)",
        "clustering",
        _f(datasets.load_iris, "target"),
        "Same flowers without the species column",
    ),
    "wine_unlabeled": (
        "Wine (no labels)",
        "clustering",
        _f(datasets.load_wine, "target"),
        "Same wines without the cultivar column",
    ),
    "mall_customers": (
        "Mall customers",
        "clustering",
        _mall,
        "200 synthetic customers, find spending segments",
    ),
    "fraud_demo": (
        "Credit card fraud (synthetic)",
        "classification",
        _fraud,
        "5,000 transactions, about 1.5% fraud. Good for imbalanced data",
    ),
    "wholesale_demo": (
        "Wholesale customers (synthetic)",
        "clustering",
        _wholesale,
        "450 customers with skewed spending. Try a log transform",
    ),
}


MAX_DATASETS = int(os.getenv("MAX_DATASETS", 10))


def store(df):
    i = uuid.uuid4().hex
    DATA[i] = df
    for old in list(DATA)[:-MAX_DATASETS]:  # keep only the newest datasets in memory
        DATA.pop(old, None)
        for k in [k for k in MODELS if k.startswith(old + ":")]:
            MODELS.pop(k, None)
        fe_forget(old)
    return info(i)


def info(i):
    for k in [k for k in MODELS if k.startswith(i + ":")]:
        MODELS.pop(k, None)  # data changed or new: drop stale models
    df = DATA[i]
    cols = [
        {
            "name": c,
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "unique": int(df[c].nunique()),
            "numeric": bool(pd.api.types.is_numeric_dtype(df[c])),
        }
        for c in df.columns
    ]
    return {
        "id": i,
        "detected": detect(df),
        "rows": len(df),
        "cols": cols,
        "duplicates": int(df.duplicated().sum()),
        "preview": df.head(8).fillna("").astype(str).to_dict("records"),
    }


@app.get("/api/algos")
def algos():
    return {
        "regression": list(REG),
        "classification": list(CLF),
        "samples": [
            {
                "id": k,
                "label": v[0],
                "task": v[1],
                "group": "unsupervised" if v[1] == "clustering" else "supervised",
                "desc": v[3],
            }
            for k, v in SAMPLES.items()
        ],
        "clustering": list(CLU),
    }


@app.post("/api/sample/{name}")
def sample(name: str):
    if name not in SAMPLES:
        raise HTTPException(404, "Unknown sample")
    return store(SAMPLES[name][2]().copy())


MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 500))


def read_csv_fast(f):
    try:
        return pd.read_csv(f, engine="pyarrow")
    except Exception:
        f.seek(0)
        return pd.read_csv(f)


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    if file.size and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            413,
            f"File is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller sample.",
        )
    try:
        df = read_csv_fast(file.file)
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    return store(df)


@app.get("/api/eda/{i}")
def eda(i: str):
    df = DATA[i]
    num = df.select_dtypes("number")
    hist = {}
    for c in num.columns[:12]:
        h, e = np.histogram(num[c].dropna(), bins=15)
        hist[c] = [
            {"x": round(float((e[j] + e[j + 1]) / 2), 3), "count": int(h[j])}
            for j in range(len(h))
        ]
    box = {}
    for c in num.columns[:12]:
        s = num[c].dropna()
        if len(s) == 0:
            continue
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        box[c] = {
            k: round(float(v), 4)
            for k, v in {
                "min": s.min(),
                "q1": q1,
                "median": med,
                "q3": q3,
                "max": s.max(),
                "lo": s[s >= lo_f].min(),
                "hi": s[s <= hi_f].max(),
            }.items()
        }
        box[c]["outliers"] = int(((s < lo_f) | (s > hi_f)).sum())
    corr = num.corr().round(2).fillna(0)
    desc = (
        num.describe()
        .T.round(3)
        .reset_index()
        .rename(columns={"index": "feature"})
        .to_dict("records")
    )
    return {
        "box": box,
        "hist": hist,
        "corr": {"cols": list(corr.columns), "z": corr.values.tolist()},
        "describe": desc,
    }


class TrainReq(BaseModel):
    dataset_id: str
    task: str
    target: str
    algos: List[str]
    test_size: float = 0.2
    scale: bool = True
    drop_duplicates: bool = True
    balance: bool = False


def downsample(a, n=120):
    idx = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
    return [a[j] for j in idx]


@app.post("/api/train")
def train(r: TrainReq):
    try:
        return _train(r)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Training failed: {e}")


def _train(r: TrainReq, tick=None):
    df = DATA[r.dataset_id].copy()
    if r.drop_duplicates:
        df = df.drop_duplicates()
    df = df.dropna(subset=[r.target])
    y = df[r.target]
    X = df.drop(columns=[r.target])
    clf = r.task == "classification"
    if (
        clf
        and pd.api.types.is_numeric_dtype(y)
        and y.nunique() > 20
        and y.nunique() > 0.05 * len(y)
    ):
        raise HTTPException(
            400,
            f"'{r.target}' has {y.nunique()} distinct numeric values, so it looks continuous. Choose Regression instead.",
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
    out, details = [], {}
    for name in r.algos:
        if tick:
            tick(
                100 * len(out) // len(r.algos),
                f"Training {name} ({len(out) + 1} of {len(r.algos)})...",
            )
        pre = ColumnTransformer(
            [
                (
                    "n",
                    Pipeline(
                        [
                            ("i", SimpleImputer(strategy="median")),
                            ("s", StandardScaler() if r.scale else "passthrough"),
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
        if clf and r.balance and "class_weight" in est.get_params():
            est.set_params(class_weight="balanced")
        pipe = Pipeline([("pre", pre), ("m", est)])
        Xf, yf = Xtr, ytr
        if name in ("SVM", "SVR") and len(Xtr) > 10000:
            idx = np.random.RandomState(42).choice(len(Xtr), 10000, replace=False)
            Xf, yf = Xtr.iloc[idx], ytr.iloc[idx]
        t = time.time()
        pipe.fit(Xf, yf)
        tt = time.time() - t
        p = pipe.predict(Xte)

        folds = min(3, int(yf.value_counts().min())) if clf else 3
        cv = (
            cross_val_score(
                pipe,
                Xf,
                yf,
                cv=folds,
                scoring="accuracy" if clf else "r2",
                n_jobs=3 if len(Xf) > 5000 else 1,
            )
            if folds >= 2 and len(Xf) <= 20000
            else None
        )
        d = {}
        if clf:
            m = {
                "accuracy": M.accuracy_score(yte, p),
                "precision": M.precision_score(
                    yte, p, average="weighted", zero_division=0
                ),
                "recall": M.recall_score(yte, p, average="weighted", zero_division=0),
                "f1": M.f1_score(yte, p, average="weighted", zero_division=0),
                "balanced_accuracy": M.balanced_accuracy_score(yte, p),
            }
            d["confusion"] = M.confusion_matrix(
                yte, p, labels=range(len(le.classes_))
            ).tolist()
            d["classes"] = list(le.classes_)
            d["report"] = M.classification_report(
                yte,
                p,
                labels=range(len(le.classes_)),
                target_names=list(le.classes_),
                output_dict=True,
                zero_division=0,
            )
            try:
                pr = pipe.predict_proba(Xte)
                if pr.shape[1] == 2:
                    m["roc_auc"] = M.roc_auc_score(yte, pr[:, 1])
                    m["pr_auc"] = M.average_precision_score(yte, pr[:, 1])
                    f, t_, _ = M.roc_curve(yte, pr[:, 1])
                    d["roc"] = [
                        {"x": float(a), "y": float(b)}
                        for a, b in downsample(list(zip(f, t_)))
                    ]
                    pp, rr, _ = M.precision_recall_curve(yte, pr[:, 1])
                    d["pr"] = [
                        {"x": float(b), "y": float(a)}
                        for a, b in downsample(list(zip(pp, rr)))
                    ]
                else:
                    m["roc_auc"] = M.roc_auc_score(yte, pr, multi_class="ovr")
                    d["roc_multi"] = []
                    for ci, cname in enumerate(le.classes_[:10]):
                        yb = (yte.to_numpy() == ci).astype(int)
                        if ci < pr.shape[1] and 0 < yb.sum() < len(yb):
                            fp_, tp_, _ = M.roc_curve(yb, pr[:, ci])
                            d["roc_multi"].append(
                                {
                                    "class": str(cname),
                                    "points": [
                                        {"x": float(a), "y": float(b)}
                                        for a, b in downsample(list(zip(fp_, tp_)), 60)
                                    ],
                                }
                            )
            except Exception:
                pass
        else:
            mse = M.mean_squared_error(yte, p)
            m = {
                "mae": M.mean_absolute_error(yte, p),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "r2": M.r2_score(yte, p),
                "within_10pct": float(
                    np.mean(np.abs(p - yte.values) <= 0.1 * np.abs(yte.values))
                ),
            }
            d["points"] = [
                {"actual": float(a), "pred": float(b)}
                for a, b in downsample(list(zip(yte, p)), 200)
            ]
            d["residuals"] = [float(v) for v in (yte.values - p)]
        mod = pipe.named_steps["m"]
        names = pre.get_feature_names_out()
        imp = getattr(mod, "feature_importances_", None)
        if imp is None and hasattr(mod, "coef_"):
            imp = np.abs(np.atleast_2d(mod.coef_)).mean(0)
        if imp is not None and len(imp) == len(names):
            top = np.argsort(imp)[::-1][:10]
            d["importance"] = [
                {"feature": str(names[j]).split("__")[-1], "value": float(imp[j])}
                for j in top
            ]
        details[name] = d
        out.append(
            {
                "model": name,
                **{k: round(float(v), 4) for k, v in m.items()},
                "cv_mean": round(float(cv.mean()), 4) if cv is not None else None,
                "cv_std": round(float(cv.std()), 4) if cv is not None else None,
                "trained_rows": len(Xf),
                "train_time": round(tt, 3),
            }
        )
        MODELS[f"{r.dataset_id}:{name}"] = {
            "pipe": pipe,
            "le": le,
            "num": num,
            "cat": cat,
            "feats": feats_of(Xtr, num, cat),
            "Xte": Xte,
            "yte": yte,
        }
    feats = [
        {"name": c, "type": "num", "default": float(X[c].median())} for c in num
    ] + [
        {
            "name": c,
            "type": "cat",
            "options": X[c].dropna().astype(str).unique().tolist()[:50],
        }
        for c in cat
    ]
    return {"results": out, "details": details, "features": feats, "task": r.task}


class PredReq(BaseModel):
    model_id: str
    row: dict


@app.post("/api/predict")
def predict(r: PredReq):
    m = MODELS.get(r.model_id)
    if not m:
        raise HTTPException(404, "Train this model first")
    row = {k: (float(v) if k in m["num"] else str(v)) for k, v in r.row.items()}
    df = pd.DataFrame([row])
    p = m["pipe"].predict(df)[0]
    res = {
        "prediction": (
            str(m["le"].inverse_transform([int(p)])[0]) if m["le"] else float(p)
        )
    }
    try:
        if m["le"]:
            res["proba"] = {
                str(c): float(v)
                for c, v in zip(m["le"].classes_, m["pipe"].predict_proba(df)[0])
            }
    except Exception:
        pass
    return res


class ClusterReq(BaseModel):
    dataset_id: str
    algo: str
    k: int = 3
    features: Optional[List[str]] = None
    k_method: str = "manual"
    eps: float = 1.2
    min_samples: int = 5
    contamination: float = 0.05
    label_col: Optional[str] = None


@app.post("/api/cluster")
def cluster(r: ClusterReq):
    return _cluster(r)


def _cluster(r: ClusterReq, tick=None):
    tk = tick or (lambda p, msg: None)
    tk(2, "Scaling the features...")

    df = DATA[r.dataset_id]
    num = df.select_dtypes("number")
    cols = [c for c in (r.features or list(num.columns)) if c in num.columns]
    X = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(num[cols])
    )
    rng = np.random.RandomState(42)
    total = len(X)
    cap = (
        8000
        if r.algo in ("agglomerative", "meanshift", "lof", "oneclass_svm")
        else 50000
    )
    keep = None
    if total > cap:
        keep = rng.choice(total, cap, replace=False)
        X = X[keep]
        uses_k = r.algo in ("kmeans", "agglomerative", "gmm")
    rows = (
        k_scan(
            X,
            lambda i, n: tk(5 + 50 * i // n, f"Testing K = {i + 2} (step {i + 1} of {n})..."),
        )
        if uses_k
        else []
    )
    picks = pick_k(rows) if rows else {}
    k = r.k if r.k_method == "manual" else picks.get(r.k_method, r.k)
    tk(60, f"Fitting {r.algo}...")
    est = CLU[r.algo](k)
    cont = max(0.001, min(r.contamination, 0.5))
    if r.algo == "dbscan":
        est = DBSCAN(eps=r.eps, min_samples=r.min_samples)
    elif r.algo == "isolation_forest":
        est = IsolationForest(contamination=cont, random_state=42)
    elif r.algo == "lof":
        est = LocalOutlierFactor(contamination=cont)
    elif r.algo == "oneclass_svm":
        est = OneClassSVM(nu=cont)
    lab = est.fit_predict(X) if r.algo != "gmm" else est.fit(X).predict(X)
    lab = np.asarray(lab)
    label_check = None
    if (
        r.algo in ("isolation_forest", "lof", "oneclass_svm")
        and r.label_col in df.columns
        and df[r.label_col].nunique(dropna=True) == 2
    ):
        yl = df[r.label_col].reset_index(drop=True)
        pos_value = (
            yl.value_counts().idxmin()
        )  # the rarer value is treated as the true outlier
        if keep is not None:
            yl = yl.iloc[keep]
        truth = (yl == pos_value).to_numpy()
        flag = lab == -1
        tp = int((truth & flag).sum())
        fp = int((~truth & flag).sum())
        fn = int((truth & ~flag).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        label_check = {
            "label": r.label_col,
            "positive": str(pos_value),
            "true_outliers": int(truth.sum()),
            "flagged": int(flag.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(pr, 4),
            "recall": round(rc, 4),
            "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
        }
    m = {}
    mask = lab != -1
    n_cl = len(set(lab[mask]))
    if r.algo in ("isolation_forest", "lof", "oneclass_svm"):
        m["anomalies"] = int((lab == -1).sum())
        m["normal"] = int(mask.sum())
    elif n_cl > 1 and mask.sum() > n_cl:
        m = {
            "silhouette": M.silhouette_score(
                X[mask], lab[mask], sample_size=5000, random_state=42
            ),
            "davies_bouldin": M.davies_bouldin_score(X[mask], lab[mask]),
            "calinski_harabasz": M.calinski_harabasz_score(X[mask], lab[mask]),
        }
        m = {k: round(float(v), 4) for k, v in m.items()}
        m["clusters"] = n_cl
        m["noise"] = int((~mask).sum())
    if r.algo == "kmeans":
        m["inertia"] = round(float(est.inertia_), 2)
    sil_plot = None
    if (
        r.algo not in ("isolation_forest", "lof", "oneclass_svm")
        and n_cl > 1
        and mask.sum() > n_cl
    ):
        idx_m = np.where(mask)[0]
        if len(idx_m) > 400:
            idx_m = rng.choice(idx_m, 400, replace=False)
        if len(set(lab[idx_m])) > 1:
            sv = M.silhouette_samples(X[idx_m], lab[idx_m])
            sil_plot = {
                "avg": round(float(sv.mean()), 4),
                "clusters": [
                    {
                        "c": int(c),
                        "values": [
                            round(float(v), 3)
                            for v in np.sort(sv[lab[idx_m] == c])[::-1]
                        ],
                    }
                    for c in np.unique(lab[idx_m])
                ],
            }
    dendro = None
    if r.algo == "agglomerative" and len(X) >= 3:
        from scipy.cluster.hierarchy import linkage, dendrogram

        xs = X if len(X) <= 300 else X[rng.choice(len(X), 300, replace=False)]
        Zl = linkage(xs, "ward")
        dd = dendrogram(Zl, truncate_mode="lastp", p=30, no_plot=True)
        mrg = len(xs) - 1
        dendro = {
            "i": dd["icoord"],
            "d": dd["dcoord"],
            "rows": len(xs),
            "cut": (
                float((Zl[mrg - k, 2] + Zl[mrg - k + 1, 2]) / 2)
                if 2 <= k <= mrg
                else None
            ),
        }
    tk(90, "Projecting to 2D with PCA...")
    pc = PCA(2, random_state=42).fit(X)
    Z = pc.transform(X)
    elbow = rows
    sel = rng.choice(len(Z), min(5000, len(Z)), replace=False)
    pts = [{"x": float(Z[j][0]), "y": float(Z[j][1]), "c": int(lab[j])} for j in sel]
    counts = {str(k): int(v) for k, v in zip(*np.unique(lab, return_counts=True))}
    return {
        "anomaly": r.algo in ("isolation_forest", "lof", "oneclass_svm"),
        "dendro": dendro,
        "silhouette_plot": sil_plot,
        "label_check": label_check,
        "sampled_rows": len(X),
        "total_rows": total,
        "k_used": int(k),
        "k_picks": picks,
        "metrics": m,
        "points": pts,
        "elbow": elbow,
        "counts": counts,
        "var": [round(float(v) * 100, 1) for v in pc.explained_variance_ratio_],
    }


fe_forget = register(app, DATA, info)
register_tune(app, DATA, MODELS, REG, CLF)
register_automl(app, DATA, MODELS, REG, CLF)
register_explain(app, MODELS)


@app.get("/api/eda/{i}/scatter")
def eda_scatter(i: str, x: str, y: str, color: Optional[str] = None):
    df = DATA[i]
    for c in (x, y):
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            raise HTTPException(400, f"'{c}' is not a numeric column.")
    cols = [x, y] + ([color] if color and color in df.columns else [])
    d = df[cols].dropna()
    if len(d) > 800:
        d = d.sample(800, random_state=42)
    cvals = d[cols[2]] if len(cols) == 3 else [None] * len(d)
    return {
        "points": [
            {"x": float(a), "y": float(b), "c": (str(c) if len(cols) == 3 else "all")}
            for a, b, c in zip(d[x], d[y], cvals)
        ]
    }


register_insights(app, DATA)


def _start_job(work, dataset_id, owner):
    if dataset_id not in DATA:
        raise HTTPException(
            404,
            "Dataset not found. The server may have restarted or cleared old data. Please reload your dataset.",
        )
    T.check_slot(owner)
    jid = uuid.uuid4().hex
    T.JOBS[jid] = {
        "id": jid,
        "status": "running",
        "pct": 0,
        "message": "Starting...",
        "cancel": False,
        "result": None,
        "error": None,
        "owner": owner,
    }
    for old in list(T.JOBS)[:-10]:
        T.JOBS.pop(old, None)

    def go():
        J = T.JOBS[jid]
        try:
            res = work(lambda p, msg: J.update(pct=int(p), message=msg))
            J.update(status="done", pct=100, message="Finished", result=res)
        except HTTPException as e:
            J.update(status="error", error=str(e.detail))
        except Exception as e:
            J.update(status="error", error=f"Run failed: {e}")

    threading.Thread(target=go, daemon=True).start()
    return {"job_id": jid}


@app.post("/api/train/start")
def train_start(r: TrainReq, x_client_id: str = Header(default="anon")):
    return _start_job(lambda tick: _train(r, tick), r.dataset_id, x_client_id)


@app.post("/api/cluster/start")
def cluster_start(r: ClusterReq, x_client_id: str = Header(default="anon")):
    return _start_job(lambda tick: _cluster(r, tick), r.dataset_id, x_client_id)
