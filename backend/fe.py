import numpy as np, pandas as pd
from typing import List
from fastapi import HTTPException
from pydantic import BaseModel
from sklearn import metrics as M
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    KBinsDiscretizer,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


# ---------- automatic K selection ----------
def k_scan(X):
    rows = []
    for k in range(2, min(10, len(X) - 1) + 1):
        km = KMeans(k, n_init=5, random_state=42).fit(X)
        l, c = km.labels_, km.cluster_centers_
        d = np.linalg.norm(c[:, None] - c[None], axis=2)
        np.fill_diagonal(d, np.inf)
        rad = (
            max(np.linalg.norm(X[l == j] - c[j], axis=1).mean() for j in range(k))
            + 1e-9
        )
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(
                    M.silhouette_score(X, l, sample_size=5000, random_state=42)
                ),
                "calinski": float(M.calinski_harabasz_score(X, l)),
                "davies_bouldin": float(M.davies_bouldin_score(X, l)),
                "nearest_cluster": float(d.min() / rad),
            }
        )  # gap to nearest other cluster vs cluster radius
    return rows


def pick_k(rows):
    ks = np.array([r["k"] for r in rows])
    g = lambda n: np.array([r[n] for r in rows])
    elbow = ks[0]
    if (
        len(ks) >= 3
    ):  # knee = point furthest from the line joining first and last inertia values
        x = (ks - ks[0]) / (ks[-1] - ks[0])
        y = (g("inertia") - g("inertia")[-1]) / (
            g("inertia")[0] - g("inertia")[-1] + 1e-12
        )
        elbow = ks[
            np.argmax(np.abs((y[-1] - y[0]) * (x - x[0]) - (x[-1] - x[0]) * (y - y[0])))
        ]
    return {
        "elbow": int(elbow),
        "silhouette": int(ks[g("silhouette").argmax()]),
        "calinski": int(ks[g("calinski").argmax()]),
        "davies_bouldin": int(ks[g("davies_bouldin").argmin()]),
        "nearest_cluster": int(ks[g("nearest_cluster").argmax()]),
    }


# ---------- feature engineering ----------
class FEReq(BaseModel):
    dataset_id: str
    op: str
    cols: List[str] = []
    params: dict = {}


def run_op(df, op, cols, p):
    allnum = df.select_dtypes("number").columns.tolist()
    nc = [c for c in cols if c in allnum] if cols else allnum
    mode_of = lambda s: s.mode().iloc[0]
    if op == "fill_missing":
        tgt = cols or df.columns.tolist()
        s = p.get("strategy", "median")
        before = int(df[tgt].isna().sum().sum())
        if s == "knn":
            n = [c for c in tgt if c in allnum]
            df[n] = KNNImputer().fit_transform(df[n])
        for c in ([] if s == "knn" else tgt):
            if not df[c].isna().any():
                continue
            isn = c in allnum
            if s == "ffill":
                df[c] = df[c].ffill().bfill()
            elif s == "constant":
                v = p.get("value", "0")
                df[c] = df[c].fillna(float(v) if isn else v)
            elif s in ("mean", "median") and isn:
                df[c] = df[c].fillna(getattr(df[c], s)())
            else:
                df[c] = df[c].fillna(mode_of(df[c]))
        return (
            df,
            f"Filled {before - int(df[tgt].isna().sum().sum())} missing cells ({s})",
        )
    if op == "drop_missing":
        th = float(p.get("threshold", 30)) / 100
        n0, c0 = df.shape
        if p.get("axis") == "columns":
            df = df.loc[:, df.isna().mean() <= th]
        else:
            df = df[df.isna().mean(axis=1) <= th]
        return (
            df,
            f"Dropped {n0 - len(df)} rows, {c0 - df.shape[1]} columns (more than {th*100:.0f}% missing)",
        )
    if op == "duplicates":
        d = df.duplicated(keep="first")
        if p.get("mode", "remove") == "remove":
            return df[~d], f"Removed {int(d.sum())} duplicate rows"
        v = str(p.get("value", "mean"))
        for c in cols or nc:
            isn = c in allnum
            if v == "mode" or (not isn and v in ("mean", "median")):
                f = mode_of(df[c])
            elif v in ("mean", "median"):
                f = getattr(df[c], v)()
            else:
                f = float(v) if isn else v
            if isn:
                df[c] = df[c].astype(float)
            df.loc[d, c] = f
        return df, f"Replaced values in {int(d.sum())} duplicate rows with {v}"
    if op == "combine":
        if len(cols) < 2:
            raise ValueError("select at least two columns")
        name = p.get("name") or "_".join(cols)
        h = p.get("how", "concat")
        if h == "concat":
            df[name] = (
                df[cols]
                .astype(str)
                .apply(lambda r: str(p.get("sep", "_")).join(r), axis=1)
            )
        elif h == "sum":
            df[name] = df[cols].sum(axis=1)
        elif h == "mean":
            df[name] = df[cols].mean(axis=1)
        elif h == "product":
            df[name] = df[cols].prod(axis=1)
        else:
            df[name] = df[cols[0]] - df[cols[1:]].sum(axis=1)
        return df, f"Created '{name}' from {', '.join(cols)} ({h})"
    if op == "drop_columns":
        return df.drop(columns=cols), f"Dropped {len(cols)} columns"
    if op == "outliers":
        m, a = p.get("method", "iqr"), p.get("action", "clip")
        f = float(p.get("factor") or (1.5 if m == "iqr" else 3))
        mask = pd.Series(False, index=df.index)
        for c in nc:
            s = df[c]
            if m == "iqr":
                q1, q3 = s.quantile([0.25, 0.75])
                lo, hi = q1 - f * (q3 - q1), q3 + f * (q3 - q1)
            else:
                lo, hi = s.mean() - f * s.std(), s.mean() + f * s.std()
            mask |= (s < lo) | (s > hi)
            if a == "clip":
                df[c] = s.clip(lo, hi)
        if a == "remove":
            df = df[~mask]
        return df, f"{int(mask.sum())} rows contained outliers ({m}); action: {a}"
    if op == "scale":
        sc = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }[p.get("method", "standard")]()
        df[nc] = sc.fit_transform(df[nc])
        return df, f"Scaled {len(nc)} columns ({p.get('method','standard')})"
    if op == "transform":
        fn = p.get("fn", "log1p")
        for c in nc:
            df[c] = (
                np.log1p(df[c].clip(lower=0))
                if fn == "log1p"
                else np.sqrt(df[c].clip(lower=0)) if fn == "sqrt" else df[c] ** 2
            )
        return df, f"Applied {fn} to {len(nc)} columns"
    if op == "encode":
        cat = [c for c in (cols or df.columns) if c not in allnum]
        if p.get("method", "onehot") == "onehot":
            df = pd.get_dummies(df, columns=cat, dtype=int)
        else:
            for c in cat:
                df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        return (
            df,
            f"Encoded {len(cat)} categorical columns ({p.get('method','onehot')})",
        )
    if op == "bin":
        df[nc] = KBinsDiscretizer(
            int(p.get("bins", 5)), encode="ordinal", strategy="quantile"
        ).fit_transform(df[nc].fillna(df[nc].median()))
        return df, f"Binned {len(nc)} columns into {p.get('bins',5)} quantile bins"
    if op == "var_threshold":
        drop = [c for c in nc if df[c].var() < float(p.get("threshold", 0.01))]
        return df.drop(columns=drop), f"Dropped low-variance columns: {drop or 'none'}"
    if op == "corr_filter":
        cm = df[nc].corr().abs()
        up = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool))
        drop = [
            c for c in up.columns if (up[c] > float(p.get("threshold", 0.95))).any()
        ]
        return (
            df.drop(columns=drop),
            f"Dropped highly correlated columns: {drop or 'none'}",
        )
    if op == "reduce":
        keep = [c for c in cols if c in df.columns]
        X = df.drop(columns=keep).select_dtypes("number")
        if X.shape[1] < 2:
            raise ValueError("need at least two numeric columns outside the kept ones")
        m = p.get("method", "pca")
        n = min(int(p.get("n", 2)), X.shape[1], 3 if m == "tsne" else 99)
        Xs = StandardScaler().fit_transform(X.fillna(X.median()))
        if m == "pca":
            mod = PCA(n).fit(Xs)
            Z = mod.transform(Xs)
            note = f"{mod.explained_variance_ratio_.sum()*100:.1f}% variance kept"
        elif m == "svd":
            mod = TruncatedSVD(n, random_state=42).fit(Xs)
            Z = mod.transform(Xs)
            note = f"{mod.explained_variance_ratio_.sum()*100:.1f}% variance kept"
        else:
            Z = TSNE(n, random_state=42, perplexity=min(30, len(Xs) - 1)).fit_transform(
                Xs
            )
            note = "t-SNE is for visualising, not for predicting new rows"
        out = pd.concat(
            [
                df.drop(columns=X.columns).reset_index(drop=True),
                pd.DataFrame(Z, columns=[f"{m}_{i+1}" for i in range(n)]),
            ],
            axis=1,
        )
        return out, f"{m.upper()}: {X.shape[1]} columns reduced to {n} ({note})"
    raise ValueError("unknown operation")


def register(app, DATA, info):
    HIST, LOG, ORIG = {}, {}, {}

    @app.post("/api/fe/apply")
    def fe_apply(r: FEReq):
        i = r.dataset_id
        df = DATA[i]
        try:
            new, msg = run_op(df.copy(), r.op, r.cols, r.params)
        except Exception as e:
            raise HTTPException(400, f"{r.op} failed: {e}")
        ORIG.setdefault(i, df)
        HIST.setdefault(i, []).append(df)
        HIST[i] = HIST[i][-5:]
        LOG.setdefault(i, []).append(msg)
        DATA[i] = new.reset_index(drop=True)
        return {**info(i), "log": LOG[i]}

    @app.post("/api/fe/undo/{i}")
    def fe_undo(i: str):
        if HIST.get(i):
            DATA[i] = HIST[i].pop()
            LOG[i].pop()
        return {**info(i), "log": LOG.get(i, [])}

    @app.post("/api/fe/reset/{i}")
    def fe_reset(i: str):
        if i in ORIG:
            DATA[i] = ORIG[i]
            HIST[i] = []
            LOG[i] = []
        return {**info(i), "log": []}

    @app.get("/api/fe/compare/{i}")
    def fe_compare(i: str):
        old = ORIG.get(i, DATA[i])
        new = DATA[i]
        f = lambda v: None if pd.isna(v) else round(float(v), 3)
        isn = pd.api.types.is_numeric_dtype

        def ov(d):
            n = len(d.select_dtypes("number").columns)
            return {
                "rows": len(d),
                "cols": d.shape[1],
                "missing": int(d.isna().sum().sum()),
                "duplicates": int(d.duplicated().sum()),
                "numeric": n,
                "categorical": d.shape[1] - n,
            }

        common = [
            c for c in old.columns if c in new.columns and isn(old[c]) and isn(new[c])
        ]
        stats, hist = [], {}
        for c in common:
            a, b = old[c].dropna(), new[c].dropna()
            stats.append(
                {
                    "column": c,
                    "mean": [f(a.mean()), f(b.mean())],
                    "std": [f(a.std()), f(b.std())],
                    "skew": [f(a.skew()), f(b.skew())],
                    "missing": [int(old[c].isna().sum()), int(new[c].isna().sum())],
                }
            )
            if len(hist) < 15 and len(a) and len(b):
                lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
                e = np.linspace(lo, hi if hi > lo else lo + 1, 16)
                ha, hb = np.histogram(a, e)[0], np.histogram(b, e)[0]
                hist[c] = [
                    {
                        "x": round(float((e[j] + e[j + 1]) / 2), 3),
                        "before": int(ha[j]),
                        "after": int(hb[j]),
                    }
                    for j in range(15)
                ]
        cc = common[:12]
        corr = None
        if len(cc) >= 2:
            corr = {
                "cols": cc,
                "delta": (new[cc].corr() - old[cc].corr())
                .round(2)
                .fillna(0)
                .values.tolist(),
            }
        return {
            "before": ov(old),
            "after": ov(new),
            "steps": LOG.get(i, []),
            "stats": stats,
            "hist": hist,
            "corr": corr,
            "added": [
                {"name": c, "dtype": str(new[c].dtype)}
                for c in new.columns
                if c not in old.columns
            ],
            "removed": [c for c in old.columns if c not in new.columns],
            "dtype_changed": [
                {"name": c, "from": str(old[c].dtype), "to": str(new[c].dtype)}
                for c in old.columns
                if c in new.columns and str(old[c].dtype) != str(new[c].dtype)
            ],
        }
