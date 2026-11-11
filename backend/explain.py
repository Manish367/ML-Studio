import importlib.util
import io
import re

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sklearn.inspection import permutation_importance


class GlobalReq(BaseModel):
    model_id: str


class LocalReq(BaseModel):
    model_id: str
    row: dict


HAS_SHAP = (
    importlib.util.find_spec("shap") is not None
)  # SHAP is optional; it is only imported when used
TREES = {
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingRegressor",
    "GradientBoostingClassifier",
}
LINEAR = {"LinearRegression", "Ridge", "Lasso", "LogisticRegression"}


def _owners(pre, cat):
    """The original column behind every column the preprocessor produces."""
    out = []
    for nm in pre.get_feature_names_out():
        base = nm.split("__", 1)[1]
        if nm.startswith("n__"):
            out.append(base)
            continue
        out.append(
            next(
                (
                    c
                    for c in sorted(cat, key=len, reverse=True)
                    if base == c or base.startswith(c + "_")
                ),
                base,
            )
        )
    return out


def _shap(m, rows):
    """SHAP values for `rows`, summed back to the original columns.
    Returns (array rows x columns x outputs, column names, base value per output, units).
    """
    import shap

    pipe, le = m["pipe"], m["le"]
    pre, mod = pipe.named_steps["pre"], pipe.named_steps["m"]
    clf, name = le is not None, type(mod).__name__
    tf = lambda d: np.asarray(pre.transform(d), dtype=float)
    bg = m["Xte"] if len(m["Xte"]) <= 100 else m["Xte"].sample(100, random_state=42)
    Xb, Xr = tf(bg), tf(rows)
    if name in TREES:
        ex = shap.TreeExplainer(mod)
        sv = ex.shap_values(Xr)
    elif name in LINEAR:
        ex = shap.LinearExplainer(mod, Xb)
        sv = ex.shap_values(Xr)
    else:
        ex = shap.KernelExplainer(
            mod.predict_proba if clf else mod.predict,
            shap.kmeans(Xb, 10) if len(Xb) > 10 else Xb,
        )
        sv = ex.shap_values(Xr, nsamples=100, silent=True)
    ev = np.ravel(ex.expected_value).astype(float)
    a = (
        np.stack([np.asarray(x) for x in sv], axis=-1)
        if isinstance(sv, list)
        else np.asarray(sv)
    )
    units = "predicted value"
    if clf:
        units = (
            "log-odds"
            if name in ("GradientBoostingClassifier", "LogisticRegression")
            else "probability"
        )
        if a.ndim == 2:  # a binary model that only explains class 1
            a, ev = np.stack([-a, a], axis=-1), np.array([-ev[0], ev[0]])
    elif a.ndim == 2:
        a = a[..., None]
    owners = _owners(pre, m["cat"])
    cols = list(dict.fromkeys(owners))
    idx = {c: k for k, c in enumerate(cols)}
    G = np.zeros((a.shape[0], len(cols), a.shape[2]))
    for j, o in enumerate(owners):
        G[:, idx[o], :] += a[:, j, :]
    return G, cols, ev, units


def register_explain(app, MODELS):
    def get(mid):
        m = MODELS.get(mid)
        if not m:
            raise HTTPException(404, "Train this model first")
        return m

    @app.post("/api/explain/global")
    def explain_global(r: GlobalReq):
        """Permutation importance: score drop when one column is shuffled."""
        m = get(r.model_id)
        if m.get("Xte") is None:
            raise HTTPException(
                400, "No test data is saved for this model. Please train it again."
            )
        X, y = m["Xte"], m["yte"]
        if len(X) > 1500:
            idx = np.random.RandomState(42).choice(len(X), 1500, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]
        scoring = "accuracy" if m["le"] is not None else "r2"
        try:
            pi = permutation_importance(
                m["pipe"], X, y, scoring=scoring, n_repeats=5, random_state=42, n_jobs=1
            )
        except Exception as e:
            raise HTTPException(400, f"Could not compute importance: {e}")
        order = np.argsort(pi.importances_mean)[::-1][:15]
        return {
            "metric": "accuracy" if scoring == "accuracy" else "R²",
            "rows": len(X),
            "items": [
                {
                    "feature": str(X.columns[j]),
                    "importance": round(float(pi.importances_mean[j]), 4),
                    "std": round(float(pi.importances_std[j]), 4),
                }
                for j in order
            ],
        }

    @app.post("/api/explain/local")
    def explain_local(r: LocalReq):
        """Occlusion: replace one input at a time with a typical value."""
        m = get(r.model_id)
        feats = m.get("feats") or []
        if not feats:
            raise HTTPException(
                400,
                "No feature information is saved for this model. Please train it again.",
            )
        base = {f["name"]: f["default"] for f in feats}
        names = [f["name"] for f in feats]
        row = {k: (float(v) if k in m["num"] else str(v)) for k, v in r.row.items()}
        df = pd.DataFrame(
            [row] + [dict(row, **{n: base[n]}) for n in names] + [dict(base)]
        )
        pipe, le = m["pipe"], m["le"]
        if le is not None and hasattr(pipe, "predict_proba"):
            P = pipe.predict_proba(df)
            c = int(np.argmax(P[0]))
            vals = P[:, c]
            pred, measure = str(le.classes_[c]), f"probability of '{le.classes_[c]}'"
        elif le is not None:
            p = pipe.predict(df)
            vals = (p == p[0]).astype(float)
            pred, measure = (
                str(le.inverse_transform([int(p[0])])[0]),
                "match with the predicted class",
            )
        else:
            vals = np.asarray(pipe.predict(df), dtype=float)
            pred, measure = float(vals[0]), "predicted value"
        items = [
            {
                "feature": n,
                "value": row.get(n),
                "contribution": round(float(vals[0] - vals[i + 1]), 4),
            }
            for i, n in enumerate(names)
        ]
        items.sort(key=lambda d: -abs(d["contribution"]))
        return {
            "prediction": pred,
            "measure": measure,
            "full": round(float(vals[0]), 4),
            "baseline": round(float(vals[-1]), 4),
            "items": items[:12],
        }

    @app.get("/api/explain/capabilities")
    def capabilities():
        return {"shap": HAS_SHAP}

    @app.post("/api/explain/shap/local")
    def shap_local(r: LocalReq):
        if not HAS_SHAP:
            raise HTTPException(400, "SHAP is not installed on this server.")
        m = get(r.model_id)
        if not m.get("feats") or m.get("Xte") is None:
            raise HTTPException(
                400, "No data is saved for this model. Please train it again."
            )
        row = {k: (float(v) if k in m["num"] else str(v)) for k, v in r.row.items()}
        df = pd.DataFrame([row])
        clf = m["le"] is not None
        c = int(m["pipe"].predict(df)[0]) if clf else 0
        try:
            G, cols, ev, units = _shap(m, df)
        except Exception as e:
            raise HTTPException(400, f"SHAP could not explain this model: {e}")
        contrib = G[0, :, c]
        base = float(ev[c])
        items = [
            {"feature": n, "value": row.get(n), "contribution": round(float(v), 4)}
            for n, v in zip(cols, contrib)
        ]
        items.sort(key=lambda d: -abs(d["contribution"]))
        pred = str(m["le"].classes_[c]) if clf else float(m["pipe"].predict(df)[0])
        return {
            "prediction": pred,
            "measure": f"{units} of '{pred}'" if clf else "predicted value",
            "method": "shap",
            "full": round(base + float(contrib.sum()), 4),
            "baseline": round(base, 4),
            "items": items[:12],
        }

    @app.post("/api/explain/shap/global")
    def shap_global(r: GlobalReq):
        if not HAS_SHAP:
            raise HTTPException(400, "SHAP is not installed on this server.")
        m = get(r.model_id)
        if m.get("Xte") is None:
            raise HTTPException(
                400, "No test data is saved for this model. Please train it again."
            )
        cap = (
            200 if type(m["pipe"].named_steps["m"]).__name__ in (TREES | LINEAR) else 40
        )
        rows = (
            m["Xte"] if len(m["Xte"]) <= cap else m["Xte"].sample(cap, random_state=42)
        )
        try:
            G, cols, ev, units = _shap(m, rows)
        except Exception as e:
            raise HTTPException(400, f"SHAP could not explain this model: {e}")
        imp = np.abs(G).mean(axis=(0, 2))
        order = np.argsort(imp)[::-1][:15]
        return {
            "metric": f"mean |SHAP| in {units}",
            "rows": len(rows),
            "method": "shap",
            "items": [
                {"feature": cols[j], "importance": round(float(imp[j]), 4), "std": 0.0}
                for j in order
            ],
        }

    @app.get("/api/model/download")
    def model_download(model_id: str):
        m = MODELS.get(model_id)
        if not m:
            raise HTTPException(404, "Train this model first")
        buf = io.BytesIO()
        joblib.dump(
            {
                "pipe": m["pipe"],
                "classes": (
                    [str(c) for c in m["le"].classes_] if m["le"] is not None else None
                ),
                "numeric_columns": m["num"],
                "categorical_columns": m["cat"],
                "feature_defaults": m.get("feats"),
                "sklearn_version": sklearn.__version__,
            },
            buf,
        )
        fname = (
            re.sub(r"[^A-Za-z0-9_-]+", "_", model_id.split(":", 1)[-1]).strip("_")
            or "model"
        )
        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="ml-studio-{fname}.joblib"'
            },
        )
