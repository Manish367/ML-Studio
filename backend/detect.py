import re, warnings
import numpy as np, pandas as pd

ANOM = {"outlier", "anomaly", "anomalous", "novelty"}
KEYS = {
    "target",
    "label",
    "class",
    "outcome",
    "species",
    "diagnosis",
    "survived",
    "churn",
    "default",
    "fraud",
    "price",
    "salary",
    "charges",
    "medv",
}


def _cat(s):
    return pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s)


def _is_class(s, n):
    u = s.nunique(dropna=True)
    if _cat(s):
        return u <= max(20, 0.05 * n)
    whole = bool((s.dropna() % 1 == 0).all())
    return u <= 2 or (whole and u <= 20 and u <= 0.1 * n)


def detect(df):
    n, cols = len(df), list(df.columns)
    for c in cols:  # labelled anomaly benchmark
        if (
            any(t in ANOM for t in re.split(r"[^a-z0-9]+", str(c).lower()))
            and df[c].nunique(dropna=True) == 2
        ):
            r = df[c].value_counts(normalize=True).min() * 100
            return {
                "type": "Anomaly detection (labels available)",
                "task": "anomaly",
                "target": c,
                "confidence": "high",
                "reason": f"'{c}' flags outliers ({r:.1f}% of rows). Anomaly detectors don't need it; use it only to check results.",
                "use": f"Use the Unsupervised tab (Isolation Forest, LOF, One-Class SVM). '{c}' is left out of the features. To predict it instead, use the Supervised tab.",
            }

    for c in cols:  # time series
        s = df[c]
        hit = pd.api.types.is_datetime64_any_dtype(s)
        if not hit and _cat(s) and any(t in str(c).lower() for t in ("date", "time")):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hit = (
                        pd.to_datetime(s.head(50), errors="coerce").notna().mean() > 0.9
                    )
            except Exception:
                hit = False
        if hit:
            return {
                "type": "Time series",
                "task": "timeseries",
                "target": None,
                "confidence": "medium",
                "reason": f"Column '{c}' contains dates or times.",
                "use": "Not directly supported yet. Sort by date and use lag features with regression.",
            }
    for c in cols:  # free text
        s = df[c]
        if (
            _cat(s)
            and s.dropna().astype(str).str.len().mean() > 50
            and s.nunique() > 0.5 * n
        ):
            return {
                "type": "Text (NLP)",
                "task": "text",
                "target": None,
                "confidence": "medium",
                "reason": f"Column '{c}' holds long free text.",
                "use": "Needs text vectorisation first. Not supported by the built-in tabs yet.",
            }
    named = [
        c
        for c in cols
        if any(t in KEYS for t in re.split(r"[^a-z0-9]+", str(c).lower()))
    ]
    t = named[-1] if named else cols[-1]
    s = df[t]
    u = int(s.nunique(dropna=True))
    if named:
        if _is_class(s, n):
            kind = "binary" if u == 2 else f"multiclass, {u} classes"
            return {
                "type": f"Classification ({kind})",
                "task": "classification",
                "target": t,
                "confidence": "high",
                "reason": f"'{t}' looks like a label with {u} distinct values.",
                "use": "Use the Supervised tab.",
            }
        if pd.api.types.is_numeric_dtype(s):
            return {
                "type": "Regression",
                "task": "regression",
                "target": t,
                "confidence": "high",
                "reason": f"'{t}' is a continuous number ({u} distinct values).",
                "use": "Use the Supervised tab.",
            }
    if not _cat(s) and _is_class(s, n) and u > 1 and not pd.api.types.is_float_dtype(s):
        return {
            "type": f"Classification ({u} classes)",
            "task": "classification",
            "target": t,
            "confidence": "medium",
            "reason": f"The last column '{t}' has only {u} distinct values, so it may be a label.",
            "use": f"Use the Supervised tab, or Unsupervised if '{t}' is not a label.",
        }
    return {
        "type": "Clustering (no target found)",
        "task": "clustering",
        "target": None,
        "confidence": "medium",
        "reason": "No column looks like a label or outcome to predict, so this is best explored for hidden groups.",
        "use": f"Use the Unsupervised tab. To predict '{t}' instead, choose Regression on the Supervised tab.",
    }
