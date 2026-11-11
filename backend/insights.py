import numpy as np, pandas as pd
from detect import detect


def register_insights(app, DATA):
    @app.get("/api/insights/{i}")
    def insights(i: str):
        df = DATA[i]
        n = len(df)
        out = []
        add = lambda level, title, detail, fix=None: out.append(
            {"level": level, "title": title, "detail": detail, "fix": fix}
        )
        num = df.select_dtypes("number")
        nc = list(num.columns)
        if n < 100:
            add(
                "warn",
                "Very small dataset",
                f"Only {n} rows. Scores can swing a lot with so little data, so treat results with caution.",
            )
        elif nc and n < 30 * len(nc):
            add(
                "info",
                "Few rows for the number of columns",
                f"{n} rows for {len(nc)} numeric columns can lead to overfitting.",
                "Remove unneeded columns or use simpler models.",
            )
        miss = df.isna().mean()
        for c in miss[miss > 0.4].index:
            add(
                "warn",
                f"'{c}' is mostly empty",
                f"{miss[c]*100:.0f}% of its values are missing.",
                "Engineer: drop the column, or fill it if it matters.",
            )
        some = miss[(miss > 0) & (miss <= 0.4)]
        if len(some):
            add(
                "info",
                f"{len(some)} column(s) have missing values",
                ", ".join(f"{c} ({v*100:.1f}%)" for c, v in some.head(6).items()),
                "Engineer: fill missing values (median for numbers, mode for text).",
            )
        dup = int(df.duplicated().sum())
        if dup:
            add(
                "warn" if dup > 0.05 * n else "info",
                f"{dup} duplicate rows",
                f"{dup / n * 100:.1f}% of the rows repeat exactly.",
                "Engineer: remove duplicates.",
            )
        const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        if const:
            add(
                "warn",
                "Columns with a single value",
                ", ".join(const[:6]) + " never change, so they carry no information.",
                "Engineer: drop these columns.",
            )
        ids = [
            c
            for c in df.columns
            if n > 20
            and df[c].nunique() == n
            and (
                pd.api.types.is_integer_dtype(df[c])
                or not pd.api.types.is_numeric_dtype(df[c])
            )
        ]
        if ids:
            add(
                "info",
                "Columns that look like IDs",
                ", ".join(ids[:6])
                + " has a different value in every row. IDs make models memorise instead of learn.",
                "Engineer: drop these columns.",
            )
        txt = [
            c
            for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and c not in ids
            and n > 20
            and df[c].nunique() > 0.5 * n
        ]
        if txt:
            add(
                "info",
                "Text columns with almost all-unique values",
                ", ".join(txt[:6]) + " will be one-hot encoded into many columns.",
                "Engineer: drop them or combine into something simpler.",
            )
        if nc:
            sk = [
                (c, float(num[c].skew()))
                for c in nc
                if num[c].nunique() > 10 and abs(num[c].skew()) > 1
            ]
            if sk:
                add(
                    "info",
                    f"{len(sk)} skewed column(s)",
                    ", ".join(f"{c} ({v:+.1f})" for c, v in sk[:5]),
                    "Engineer: transform with log1p to even out the spread.",
                )
            outs = []
            for c in nc:
                s = num[c].dropna()
                if s.nunique() > 10 and len(s):
                    q1, q3 = s.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    frac = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())
                    if frac > 0.05:
                        outs.append((c, frac))
            if outs:
                add(
                    "info",
                    f"{len(outs)} column(s) with many outliers",
                    ", ".join(f"{c} ({f*100:.0f}%)" for c, f in outs[:5]),
                    "Engineer: clip or remove outliers, or use tree-based models.",
                )
            if len(nc) > 1:
                cm = num[nc[:30]].corr().abs()
                pairs = []
                for a in range(len(cm.columns)):
                    for b in range(a + 1, len(cm.columns)):
                        if cm.iloc[a, b] > 0.9:
                            pairs.append(
                                (cm.columns[a], cm.columns[b], float(cm.iloc[a, b]))
                            )
                if pairs:
                    add(
                        "info",
                        f"{len(pairs)} highly correlated pair(s)",
                        ", ".join(f"{a} and {b} ({v:.2f})" for a, b, v in pairs[:4]),
                        "Engineer: remove correlated features.",
                    )
            rng_ = num.max() - num.min()
            rng_ = rng_[rng_ > 0]
            if len(rng_) > 1 and rng_.max() / rng_.min() > 100:
                add(
                    "info",
                    "Columns on very different scales",
                    f"The biggest range is over {rng_.max() / rng_.min():.0f} times the smallest.",
                    "Models in this app scale features automatically. For clustering, scale in Engineer.",
                )
        d = detect(df)
        t = d.get("target")
        task = d.get("task")
        if t in df.columns and task in ("classification", "anomaly"):
            vc = df[t].value_counts()
            if len(vc) > 1:
                share = vc.min() / vc.sum()
                if share < 0.1:
                    add(
                        "warn",
                        "Imbalanced target",
                        f"The rarest class of '{t}' is only {share*100:.1f}% of rows. Plain accuracy will look great even for a useless model.",
                        "Supervised: tick 'Balance classes' and judge by recall, balanced accuracy and PR-AUC.",
                    )
                elif share < 0.3:
                    add(
                        "info",
                        "Somewhat imbalanced target",
                        f"The rarest class of '{t}' is {share*100:.0f}% of rows.",
                        "Supervised: consider 'Balance classes'.",
                    )
                if vc.min() < 5:
                    add(
                        "warn",
                        "A class has fewer than 5 rows",
                        f"'{vc.idxmin()}' has {int(vc.min())} row(s), too few to learn or to evaluate.",
                        "Remove or merge that class.",
                    )
            codes = pd.factorize(df[t])[0]
            leaks = (
                [
                    c
                    for c in nc
                    if c != t
                    and num[c].nunique() > 1
                    and abs(np.corrcoef(num[c].fillna(num[c].median()), codes)[0, 1])
                    > 0.95
                ]
                if len(vc) == 2
                else []
            )
            if leaks:
                add(
                    "warn",
                    "Possible leakage",
                    ", ".join(leaks[:4])
                    + f" is almost perfectly linked to '{t}'. If it is only known after the outcome, drop it.",
                    "Engineer: drop these columns.",
                )
        elif t in df.columns and task == "regression":
            leaks = [
                c
                for c in nc
                if c != t
                and num[c].nunique() > 1
                and abs(num[[c, t]].corr().iloc[0, 1]) > 0.98
            ]
            if leaks:
                add(
                    "warn",
                    "Possible leakage",
                    ", ".join(leaks[:4]) + f" is almost perfectly linked to '{t}'.",
                    "Engineer: drop these columns if they are not available at prediction time.",
                )
            if num[t].nunique() > 10 and abs(num[t].skew()) > 1.5:
                add(
                    "info",
                    f"Target '{t}' is skewed",
                    f"Skew {num[t].skew():+.1f}. Errors on the largest values will dominate.",
                    "Consider modelling a log-transformed target.",
                )
        if not out:
            add(
                "good",
                "No obvious problems found",
                "Missing values, duplicates, scale, skew and imbalance all look fine.",
            )
        order = {"warn": 0, "info": 1, "good": 2}
        out.sort(key=lambda x: order[x["level"]])
        return {"items": out[:14]}
