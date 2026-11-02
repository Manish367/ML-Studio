import { useState } from "react";
const G = ({ title, children }) => (
  <div className="glass p-4">
    {title && <h6 className="fw-semibold mb-3">{title}</h6>}
    {children}
  </div>
);
const post = async (u, b) => {
  const r = await fetch("/api" + u, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(b || {}),
  });
  if (!r.ok) {
    let m = `Request failed (${r.status})`;
    try {
      m = (await r.json()).detail || m;
    } catch {}
    throw new Error(m);
  }
  return r.json();
};
const S = (k, o, d, l) => ({ k, t: "sel", o, d, l: l || k });
const N = (k, d, l) => ({ k, t: "num", d, l: l || k });
const T = (k, d, l) => ({ k, t: "txt", d, l: l || k });
const OPS = {
  fill_missing: {
    label: "Fill missing values",
    hint: "Columns to fill (empty = all)",
    p: [
      S(
        "strategy",
        ["mean", "median", "mode", "constant", "ffill", "knn"],
        "median",
      ),
      T("value", "0", "Constant value"),
    ],
  },
  drop_missing: {
    label: "Drop rows / columns with nulls",
    hint: "",
    p: [
      S("axis", ["rows", "columns"], "rows"),
      N("threshold", 30, "Max % missing allowed"),
    ],
  },
  duplicates: {
    label: "Handle duplicate rows",
    hint: "Columns to overwrite (replace mode only)",
    p: [
      S("mode", ["remove", "replace"], "remove"),
      T("value", "mean", "Replace with (mean, median, mode or a value)"),
    ],
  },
  combine: {
    label: "Combine columns",
    hint: "Pick two or more columns",
    p: [
      S("how", ["concat", "sum", "mean", "product", "difference"], "sum"),
      T("name", "", "New column name"),
      T("sep", "_", "Separator (concat)"),
    ],
  },
  drop_columns: { label: "Drop columns", hint: "Columns to remove", p: [] },
  outliers: {
    label: "Handle outliers / noise",
    hint: "Numeric columns (empty = all)",
    p: [
      S("method", ["iqr", "zscore"], "iqr"),
      S("action", ["clip", "remove"], "clip"),
      N("factor", 1.5, "Factor (IQR 1.5, z-score 3)"),
    ],
  },
  scale: {
    label: "Scale numeric columns",
    hint: "Numeric columns (empty = all)",
    p: [S("method", ["standard", "minmax", "robust"], "standard")],
  },
  transform: {
    label: "Transform (skew / noise)",
    hint: "Numeric columns (empty = all)",
    p: [S("fn", ["log1p", "sqrt", "square"], "log1p")],
  },
  encode: {
    label: "Encode categorical columns",
    hint: "Text columns (empty = all)",
    p: [S("method", ["onehot", "label"], "onehot")],
  },
  bin: {
    label: "Bin into groups",
    hint: "Numeric columns",
    p: [N("bins", 5, "Number of bins")],
  },
  var_threshold: {
    label: "Remove low-variance features",
    hint: "Numeric columns (empty = all)",
    p: [N("threshold", 0.01, "Minimum variance")],
  },
  corr_filter: {
    label: "Remove correlated features",
    hint: "Numeric columns (empty = all)",
    p: [N("threshold", 0.95, "Max correlation")],
  },
  reduce: {
    label: "Dimensionality reduction",
    hint: "Columns to keep untouched (e.g. the target)",
    p: [S("method", ["pca", "svd", "tsne"], "pca"), N("n", 2, "Components")],
  },
};
export default function Engineer({ ds, load, run }) {
  const [op, setOp] = useState("fill_missing");
  const [cols, setCols] = useState([]);
  const [vals, setVals] = useState({});
  const cfg = OPS[op];
  const params = {};
  cfg.p.forEach((x) => (params[x.k] = vals[op + x.k] ?? x.d));
  const call = (u, b) => run(async () => load(await post(u, b)));
  return (
    <div className="row g-3">
      <div className="col-lg-4">
        <G title="Feature engineering step">
          <select
            className="form-select mb-3"
            value={op}
            onChange={(e) => {
              setOp(e.target.value);
              setCols([]);
            }}
          >
            {Object.entries(OPS).map(([k, v]) => (
              <option key={k} value={k}>
                {v.label}
              </option>
            ))}
          </select>
          {cfg.p.map((x) => (
            <div key={x.k} className="mb-2">
              <label className="small opacity-75">{x.l}</label>
              {x.t === "sel" ? (
                <select
                  className="form-select"
                  value={params[x.k]}
                  onChange={(e) =>
                    setVals({ ...vals, [op + x.k]: e.target.value })
                  }
                >
                  {x.o.map((o) => (
                    <option key={o}>{o}</option>
                  ))}
                </select>
              ) : (
                <input
                  className="form-control"
                  type={x.t === "num" ? "number" : "text"}
                  step="any"
                  value={params[x.k]}
                  onChange={(e) =>
                    setVals({ ...vals, [op + x.k]: e.target.value })
                  }
                />
              )}
            </div>
          ))}
          <div className="small opacity-75 mt-2">
            {cfg.hint || "Columns not needed for this step"}
          </div>
          {cfg.hint && (
            <div>
              <span
                className="chip"
                onClick={() =>
                  setCols(ds.cols.filter((c) => c.numeric).map((c) => c.name))
                }
              >
                all numeric
              </span>
              <span className="chip" onClick={() => setCols([])}>
                clear
              </span>
            </div>
          )}
          {cfg.hint &&
            ds.cols.map((c) => (
              <span
                key={c.name}
                className={`chip ${cols.includes(c.name) ? "on" : ""}`}
                onClick={() =>
                  setCols(
                    cols.includes(c.name)
                      ? cols.filter((x) => x !== c.name)
                      : [...cols, c.name],
                  )
                }
              >
                {c.name}
                {c.missing ? ` · ${c.missing} null` : ""}
              </span>
            ))}
          <button
            className="btn btn-glow w-100 mt-3"
            onClick={() =>
              call("/fe/apply", { dataset_id: ds.id, op, cols, params })
            }
          >
            Apply step
          </button>
          <div className="d-flex gap-2 mt-2">
            <button
              className="btn btn-outline-light btn-sm flex-fill"
              onClick={() => call("/fe/undo/" + ds.id)}
            >
              Undo last
            </button>
            <button
              className="btn btn-outline-light btn-sm flex-fill"
              onClick={() => call("/fe/reset/" + ds.id)}
            >
              Reset all
            </button>
          </div>
        </G>
      </div>
      <div className="col-lg-8 d-grid gap-3">
        <G title="Applied steps">
          {(ds.log || []).length ? (
            <ol className="mb-0 small">
              {ds.log.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ol>
          ) : (
            <span className="opacity-75">
              No steps yet. Every step changes the dataset used by the other
              tabs.
            </span>
          )}
        </G>
        <G
          title={`Current data: ${ds.rows} rows, ${ds.cols.length} columns, ${ds.duplicates} duplicates`}
        >
          <div style={{ overflow: "auto", maxHeight: 380 }}>
            <table className="table table-sm">
              <thead>
                <tr>
                  {ds.cols.map((c) => (
                    <th key={c.name}>
                      {c.name}
                      <div className="small opacity-50 fw-normal">
                        {c.dtype} · {c.missing} null
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ds.preview.map((r, i) => (
                  <tr key={i}>
                    {ds.cols.map((c) => (
                      <td key={c.name}>{r[c.name]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </G>
      </div>
    </div>
  );
}
