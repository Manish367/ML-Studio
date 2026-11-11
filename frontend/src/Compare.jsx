import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
const G = ({ title, children }) => (
  <div className="glass p-4">
    {title && <h6 className="fw-semibold mb-3">{title}</h6>}
    {children}
  </div>
);
const ax = { stroke: "#a5b4fc", fontSize: 11 };
const OV = [
  ["rows", "Rows", 0],
  ["cols", "Columns", 0],
  ["missing", "Missing cells", 1],
  ["duplicates", "Duplicate rows", 1],
  ["numeric", "Numeric columns", 0],
  ["categorical", "Text columns", 0],
];
const Pair = ({ v }) => {
  const ch = v[0] !== v[1];
  return (
    <span style={{ color: ch ? "#facc15" : "inherit" }}>
      {v[0] ?? "-"} → {v[1] ?? "-"}
    </span>
  );
};

export default function Compare({ ds, run }) {
  const [d, setD] = useState(null);
  const [col, setCol] = useState("");
  useEffect(() => {
    run(async () => {
      const r = await fetch(
        (import.meta.env.VITE_API_URL || "") + "/api/fe/compare/" + ds.id,
      );
      if (!r.ok) throw new Error("Could not load comparison");
      const j = await r.json();
      setD(j);
      setCol((c) => (c && j.hist[c] ? c : Object.keys(j.hist)[0] || ""));
    });
  }, [ds.id, (ds.log || []).length, ds.cols.length]);
  if (!d) return null;
  const same = !d.steps.length;
  return (
    <div className="d-grid gap-3">
      <G title="What changed">
        {same && (
          <div className="opacity-75 mb-3">
            No feature engineering steps applied yet, so before and after are
            identical.
          </div>
        )}
        <div className="row g-3">
          {OV.map(([k, label, lowerBetter]) => {
            const b = d.before[k],
              a = d.after[k],
              diff = a - b;
            const color =
              diff === 0
                ? "#a5b4fc"
                : lowerBetter
                  ? diff < 0
                    ? "#34d399"
                    : "#f87171"
                  : "#facc15";
            return (
              <div key={k} className="col-6 col-md-4 col-xl-2">
                <div className="small opacity-75">{label}</div>
                <div className="fs-5 fw-bold">
                  {b} → {a}
                </div>
                <div className="small" style={{ color }}>
                  {diff > 0 ? "+" : ""}
                  {diff}
                </div>
              </div>
            );
          })}
        </div>
        {(d.added.length > 0 ||
          d.removed.length > 0 ||
          d.dtype_changed.length > 0) && (
          <div className="mt-3 small">
            {d.added.length > 0 && (
              <div className="mb-1">
                <b style={{ color: "#34d399" }}>Added:</b>{" "}
                {d.added.map((c) => `${c.name} (${c.dtype})`).join(", ")}
              </div>
            )}
            {d.removed.length > 0 && (
              <div className="mb-1">
                <b style={{ color: "#f87171" }}>Removed:</b>{" "}
                {d.removed.join(", ")}
              </div>
            )}
            {d.dtype_changed.length > 0 && (
              <div>
                <b style={{ color: "#facc15" }}>Type changed:</b>{" "}
                {d.dtype_changed
                  .map((c) => `${c.name} (${c.from} → ${c.to})`)
                  .join(", ")}
              </div>
            )}
          </div>
        )}
      </G>
      {d.steps.length > 0 && (
        <G title="Steps applied">
          <ol className="mb-0 small">
            {d.steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>
        </G>
      )}
      {col && (
        <G title="Distribution before vs after">
          <select
            className="form-select w-auto mb-3"
            value={col}
            onChange={(e) => setCol(e.target.value)}
          >
            {Object.keys(d.hist).map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <div style={{ height: 260 }}>
            <ResponsiveContainer>
              <BarChart data={d.hist[col]}>
                <CartesianGrid stroke="#ffffff18" />
                <XAxis dataKey="x" {...ax} />
                <YAxis {...ax} width={32} />
                <Tooltip
                  contentStyle={{
                    background: "#0b1020",
                    border: "1px solid #ffffff33",
                    borderRadius: 10,
                  }}
                />
                <Legend />
                <Bar dataKey="before" fill="#a78bfa" radius={[4, 4, 0, 0]} />
                <Bar dataKey="after" fill="#22d3ee" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </G>
      )}
      <G title="Column statistics (before → after)">
        <div style={{ overflow: "auto", maxHeight: 380 }}>
          <table className="table table-sm">
            <thead>
              <tr>
                <th>Column</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Skew</th>
                <th>Missing</th>
              </tr>
            </thead>
            <tbody>
              {d.stats.map((r) => (
                <tr key={r.column}>
                  <td>{r.column}</td>
                  <td>
                    <Pair v={r.mean} />
                  </td>
                  <td>
                    <Pair v={r.std} />
                  </td>
                  <td>
                    <Pair v={r.skew} />
                  </td>
                  <td>
                    <Pair v={r.missing} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small opacity-50">
          Values that changed are highlighted in yellow. Skew closer to 0 means
          a more symmetric distribution.
        </div>
      </G>
      {d.corr && (
        <G title="Correlation change (after minus before)">
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", fontSize: 10 }}>
              <tbody>
                {d.corr.delta.map((row, i) => (
                  <tr key={i}>
                    <td className="pe-2 text-end opacity-75">
                      {d.corr.cols[i]}
                    </td>
                    {row.map((v, j) => (
                      <td
                        key={j}
                        title={`${d.corr.cols[i]} × ${d.corr.cols[j]}: ${v}`}
                        style={{
                          width: 34,
                          height: 28,
                          textAlign: "center",
                          background:
                            v >= 0
                              ? `rgba(34,211,238,${Math.min(1, Math.abs(v) * 2)})`
                              : `rgba(244,114,182,${Math.min(1, Math.abs(v) * 2)})`,
                        }}
                      >
                        {v.toFixed(1)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="small opacity-50 mt-2">
            Cyan = relationship got stronger, pink = weaker.
          </div>
        </G>
      )}
    </div>
  );
}
