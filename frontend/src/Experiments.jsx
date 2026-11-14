import { useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { useNarrow } from "./useNarrow.js";
const G = ({ title, children }) => (
  <div className="glass p-4">
    {title && <h6 className="fw-semibold mb-3">{title}</h6>}
    {children}
  </div>
);
const save = (name, text, type) => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
};
const cell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
const COLS = [
  "time",
  "kind",
  "task",
  "target",
  "summary",
  "settings",
  "scoreLabel",
  "score",
  "headline",
  "data",
  "detected",
  "steps",
];

export default function Experiments({ exps, setExps }) {
  const labels = [
    ...new Set(
      exps
        .filter((e) => e.score != null && e.scoreLabel)
        .map((e) => e.scoreLabel),
    ),
  ];
  const [lab, setLab] = useState("");
  const narrow = useNarrow();
  const cur = labels.includes(lab) ? lab : labels[0];
  const chart = exps
    .map((e, i) => ({
      run: `#${i + 1} ${e.kind}`,
      score: e.score,
      label: e.scoreLabel,
    }))
    .filter((d) => d.score != null && d.label === cur);
  const json = () =>
    save(
      "ml-studio-experiments.json",
      JSON.stringify(
        exps.map(({ key, ...e }) => e),
        null,
        2,
      ),
      "application/json",
    );
  const csv = () =>
    save(
      "ml-studio-experiments.csv",
      [
        COLS.join(","),
        ...exps.map((e) =>
          COLS.map((c) =>
            cell(Array.isArray(e[c]) ? e[c].join(" | ") : e[c]),
          ).join(","),
        ),
      ].join("\n"),
      "text/csv",
    );
  if (!exps.length)
    return (
      <G>
        Nothing here yet. Every training, tuning, AutoML and clustering run you
        do in this visit is listed here so you can compare them. The list is
        cleared when you reload the page, so export it first if you want to keep
        it.
      </G>
    );
  return (
    <div className="d-grid gap-3">
      <G
        title={`${exps.length} run${exps.length === 1 ? "" : "s"} in this session`}
      >
        <div className="d-flex flex-wrap gap-2 align-items-center">
          <button className="btn btn-glow btn-sm" onClick={json}>
            Export JSON
          </button>
          <button className="btn btn-glow btn-sm" onClick={csv}>
            Export CSV
          </button>
          <button
            className="btn btn-outline-light btn-sm"
            onClick={() => setExps([])}
          >
            Clear all
          </button>
          <span className="small opacity-50">
            Cleared when you reload the page. Export to keep it.
          </span>
        </div>
      </G>
      {chart.length > 0 && (
        <G title="Compare runs">
          {labels.length > 1 && (
            <select
              className="form-select w-auto mb-2"
              value={cur}
              onChange={(e) => setLab(e.target.value)}
            >
              {labels.map((l) => (
                <option key={l}>{l}</option>
              ))}
            </select>
          )}
          <div style={{ height: 40 + chart.length * 30 }}>
            <ResponsiveContainer>
              <BarChart
                data={chart}
                layout="vertical"
                margin={{ left: 10, right: 20 }}
              >
                <CartesianGrid stroke="#ffffff18" />
                <XAxis
                  type="number"
                  stroke="#a5b4fc"
                  fontSize={11}
                  domain={cur === 'accuracy' ? [0, 1] : ['auto', 'auto']}
                />
                <YAxis
                  type="category"
                  dataKey="run"
                  stroke="#a5b4fc"
                  fontSize={11}
                  width={narrow ? 85 : 110}
                />
                <Tooltip
                  contentStyle={{
                    background: "#0b1020",
                    border: "1px solid #ffffff33",
                    borderRadius: 10,
                  }}
                />
                <Bar dataKey="score" fill="#22d3ee" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="small opacity-50">
            Only runs measured with the same score ({cur}) share a chart.
            Different datasets and settings are not directly comparable.
          </div>
        </G>
      )}
      <G>
        <div style={{ overflow: "auto" }}>
          <table className="table table-sm">
            <thead>
              <tr>
                <th>#</th>
                <th>Time</th>
                <th>Type</th>
                <th>Data</th>
                <th>Setup</th>
                <th>Result</th>
                <th>Steps</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {exps.map((e, i) => (
                <tr key={e.key}>
                  <td>{i + 1}</td>
                  <td>{e.time}</td>
                  <td>
                    {e.kind}
                    {e.target ? (
                      <div className="small opacity-50">{e.target}</div>
                    ) : null}
                  </td>
                  <td className="small">
                    {e.data}
                    <div className="opacity-50">{e.detected}</div>
                  </td>
                  <td className="small">
                    {e.summary}
                    <div className="opacity-50">{e.settings}</div>
                  </td>
                  <td className="small">
                    {e.headline}
                    {e.score != null && (
                      <div>
                        {e.scoreLabel}: <b>{e.score}</b>
                      </div>
                    )}
                  </td>
                  <td className="small" title={e.steps.join("\n")}>
                    {e.steps.length
                      ? `${e.steps.length} step${e.steps.length === 1 ? "" : "s"}`
                      : "none"}
                  </td>
                  <td>
                    <span
                      style={{ cursor: "pointer" }}
                      onClick={() =>
                        setExps(exps.filter((x) => x.key !== e.key))
                      }
                    >
                      ✕
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </G>
    </div>
  );
}
