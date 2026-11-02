import { useEffect, useState } from "react";
import Engineer from "./Engineer.jsx";
import Compare from "./Compare.jsx";
import Loader from "./Loader.jsx";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceLine,
} from "recharts";

const COL = [
  "#22d3ee",
  "#a78bfa",
  "#f472b6",
  "#facc15",
  "#34d399",
  "#fb923c",
  "#60a5fa",
  "#f87171",
];
const api = async (url, body, form) => {
  const r = await fetch(
    "/api" + url,
    body || form
      ? {
          method: "POST",
          ...(form
            ? { body: form }
            : {
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
              }),
        }
      : undefined,
  );
  if (!r.ok) {
    let m = `Request failed (${r.status})`;
    try {
      m = (await r.json()).detail || m;
    } catch {}
    throw new Error(m);
  }
  return r.json();
};
const G = ({ title, children, className = "" }) => (
  <div className={`glass p-4 ${className}`}>
    {title && <h6 className="fw-semibold mb-3">{title}</h6>}
    {children}
  </div>
);
const ax = { stroke: "#a5b4fc", fontSize: 11 };
const Gauge = ({ label, v }) => (
  <div className="text-center">
    <div
      className="mx-auto rounded-full grid place-items-center"
      style={{
        width: 92,
        height: 92,
        background: `conic-gradient(#22d3ee ${Math.max(0, v) * 360}deg, rgba(255,255,255,.12) 0)`,
      }}
    >
      <div
        className="rounded-full grid place-items-center fw-bold"
        style={{ width: 74, height: 74, background: "#10172e" }}
      >
        {(v * 100).toFixed(1)}
      </div>
    </div>
    <div className="small opacity-75 mt-1">{label}</div>
  </div>
);
const Chart = ({ h = 240, children }) => (
  <div style={{ height: h }}>
    <ResponsiveContainer>{children}</ResponsiveContainer>
  </div>
);
const Tip = (
  <Tooltip
    contentStyle={{
      background: "#0b1020",
      border: "1px solid #ffffff33",
      borderRadius: 10,
    }}
  />
);
const Corr = ({ c }) => (
  <div style={{ overflowX: "auto" }}>
    <table style={{ borderCollapse: "collapse", fontSize: 10 }}>
      <tbody>
        {c.z.map((row, i) => (
          <tr key={i}>
            <td className="pe-2 text-end opacity-75">{c.cols[i]}</td>
            {row.map((v, j) => (
              <td
                key={j}
                title={`${c.cols[i]} × ${c.cols[j]}: ${v}`}
                style={{
                  width: 34,
                  height: 28,
                  textAlign: "center",
                  background:
                    v >= 0
                      ? `rgba(34,211,238,${Math.abs(v)})`
                      : `rgba(244,114,182,${Math.abs(v)})`,
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
);
const Confusion = ({ m, classes }) => (
  <table className="table table-sm text-center mb-0">
    <thead>
      <tr>
        <th />
        {classes.map((c) => (
          <th key={c}>pred {c}</th>
        ))}
      </tr>
    </thead>
    <tbody>
      {m.map((r, i) => (
        <tr key={i}>
          <th>true {classes[i]}</th>
          {r.map((v, j) => (
            <td
              key={j}
              style={{
                background: `rgba(34,211,238,${Math.min(0.85, (v / Math.max(1, Math.max(...r))) * 0.85)})`,
                fontWeight: i === j ? 700 : 400,
              }}
            >
              {v}
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  </table>
);
const Chips = ({ items, sel, set }) =>
  items.map((a) => (
    <span
      key={a}
      className={`chip ${sel.includes(a) ? "on" : ""}`}
      onClick={() =>
        set(sel.includes(a) ? sel.filter((x) => x !== a) : [...sel, a])
      }
    >
      {a}
    </span>
  ));

export default function App() {
  const [tab, setTab] = useState("data");
  const [meta, setMeta] = useState(null);
  const [ds, setDs] = useState(null);
  const [eda, setEda] = useState(null);
  const [tr, setTr] = useState(null);
  const [cl, setCl] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  useEffect(() => {
    api("/algos")
      .then(setMeta)
      .catch(() => setErr("Backend not running. Start it on port 8000."));
  }, []);
  const run = async (f) => {
    setBusy(true);
    setErr("");
    try {
      await f();
    } catch (e) {
      setErr(e.message);
    }
    setBusy(false);
  };
  const load = (d) => {
    setDs(d);
    setEda(null);
    setTr(null);
    setCl(null);
  };
  const tabs = [
    ["data", "Dataset"],
    ["eda", "Explore"],
    ["engineer", "Engineer"],
    ["compare", "Before vs After"],
    ["train", "Supervised"],
    ["cluster", "Unsupervised"],
    ["predict", "Predict"],
  ];
  return (
    <div className="container py-4">
      <div className="d-flex flex-wrap flex-lg-nowrap align-items-center justify-content-between gap-3 mb-4">
        <div style={{ minWidth: 0 }}>
          <h2 className="fw-bold mb-0">ML Studio</h2>
          <div className="opacity-75 small">
            Upload a CSV, compare models, see what they learned.
          </div>
        </div>
        <div className="glass p-2 d-flex flex-wrap flex-lg-nowrap gap-1 ms-auto">
          {tabs.map(([k, l]) => (
            <div
              key={k}
              className={`tabbtn ${tab === k ? "on" : ""}`}
              onClick={() => setTab(k)}
            >
              {l}
            </div>
          ))}
        </div>
      </div>
      {err && <div className="alert alert-danger glass">{err}</div>}
      {busy && (
        <div className="progress mb-3" style={{ height: 4 }}>
          <div className="progress-bar progress-bar-striped progress-bar-animated w-100" />
        </div>
      )}
      {tab === "data" && <Data meta={meta} ds={ds} load={load} run={run} />}
      {tab !== "data" && !ds && (
        <G>Load a sample or upload a CSV on the Dataset tab first.</G>
      )}
      {ds && tab === "eda" && (
        <Eda ds={ds} eda={eda} setEda={setEda} run={run} />
      )}
      {ds && tab === "engineer" && <Engineer ds={ds} load={load} run={run} />}
      {ds && tab === "compare" && <Compare ds={ds} run={run} />}
      {ds && tab === "train" && (
        <Train
          ds={ds}
          meta={meta}
          tr={tr}
          setTr={setTr}
          run={run}
          busy={busy}
        />
      )}
      {ds && tab === "cluster" && (
        <Cluster
          ds={ds}
          meta={meta}
          cl={cl}
          setCl={setCl}
          run={run}
          busy={busy}
        />
      )}
      {ds && tab === "predict" && <Predict ds={ds} tr={tr} run={run} />}
    </div>
  );
}

function Data({ meta, ds, load, run }) {
  const pick = (g, t) =>
    (meta?.samples || []).filter((s) => s.group === g && (!t || s.task === t));
  const Group = ({ title, sub, items }) => (
    <div className="mb-3">
      <div className="fw-semibold small">{title}</div>
      <div className="small opacity-50 mb-1">{sub}</div>
      {items.map((s) => (
        <span
          key={s.id}
          className="chip"
          title={s.desc}
          onClick={() =>
            run(async () => load(await api("/sample/" + s.id, {})))
          }
        >
          {s.label}
        </span>
      ))}
    </div>
  );
  const d = ds?.detected;
  return (
    <div className="row g-3">
      <div className="col-lg-4">
        <G title="Start with data">
          <input
            type="file"
            accept=".csv"
            className="form-control mb-3"
            onChange={(e) => {
              const f = new FormData();
              f.append("file", e.target.files[0]);
              run(async () => load(await api("/upload", null, f)));
            }}
          />
          <div className="small opacity-75 mb-3">Or try an example dataset</div>
          <Group
            title="Supervised: classification"
            sub="Has a label column to predict (a category)"
            items={pick("supervised", "classification")}
          />
          <Group
            title="Supervised: regression"
            sub="Has a number column to predict"
            items={pick("supervised", "regression")}
          />
          <Group
            title="Unsupervised: clustering"
            sub="No label column, find hidden groups"
            items={pick("unsupervised")}
          />
        </G>
      </div>
      <div className="col-lg-8 d-grid gap-3">
        {d && (
          <G>
            <div className="d-flex flex-wrap align-items-center gap-3">
              <span
                className="badge rounded-pill px-3 py-2"
                style={{
                  background: "linear-gradient(135deg,#22d3ee,#8b5cf6)",
                  fontSize: ".95rem",
                }}
              >
                {d.type}
              </span>
              <div className="small">
                <div>{d.reason}</div>
                {d.target && (
                  <div>
                    Label column: <b>{d.target}</b>
                  </div>
                )}
                <div className="opacity-75">
                  {d.use} (confidence: {d.confidence})
                </div>
              </div>
              {d.target && (
                <div className="ms-auto small">
                  <div className="opacity-75 mb-1">Not what you expected?</div>
                  <span
                    className={`chip ${!d.chosen ? "on" : ""}`}
                    onClick={() =>
                      d.chosen && load({ ...ds, detected: d.auto })
                    }
                  >
                    As detected
                  </span>
                  <span
                    className={`chip ${d.chosen ? "on" : ""}`}
                    onClick={() =>
                      !d.chosen &&
                      load({
                        ...ds,
                        detected: {
                          ...d,
                          auto: d,
                          chosen: true,
                          type: "Unsupervised (chosen by you)",
                          task: "clustering",
                          reason:
                            "You chose to treat this dataset as unlabeled.",
                          use: `Use the Unsupervised tab. '${d.target}' is left out of the features by default.`,
                          confidence: "set by you",
                        },
                      })
                    }
                  >
                    Treat as unsupervised
                  </span>
                </div>
              )}
            </div>
          </G>
        )}
        {ds ? (
          <G
            title={`${ds.rows} rows · ${ds.cols.length} columns · ${ds.duplicates} duplicates`}
          >
            <div style={{ overflow: "auto", maxHeight: 420 }}>
              <table className="table table-sm">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Missing</th>
                    <th>Unique</th>
                  </tr>
                </thead>
                <tbody>
                  {ds.cols.map((c) => (
                    <tr key={c.name}>
                      <td>{c.name}</td>
                      <td>{c.dtype}</td>
                      <td className={c.missing ? "text-warning" : ""}>
                        {c.missing}
                      </td>
                      <td>{c.unique}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </G>
        ) : (
          <G>No dataset loaded yet.</G>
        )}
      </div>
    </div>
  );
}

function Eda({ ds, eda, setEda, run }) {
  useEffect(() => {
    if (!eda) run(async () => setEda(await api("/eda/" + ds.id)));
  }, []);
  if (!eda)
    return (
      <Loader
        title="Analysing your dataset"
        steps={[
          "Building histograms...",
          "Computing correlations...",
          "Summarising columns...",
        ]}
      />
    );
  return (
    <div className="row g-3">
      <div className="col-12">
        <G title="Correlation heatmap">
          <Corr c={eda.corr} />
        </G>
      </div>
      {Object.entries(eda.hist).map(([k, d], i) => (
        <div key={k} className="col-md-6 col-lg-4">
          <G title={k}>
            <Chart h={170}>
              <BarChart data={d}>
                <XAxis dataKey="x" {...ax} />
                <YAxis {...ax} width={30} />
                {Tip}
                <Bar dataKey="count" fill={COL[i % 8]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </Chart>
          </G>
        </div>
      ))}
      <div className="col-12">
        <G title="Summary statistics">
          <div style={{ overflow: "auto" }}>
            <table className="table table-sm">
              <thead>
                <tr>
                  {Object.keys(eda.describe[0] || {}).map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {eda.describe.map((r, i) => (
                  <tr key={i}>
                    {Object.values(r).map((v, j) => (
                      <td key={j}>{v}</td>
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

function Train({ ds, meta, tr, setTr, run, busy }) {
  const [task, setTask] = useState(
    ds.detected?.task === "regression" ? "regression" : "classification",
  );
  const [target, setTarget] = useState(
    ds.detected?.target || ds.cols.at(-1).name,
  );
  const [algos, setAlgos] = useState([]);
  const [ts, setTs] = useState(0.2);
  const [scale, setScale] = useState(true);
  const [dedupe, setDedupe] = useState(true);
  const [sel, setSel] = useState(null);
  const [mk, setMk] = useState("accuracy");
  const tc = ds.cols.find((c) => c.name === target);
  const warn =
    task === "classification" &&
    tc?.numeric &&
    tc.unique > 20 &&
    tc.unique > 0.05 * ds.rows
      ? {
          to: "regression",
          msg: `'${target}' has ${tc.unique} distinct numbers, so it looks continuous.`,
        }
      : task === "regression" && tc && !tc.numeric
        ? {
            to: "classification",
            msg: `'${target}' contains text or categories.`,
          }
        : null;
  const list = meta[task];
  const metricsList =
    task === "classification"
      ? ["accuracy", "precision", "recall", "f1", "roc_auc"]
      : ["r2", "within_10pct", "mae", "rmse"];
  const key = metricsList.includes(mk) ? mk : metricsList[0];
  const d = tr && sel && tr.details[sel];
  const row = tr && tr.results.find((r) => r.model === sel);
  const pv = d?.points ? d.points.flatMap((p) => [p.actual, p.pred]) : [0, 1];
  const lo = Math.floor(Math.min(...pv)),
    hi = Math.ceil(Math.max(...pv));
  return (
    <div className="row g-3">
      <div className="col-lg-4">
        <G title="Experiment setup">
          <label className="small opacity-75">Learning type</label>
          <select
            className="form-select mb-2"
            value={task}
            onChange={(e) => {
              setTask(e.target.value);
              setAlgos([]);
            }}
          >
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
          <label className="small opacity-75">Target column</label>
          <select
            className="form-select mb-2"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            {ds.cols.map((c) => (
              <option key={c.name}>{c.name}</option>
            ))}
          </select>
          {warn && (
            <div
              className="small mb-2 p-2 rounded"
              style={{
                background: "rgba(250,204,21,.15)",
                border: "1px solid rgba(250,204,21,.4)",
              }}
            >
              {warn.msg}{" "}
              <span
                className="chip on"
                onClick={() => {
                  setTask(warn.to);
                  setAlgos([]);
                }}
              >
                Switch to {warn.to}
              </span>
            </div>
          )}
          <label className="small opacity-75">Test size: {ts}</label>
          <input
            type="range"
            className="form-range"
            min=".1"
            max=".5"
            step=".05"
            value={ts}
            onChange={(e) => setTs(+e.target.value)}
          />
          <div className="form-check">
            <input
              className="form-check-input"
              type="checkbox"
              checked={scale}
              onChange={(e) => setScale(e.target.checked)}
              id="sc"
            />
            <label className="form-check-label" htmlFor="sc">
              Scale numeric features
            </label>
          </div>
          <div className="form-check mb-2">
            <input
              className="form-check-input"
              type="checkbox"
              checked={dedupe}
              onChange={(e) => setDedupe(e.target.checked)}
              id="dd"
            />
            <label className="form-check-label" htmlFor="dd">
              Remove duplicates
            </label>
          </div>
          <div className="small opacity-75">
            Algorithms (pick several to compare)
          </div>
          <Chips items={list} sel={algos} set={setAlgos} />
          <button
            className="btn btn-glow w-100 mt-3"
            disabled={!algos.length}
            onClick={() =>
              run(async () => {
                const r = await api("/train", {
                  dataset_id: ds.id,
                  task,
                  target,
                  algos,
                  test_size: ts,
                  scale,
                  drop_duplicates: dedupe,
                });
                setTr(r);
                setSel(r.results[0].model);
              })
            }
          >
            Train and compare
          </button>
          <div className="small opacity-50 mt-2">
            Imputing, encoding and scaling run inside a pipeline fitted on the
            training split only, so nothing leaks from the test set.
          </div>
        </G>
      </div>
      <div className="col-lg-8">
        {busy ? (
          <Loader
            title="Training your models"
            steps={[
              "Splitting data into train and test sets...",
              "Imputing, encoding and scaling features...",
              "Fitting the models...",
              "Running cross-validation...",
              "Computing metrics and curves...",
            ]}
          />
        ) : tr ? (
          <div className="d-grid gap-3">
            <G title="Model comparison">
              <div style={{ overflow: "auto" }}>
                <table className="table table-sm">
                  <thead>
                    <tr>
                      <th>Model</th>
                      {Object.keys(tr.results[0])
                        .slice(1)
                        .map((k) => (
                          <th key={k}>{k}</th>
                        ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tr.results.map((r) => (
                      <tr
                        key={r.model}
                        onClick={() => setSel(r.model)}
                        style={{
                          cursor: "pointer",
                          background:
                            sel === r.model ? "rgba(34,211,238,.15)" : "",
                        }}
                      >
                        {Object.values(r).map((v, i) => (
                          <td key={i}>{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <select
                className="form-select w-auto mb-2"
                value={key}
                onChange={(e) => setMk(e.target.value)}
              >
                {metricsList.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
              <Chart h={270}>
                <BarChart data={tr.results}>
                  <CartesianGrid stroke="#ffffff18" />
                  <XAxis
                    dataKey="model"
                    interval={0}
                    angle={-30}
                    textAnchor="end"
                    height={85}
                    {...ax}
                  />
                  <YAxis {...ax} width={36} />
                  {Tip}
                  <Bar dataKey={key} fill="#a78bfa" radius={[6, 6, 0, 0]} />
                </BarChart>
              </Chart>
            </G>
            {d && (
              <>
                <G title={`${sel}: results on held-out test set`}>
                  <div className="d-flex flex-wrap justify-content-around gap-3">
                    {task === "classification" ? (
                      ["accuracy", "precision", "recall", "f1", "roc_auc"]
                        .filter((k) => row[k] !== undefined)
                        .map((k) => <Gauge key={k} label={k} v={row[k]} />)
                    ) : (
                      <>
                        <Gauge label="R² (variation explained)" v={row.r2} />
                        <Gauge
                          label="within ±10% of actual"
                          v={row.within_10pct}
                        />
                        {["mae", "rmse"].map((k) => (
                          <div key={k} className="text-center">
                            <div className="fs-4 fw-bold">{row[k]}</div>
                            <div className="small opacity-75">
                              {k} (average error in target units)
                            </div>
                          </div>
                        ))}
                      </>
                    )}
                  </div>
                </G>
                <div className="row g-3">
                  {d.confusion && (
                    <div className="col-md-6">
                      <G title="Confusion matrix">
                        <Confusion m={d.confusion} classes={d.classes} />
                      </G>
                    </div>
                  )}
                  {d.roc && (
                    <div className="col-md-6">
                      <G title="ROC curve">
                        <Chart h={220}>
                          <LineChart data={d.roc}>
                            <CartesianGrid stroke="#ffffff18" />
                            <XAxis dataKey="x" type="number" {...ax} />
                            <YAxis {...ax} width={30} />
                            {Tip}
                            <Line
                              dataKey="y"
                              stroke="#22d3ee"
                              dot={false}
                              strokeWidth={2}
                            />
                          </LineChart>
                        </Chart>
                      </G>
                    </div>
                  )}
                  {d.pr && (
                    <div className="col-md-6">
                      <G title="Precision-recall curve">
                        <Chart h={220}>
                          <LineChart data={d.pr}>
                            <CartesianGrid stroke="#ffffff18" />
                            <XAxis dataKey="x" type="number" {...ax} />
                            <YAxis {...ax} width={30} />
                            {Tip}
                            <Line
                              dataKey="y"
                              stroke="#f472b6"
                              dot={false}
                              strokeWidth={2}
                            />
                          </LineChart>
                        </Chart>
                      </G>
                    </div>
                  )}
                  {d.points && (
                    <div className="col-md-6">
                      <G title="Actual vs predicted">
                        <Chart h={260}>
                          <ScatterChart margin={{ left: 8, bottom: 18 }}>
                            <CartesianGrid stroke="#ffffff18" />
                            <XAxis
                              dataKey="actual"
                              name="Actual"
                              type="number"
                              domain={[lo, hi]}
                              {...ax}
                              label={{
                                value: "Actual value",
                                position: "insideBottom",
                                offset: -10,
                                fill: "#a5b4fc",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              dataKey="pred"
                              name="Predicted"
                              type="number"
                              domain={[lo, hi]}
                              {...ax}
                              width={55}
                              label={{
                                value: "Predicted value",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#a5b4fc",
                                fontSize: 11,
                              }}
                            />
                            {Tip}
                            <ReferenceLine
                              segment={[
                                { x: lo, y: lo },
                                { x: hi, y: hi },
                              ]}
                              stroke="#facc15"
                              strokeDasharray="6 4"
                            />
                            <Scatter
                              data={d.points}
                              fill="#22d3ee"
                              fillOpacity={0.6}
                            />
                          </ScatterChart>
                        </Chart>
                        <div className="small opacity-75 mt-2">
                          Each dot is one unseen test row. The yellow dashed
                          line is a perfect prediction. Dots above it are
                          overestimates, below it are underestimates.
                        </div>
                      </G>
                    </div>
                  )}
                  {d.residuals && (
                    <div className="col-md-6">
                      <G title="Residual distribution">
                        <Chart h={240}>
                          <BarChart data={hist(d.residuals)}>
                            <XAxis dataKey="x" {...ax} />
                            <YAxis {...ax} width={30} />
                            {Tip}
                            <Bar
                              dataKey="n"
                              fill="#facc15"
                              radius={[4, 4, 0, 0]}
                            />
                          </BarChart>
                        </Chart>
                      </G>
                    </div>
                  )}
                  {d.importance && (
                    <div className="col-md-6">
                      <G title="Feature importance">
                        <Chart h={240}>
                          <BarChart data={d.importance} layout="vertical">
                            <XAxis type="number" {...ax} />
                            <YAxis
                              type="category"
                              dataKey="feature"
                              {...ax}
                              width={90}
                            />
                            {Tip}
                            <Bar
                              dataKey="value"
                              fill="#34d399"
                              radius={[0, 6, 6, 0]}
                            />
                          </BarChart>
                        </Chart>
                      </G>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        ) : (
          <G>Choose algorithms and train to see results.</G>
        )}
      </div>
    </div>
  );
}
const hist = (v, b = 15) => {
  const lo = Math.min(...v),
    hi = Math.max(...v),
    w = (hi - lo) / b || 1,
    o = Array.from({ length: b }, (_, i) => ({
      x: +(lo + w * (i + 0.5)).toFixed(2),
      n: 0,
    }));
  v.forEach((a) => o[Math.min(b - 1, Math.floor((a - lo) / w))].n++);
  return o;
};

function Cluster({ ds, meta, cl, setCl, run, busy }) {
  const all = ds.cols.filter((c) => c.numeric).map((c) => c.name);
  const nums = all.filter((c) => c !== ds.detected?.target);
  const [algo, setAlgo] = useState("kmeans");
  const [k, setK] = useState(3);
  const [f, setF] = useState(nums);
  const [auto, setAuto] = useState(true);
  const [method, setMethod] = useState("silhouette");
  useEffect(() => setF(nums), [nums.join()]);
  const KM = {
    elbow: "Elbow method",
    silhouette: "Silhouette score",
    calinski: "Calinski-Harabasz",
    davies_bouldin: "Davies-Bouldin",
    nearest_cluster: "Nearest cluster distance",
  };
  const usesK = ["kmeans", "agglomerative", "gmm"].includes(algo);
  const labs = cl ? [...new Set(cl.points.map((p) => p.c))].sort() : [];
  const go = () =>
    run(async () => {
      const r = await api("/cluster", {
        dataset_id: ds.id,
        algo,
        k,
        features: f,
        k_method: auto ? method : "manual",
      });
      setCl(r);
      if (auto && usesK) setK(r.k_used);
    });
  return (
    <div className="row g-3">
      <div className="col-lg-4">
        <G title="Clustering and anomaly setup">
          <select
            className="form-select mb-2"
            value={algo}
            onChange={(e) => setAlgo(e.target.value)}
          >
            {meta.clustering.map((a) => (
              <option key={a}>{a}</option>
            ))}
          </select>
          {usesK && (
            <div className="mb-2">
              <span
                className={`chip ${auto ? "on" : ""}`}
                onClick={() => setAuto(true)}
              >
                Let the algorithm decide K
              </span>
              <span
                className={`chip ${!auto ? "on" : ""}`}
                onClick={() => setAuto(false)}
              >
                Assign K manually
              </span>
              {auto && (
                <select
                  className="form-select my-2"
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                >
                  {Object.entries(KM).map(([a, b]) => (
                    <option key={a} value={a}>
                      {b}
                    </option>
                  ))}
                </select>
              )}
              <label className="small opacity-75 d-block">
                K: {k}
                {auto
                  ? cl
                    ? ` (chosen by ${KM[method]})`
                    : " (set when you run)"
                  : " (manual)"}
              </label>
              <input
                type="range"
                className="form-range"
                min="2"
                max="10"
                value={k}
                disabled={auto}
                onChange={(e) => setK(+e.target.value)}
              />
              {auto && cl && (
                <button
                  className="btn btn-outline-light btn-sm"
                  onClick={() => setAuto(false)}
                >
                  Change K myself
                </button>
              )}
            </div>
          )}
          <div className="small opacity-75">Features</div>
          <Chips items={all} sel={f} set={setF} />
          <button
            className="btn btn-glow w-100 mt-3"
            disabled={f.length < 2}
            onClick={go}
          >
            Run
          </button>
        </G>
      </div>
      <div className="col-lg-8">
        {busy ? (
          <Loader
            title="Finding patterns in your data"
            steps={[
              "Scaling the features...",
              "Testing different values of K...",
              "Fitting the algorithm...",
              "Scoring cluster quality...",
              "Projecting to 2D with PCA...",
            ]}
          />
        ) : cl ? (
          <div className="d-grid gap-3">
            {usesK && (
              <G title="K suggested by each method (click one to use it manually)">
                {Object.entries(cl.k_picks).map(([a, v]) => (
                  <span
                    key={a}
                    className={`chip ${v === k ? "on" : ""}`}
                    onClick={() => {
                      setAuto(false);
                      setK(v);
                    }}
                  >
                    {KM[a]}: K = {v}
                  </span>
                ))}
              </G>
            )}
            {cl.sampled_rows < cl.total_rows && (
              <div className="small opacity-75">
                Used a random sample of {cl.sampled_rows.toLocaleString()} of{" "}
                {cl.total_rows.toLocaleString()} rows for speed. The plot shows
                up to 5,000 points.
              </div>
            )}
            <G title="Metrics">
              <div className="d-flex flex-wrap gap-4">
                {Object.entries(cl.metrics).map(([a, b]) => (
                  <div key={a}>
                    <div className="fs-4 fw-bold">{b}</div>
                    <div className="small opacity-75">
                      {a.replace("_", " ")}
                    </div>
                  </div>
                ))}
              </div>
            </G>
            <G title={`PCA 2D view (PC1 ${cl.var[0]}%, PC2 ${cl.var[1]}%)`}>
              <Chart h={320}>
                <ScatterChart>
                  <CartesianGrid stroke="#ffffff18" />
                  <XAxis dataKey="x" type="number" {...ax} />
                  <YAxis dataKey="y" type="number" {...ax} width={30} />
                  {Tip}
                  <Legend />
                  {labs.map((l) => (
                    <Scatter
                      key={l}
                      name={l === -1 ? "outlier" : `cluster ${l}`}
                      data={cl.points.filter((p) => p.c === l)}
                      fill={l === -1 ? "#f87171" : COL[l % 8]}
                    />
                  ))}
                </ScatterChart>
              </Chart>
            </G>
            <div className="row g-3">
              {[
                ["inertia", "Elbow method (inertia)", "#22d3ee"],
                ["silhouette", "Silhouette by K", "#f472b6"],
                ["nearest_cluster", "Nearest cluster distance", "#facc15"],
              ].map(([key, t, c]) => (
                <div key={key} className="col-md-4">
                  <G title={t}>
                    <Chart h={170}>
                      <LineChart data={cl.elbow}>
                        <XAxis dataKey="k" {...ax} />
                        <YAxis {...ax} width={40} />
                        {Tip}
                        <Line dataKey={key} stroke={c} strokeWidth={2} />
                      </LineChart>
                    </Chart>
                  </G>
                </div>
              ))}
              <div className="col-12">
                <G title="Cluster distribution">
                  <Chart h={180}>
                    <BarChart
                      data={Object.entries(cl.counts).map(([a, n]) => ({
                        c: a === "-1" ? "outlier" : a,
                        n,
                      }))}
                    >
                      <XAxis dataKey="c" {...ax} />
                      <YAxis {...ax} width={30} />
                      {Tip}
                      <Bar dataKey="n" fill="#a78bfa" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </Chart>
                </G>
              </div>
            </div>
          </div>
        ) : (
          <G>Run an algorithm to see clusters and outliers.</G>
        )}
      </div>
    </div>
  );
}

function Predict({ ds, tr, run }) {
  const [model, setModel] = useState("");
  const [row, setRow] = useState({});
  const [res, setRes] = useState(null);
  if (!tr) return <G>Train at least one model on the Supervised tab first.</G>;
  const m = model || tr.results[0].model;
  return (
    <div className="row g-3">
      <div className="col-lg-6">
        <G title="Enter new values">
          <select
            className="form-select mb-3"
            value={m}
            onChange={(e) => setModel(e.target.value)}
          >
            {tr.results.map((r) => (
              <option key={r.model}>{r.model}</option>
            ))}
          </select>
          <div className="row g-2">
            {tr.features.map((f) => (
              <div className="col-6" key={f.name}>
                <label className="small opacity-75">{f.name}</label>
                {f.type === "num" ? (
                  <input
                    className="form-control"
                    type="number"
                    step="any"
                    value={row[f.name] ?? f.default}
                    onChange={(e) =>
                      setRow({ ...row, [f.name]: e.target.value })
                    }
                  />
                ) : (
                  <select
                    className="form-select"
                    value={row[f.name] ?? f.options[0]}
                    onChange={(e) =>
                      setRow({ ...row, [f.name]: e.target.value })
                    }
                  >
                    {f.options.map((o) => (
                      <option key={o}>{o}</option>
                    ))}
                  </select>
                )}
              </div>
            ))}
          </div>
          <button
            className="btn btn-glow w-100 mt-3"
            onClick={() =>
              run(async () => {
                const full = {};
                tr.features.forEach(
                  (f) =>
                    (full[f.name] =
                      row[f.name] ??
                      (f.type === "num" ? f.default : f.options[0])),
                );
                setRes(
                  await api("/predict", {
                    model_id: `${ds.id}:${m}`,
                    row: full,
                  }),
                );
              })
            }
          >
            Predict
          </button>
        </G>
      </div>
      <div className="col-lg-6">
        {res && (
          <G title="Prediction">
            <div className="display-5 fw-bold">
              {typeof res.prediction === "number"
                ? res.prediction.toFixed(3)
                : res.prediction}
            </div>
            {res.proba &&
              Object.entries(res.proba).map(([c, p]) => (
                <div key={c} className="mt-2">
                  <div className="d-flex justify-content-between small">
                    <span>{c}</span>
                    <span>{(p * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress" style={{ height: 6 }}>
                    <div
                      className="progress-bar"
                      style={{ width: `${p * 100}%`, background: "#22d3ee" }}
                    />
                  </div>
                </div>
              ))}
          </G>
        )}
      </div>
    </div>
  );
}
