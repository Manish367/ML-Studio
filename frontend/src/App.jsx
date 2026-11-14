import { runJob } from "./jobs.js";
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
import Tuner from "./Tuner.jsx";
import AutoML from "./AutoML.jsx";
import Experiments from "./Experiments.jsx";
import { logExp, trainRecord, clusterRecord } from "./experiments.js";
import { AlgoNote } from "./Learn.jsx";
import { MultiRoc, Dendrogram, SilhouettePlot } from "./Viz.jsx";
import { useNarrow } from "./useNarrow.js";

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
    (import.meta.env.VITE_API_URL || "") + "/api" + url,
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
  const [exps, setExps] = useState([]);
  useEffect(() => {
    const h = (e) =>
      setExps((x) =>
        e.detail.jobId && x.some((y) => y.jobId === e.detail.jobId)
          ? x
          : [
              ...x,
              {
                key: crypto.randomUUID(),
                time: new Date().toLocaleTimeString(),
                ...e.detail,
              },
            ],
      );
    window.addEventListener("ml-exp", h);
    return () => window.removeEventListener("ml-exp", h);
  }, []);
  const [err, setErr] = useState("");
  const [waking, setWaking] = useState(true);
  const loadMeta = async (n = 0) => {
    setWaking(true);
    try {
      setMeta(await api("/algos"));
      setErr("");
      setWaking(false);
    } catch {
      if (n >= 12) {
        setWaking(false);
        setErr("Could not reach the server. It may still be starting up (free hosting). Tap Retry in a minute.");
      } else setTimeout(() => loadMeta(n + 1), 5000);
    }
  };
  useEffect(() => { loadMeta(); }, []);
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
    ["compare", "Before/After"],
    ["train", "Supervised"],
    ["cluster", "Unsupervised"],
    ["predict", "Predict"],
    ["history", "History"],
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
        <div className="glass p-2 d-flex flex-wrap flex-lg-nowrap gap-1 ms-auto navscroll">
          {tabs.map(([k, l]) => (
            <div
              key={k}
              className={`tabbtn ${tab === k ? "on" : ""}`}
              onClick={() => { setTab(k); window.scrollTo({ top: 0, behavior: "smooth" }); }}
            >
              {l}
            </div>
          ))}
        </div>
      </div>
            {!meta && waking && (
        <div className="alert glass py-2 small">Waking up the server... the free host sleeps when idle, so the first load can take up to a minute. Examples appear as soon as it responds.</div>
      )}
      {err && (
        <div className="alert alert-danger glass d-flex flex-wrap gap-2 align-items-center">
          <span>{err}</span>
          {!meta && <button className="btn btn-outline-light btn-sm ms-auto" onClick={() => loadMeta()}>Retry</button>}
        </div>
      )}
      {busy && (
        <div className="progress mb-3" style={{ height: 4 }}>
          <div className="progress-bar progress-bar-striped progress-bar-animated w-100" />
        </div>
      )}
      {tab === "data" && <Data meta={meta} ds={ds} load={load} run={run} />}
      {tab !== "data" && tab !== "history" && !ds && (
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
      {ds && tab === "predict" && <Predict ds={ds} run={run} />}
      {tab === "history" && <Experiments exps={exps} setExps={setExps} />}
    </div>
  );
}

function Data({ meta, ds, load, run }) {
  const [ins, setIns] = useState(null);
  useEffect(() => {
    if (!ds) {
      setIns(null);
      return;
    }
    api("/insights/" + ds.id)
      .then((r) => setIns(r.items))
      .catch(() => setIns(null));
  }, [ds?.id, ds?.rows, ds?.cols.length]);
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
          {!meta && <div className="small opacity-75 mb-3">Loading examples... the free server can take up to a minute to wake up.</div>}
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
        {ins && ins.length > 0 && (
          <G title="Dataset insights">
            {ins.map((x, i) => (
              <div key={i} className="d-flex gap-2 mb-2">
                <span
                  style={{
                    color:
                      x.level === "warn"
                        ? "#f87171"
                        : x.level === "good"
                          ? "#34d399"
                          : "#facc15",
                  }}
                >
                  ●
                </span>
                <div>
                  <div className="fw-semibold small">{x.title}</div>
                  <div className="small opacity-75">{x.detail}</div>
                  {x.fix && (
                    <div className="small" style={{ color: "#22d3ee" }}>
                      Suggestion: {x.fix}
                    </div>
                  )}
                </div>
              </div>
            ))}
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
const Box = ({ name, b }) => {
  const span = b.max - b.min || 1,
    p = (v) => ((v - b.min) / span) * 100;
  const dot = (left) => (
    <div
      style={{
        position: "absolute",
        top: 7,
        left: `${left}%`,
        width: 4,
        height: 4,
        borderRadius: 2,
        background: "#f472b6",
      }}
    />
  );
  return (
    <div
      className="d-flex align-items-center gap-2 mb-2"
      title={`min ${b.min} · Q1 ${b.q1} · median ${b.median} · Q3 ${b.q3} · max ${b.max}`}
    >
      <div className="small text-end opacity-75" style={{ width: 130 }}>
        {name}
      </div>
      <div style={{ position: "relative", flex: 1, height: 18 }}>
        <div
          style={{
            position: "absolute",
            top: 8,
            height: 2,
            left: `${p(b.lo)}%`,
            width: `${p(b.hi) - p(b.lo)}%`,
            background: "#a5b4fc",
          }}
        />
        <div
          style={{
            position: "absolute",
            top: 2,
            height: 14,
            left: `${p(b.q1)}%`,
            width: `${Math.max(0.5, p(b.q3) - p(b.q1))}%`,
            background: "rgba(34,211,238,.45)",
            border: "1px solid #22d3ee",
          }}
        />
        <div
          style={{
            position: "absolute",
            top: 0,
            height: 18,
            width: 2,
            left: `${p(b.median)}%`,
            background: "#facc15",
          }}
        />
        {b.min < b.lo && dot(p(b.min))}
        {b.max > b.hi && dot(p(b.max))}
      </div>
      <div className="small opacity-50" style={{ width: 80 }}>
        {b.outliers} outliers
      </div>
    </div>
  );
};
function Eda({ ds, eda, setEda, run }) {
  useEffect(() => {
    if (!eda) run(async () => setEda(await api("/eda/" + ds.id)));
  }, []);
  const [sx, setSx] = useState("");
  const [sy, setSy] = useState("");
  const [sc, setSc] = useState("");
  const [pts, setPts] = useState(null);
  useEffect(() => {
    if (!sx || !sy) {
      setPts(null);
      return;
    }
    api(
      `/eda/${ds.id}/scatter?x=${encodeURIComponent(sx)}&y=${encodeURIComponent(sy)}${sc ? `&color=${encodeURIComponent(sc)}` : ""}`,
    )
      .then(setPts)
      .catch(() => setPts(null));
  }, [sx, sy, sc]);
  const numCols = ds.cols.filter((c) => c.numeric).map((c) => c.name);
  const colorCols = ds.cols
    .filter((c) => c.unique > 1 && c.unique <= 8)
    .map((c) => c.name);
  const groups = pts ? [...new Set(pts.points.map((p) => p.c))] : [];
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
      {eda.box && Object.keys(eda.box).length > 0 && (
        <div className="col-12">
          <G title="Box plots">
            {Object.entries(eda.box).map(([k, b]) => (
              <Box key={k} name={k} b={b} />
            ))}
            <div className="small opacity-75">
              Each box uses its own scale, from that column's minimum to its
              maximum. The box is the middle 50% of values, the yellow line is
              the median, the thin line is the normal range, and pink dots mark
              values beyond it (outliers). Hover a row for the exact numbers.
            </div>
          </G>
        </div>
      )}
      <div className="col-12">
        <G title="Scatter plot">
          <div className="d-flex flex-wrap gap-2 mb-2">
            <select
              className="form-select w-auto"
              value={sx}
              onChange={(e) => setSx(e.target.value)}
            >
              <option value="">X axis...</option>
              {numCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <select
              className="form-select w-auto"
              value={sy}
              onChange={(e) => setSy(e.target.value)}
            >
              <option value="">Y axis...</option>
              {numCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <select
              className="form-select w-auto"
              value={sc}
              onChange={(e) => setSc(e.target.value)}
            >
              <option value="">No colour</option>
              {colorCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>
          {pts ? (
            <Chart h={320}>
              <ScatterChart margin={{ left: 8, bottom: 18 }}>
                <CartesianGrid stroke="#ffffff18" />
                <XAxis dataKey="x" name={sx} type="number" {...ax} />
                <YAxis dataKey="y" name={sy} type="number" {...ax} width={50} />
                {Tip}
                {groups.length > 1 && <Legend />}
                {groups.map((g, i) => (
                  <Scatter
                    key={g}
                    name={g}
                    data={pts.points.filter((p) => p.c === g)}
                    fill={COL[i % 8]}
                    fillOpacity={0.6}
                  />
                ))}
              </ScatterChart>
            </Chart>
          ) : (
            <div className="small opacity-75">
              Pick two numeric columns to see how they relate. Optionally colour
              the dots by a column with a few categories. Shows up to 800 rows.
            </div>
          )}
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
  const [balance, setBalance] = useState(false);
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
      ? [
          "accuracy",
          "balanced_accuracy",
          "precision",
          "recall",
          "f1",
          "roc_auc",
          "pr_auc",
        ]
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
          {task === "classification" && (
            <div className="form-check mb-2">
              <input
                className="form-check-input"
                type="checkbox"
                checked={balance}
                onChange={(e) => setBalance(e.target.checked)}
                id="bl"
              />
              <label className="form-check-label" htmlFor="bl">
                Balance classes{" "}
                <span className="opacity-50">
                  (helps when one class is rare)
                </span>
              </label>
            </div>
          )}
          <div className="small opacity-75">
            Algorithms (pick several to compare)
          </div>
          <Chips items={list} sel={algos} set={setAlgos} />
          {algos.length > 0 && (
            <AlgoNote
              key={algos[algos.length - 1]}
              name={algos[algos.length - 1]}
            />
          )}
          <button
            className="btn btn-glow w-100 mt-3"
            disabled={!algos.length}
            onClick={() =>
              run(async () => {
                const r = await runJob("/train/start", {
                  dataset_id: ds.id,
                  task,
                  target,
                  algos,
                  test_size: ts,
                  scale,
                  drop_duplicates: dedupe,
                  balance,
                });
                setTr(r);
                setSel(r.results[0].model);
                logExp(
                  trainRecord(r, ds, task, target, algos, ts, scale, dedupe),
                );
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
        <AutoML
          ds={ds}
          task={task}
          target={target}
          scale={scale}
          dedupe={dedupe}
          ts={ts}
          balance={balance}
        />
        <Tuner
          ds={ds}
          task={task}
          target={target}
          algos={algos}
          scale={scale}
          dedupe={dedupe}
          ts={ts}
          balance={balance}
        />
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
                      [
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "pr_auc",
                      ]
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
                {d.roc_multi && d.roc_multi.length > 0 && (
                  <div className="col-md-6">
                    <G title="ROC curves (each class vs the rest)">
                      <MultiRoc curves={d.roc_multi} />
                    </G>
                  </div>
                )}
                <div className="row g-3">
                  {d.confusion && (
                    <div className="col-md-6">
                      <G title="Confusion matrix">
                        <Confusion m={d.confusion} classes={d.classes} />
                      </G>
                    </div>
                  )}
                  {d.report && (
                    <div className="col-12">
                      <G title="Classification report">
                        <div style={{ overflow: "auto" }}>
                          <table className="table table-sm">
                            <thead>
                              <tr>
                                <th>Class</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1</th>
                                <th>Support</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(d.report)
                                .filter(([, v]) => typeof v === "object")
                                .map(([k, v]) => (
                                  <tr
                                    key={k}
                                    style={{
                                      fontWeight: k.includes("avg") ? 600 : 400,
                                    }}
                                  >
                                    <td>{k}</td>
                                    <td>{v.precision.toFixed(3)}</td>
                                    <td>{v.recall.toFixed(3)}</td>
                                    <td>{v["f1-score"].toFixed(3)}</td>
                                    <td>{Math.round(v.support)}</td>
                                  </tr>
                                ))}
                            </tbody>
                          </table>
                        </div>
                        <div className="small opacity-75">
                          Precision: of the rows predicted as this class, how
                          many were right. Recall: of the rows that truly belong
                          to it, how many were found. Support: how many test
                          rows belong to it.
                        </div>
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
  const [eps, setEps] = useState(1.2);
  const [minS, setMinS] = useState(5);
  const [cont, setCont] = useState(5);
  const labelCol =
    ds.detected?.task === "anomaly"
      ? ds.detected.target
      : ds.detected?.auto?.task === "anomaly"
        ? ds.detected.auto.target
        : null;
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
      const r = await runJob("/cluster/start", {
        dataset_id: ds.id,
        algo,
        k,
        features: f,
        k_method: auto ? method : "manual",
        eps,
        min_samples: minS,
        contamination: cont / 100,
        label_col: labelCol,
      });
      setCl(r);
      if (auto && usesK) setK(r.k_used);
      logExp(clusterRecord(r, ds, algo, f.length, usesK, auto, method));
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
          <AlgoNote key={algo} name={algo} />
          {algo === "dbscan" && (
            <div className="mb-2">
              <label className="small opacity-75 d-block">
                Neighbourhood size (eps): {eps}
              </label>
              <input
                type="range"
                className="form-range"
                min="0.2"
                max="5"
                step="0.1"
                value={eps}
                onChange={(e) => setEps(+e.target.value)}
              />
              <label className="small opacity-75 d-block">
                Minimum points per group: {minS}
              </label>
              <input
                type="range"
                className="form-range"
                min="2"
                max="20"
                value={minS}
                onChange={(e) => setMinS(+e.target.value)}
              />
            </div>
          )}
          {["isolation_forest", "lof", "oneclass_svm"].includes(algo) && (
            <div className="mb-2">
              <label className="small opacity-75 d-block">
                Expected share of outliers: {cont}%
              </label>
              <input
                type="range"
                className="form-range"
                min="1"
                max="20"
                value={cont}
                onChange={(e) => setCont(+e.target.value)}
              />
            </div>
          )}
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
            {cl.label_check && (
              <G title={`Check against '${cl.label_check.label}'`}>
                <div className="d-flex flex-wrap justify-content-around gap-3">
                  <Gauge label="precision" v={cl.label_check.precision} />
                  <Gauge label="recall" v={cl.label_check.recall} />
                  <Gauge label="F1" v={cl.label_check.f1} />
                </div>
                <div className="small opacity-75 mt-2">
                  Treats label value '{cl.label_check.positive}' (the rarer one,{" "}
                  {cl.label_check.true_outliers} rows) as a true outlier and
                  compares it with the {cl.label_check.flagged} rows the
                  detector flagged. It caught {cl.label_check.tp} of{" "}
                  {cl.label_check.true_outliers} true outliers and raised{" "}
                  {cl.label_check.fp} false alarms. Precision is how many flags
                  were right, and recall is how many true outliers were found.
                </div>
              </G>
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
                      name={l === -1 ? "outlier" : cl.anomaly ? "normal" : `cluster ${l}`}
                      data={cl.points.filter((p) => p.c === l)}
                      fill={l === -1 ? "#f87171" : COL[l % 8]}
                    />
                  ))}
                </ScatterChart>
              </Chart>
            </G>
            {cl.silhouette_plot && (
              <G
                title={`Silhouette analysis (average ${cl.silhouette_plot.avg})`}
              >
                <SilhouettePlot s={cl.silhouette_plot} />
                <div className="small opacity-75 mt-2">
                  Each bar is one point, grouped by cluster. Longer bars to the
                  right mean a point sits well inside its cluster. Bars to the
                  left of the grey line (negative values) are points that may
                  belong to a different cluster. The yellow line is the average.
                  Shows up to 400 points.
                </div>
              </G>
            )}
            {cl.dendro && (
              <G title={`Hierarchical tree (sample of ${cl.dendro.rows} rows)`}>
                <Dendrogram t={cl.dendro} />
                <div className="small opacity-75 mt-2">
                  Rows merge from the bottom upwards. The higher two branches
                  join, the less alike they are. The dashed line shows where
                  your chosen K cuts the tree into clusters. The lowest branches
                  are bundled together to keep it readable.
                </div>
              </G>
            )}
            <div className="row g-3">
              {[
                ["inertia", "Elbow method (inertia)", "#22d3ee"],
                ["silhouette", "Silhouette by K", "#f472b6"],
                ["nearest_cluster", "Nearest cluster distance", "#facc15"],
              ].filter(() => cl.elbow.length > 0).map(([key, t, c]) => (
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
                        c: a === "-1" ? "outlier" : cl.anomaly ? "normal" : a,
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

function Predict({ ds, run }) {
  const [models, setModels] = useState([]);
  const narrow = useNarrow()
  const [model, setModel] = useState("");
  const [row, setRow] = useState({});
  const [res, setRes] = useState(null);
  const [why, setWhy] = useState(null);
  const [imp, setImp] = useState(null);
  const [hasShap, setHasShap] = useState(false);
  const [method, setMethod] = useState("whatif");
  useEffect(() => {
    api("/models/" + ds.id)
      .then((l) => {
        setModels(l);
        setModel(l[0]?.name || "");
      })
      .catch(() => {});
  }, [ds.id]);
  useEffect(() => {
    api("/explain/capabilities")
      .then((r) => setHasShap(!!r.shap))
      .catch(() => {});
  }, []);
  const m = models.find((x) => x.name === model);
  if (!m)
    return (
      <G>
        Train a model first on the Supervised tab (or with Tuning or AutoML),
        then come back here to make predictions and see explanations.
      </G>
    );
  const modelId = `${ds.id}:${m.name}`;
  const path = method === "shap" ? "/explain/shap" : "/explain";
  const download =
    (import.meta.env.VITE_API_URL || "") +
    "/api/model/download?model_id=" +
    encodeURIComponent(modelId);
  const body = () => {
    const o = {};
    m.feats.forEach((f) => {
      o[f.name] = row[f.name] ?? f.default;
    });
    return { model_id: modelId, row: o };
  };
  const pick = (e) => {
    setModel(e.target.value);
    setRow({});
    setRes(null);
    setWhy(null);
    setImp(null);
  };
  const chooseMethod = (k) => {
    setMethod(k);
    setWhy(null);
    setImp(null);
  };
  const predict = () =>
    run(async () => {
      const b = body();
      setRes(await api("/predict", b));
      setWhy(await api(path + "/local", b));
    });
  const importance = () =>
    run(async () => setImp(await api(path + "/global", { model_id: modelId })));
  const bars = why?.items.map((x) => ({
    label: `${x.feature} = ${typeof x.value === "number" ? +x.value.toPrecision(4) : x.value}`,
    pos: Math.max(0, x.contribution),
    neg: Math.min(0, x.contribution),
  }));
  return (
    <div className="row g-3">
      <div className="col-lg-5">
        <G title="Enter new values">
          <select className="form-select mb-3" value={model} onChange={pick}>
            {models.map((x) => (
              <option key={x.name}>{x.name}</option>
            ))}
          </select>
          <div className="row g-2">
            {m.feats.map((f) => (
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
                    value={row[f.name] ?? f.default}
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
          <button className="btn btn-glow w-100 mt-3" onClick={predict}>
            Predict and explain
          </button>
          <a
            className="btn btn-outline-light btn-sm w-100 mt-2"
            href={download}
          >
            Download this model (.joblib)
          </a>
          <details className="small mt-2">
            <summary style={{ cursor: "pointer", color: "#22d3ee" }}>
              How to use the downloaded file
            </summary>
            <pre
              className="small mb-0 mt-1"
              style={{ whiteSpace: "pre-wrap" }}
            >{`import joblib, pandas as pd
b = joblib.load("the-file-you-downloaded.joblib")  # only load files you trust
row = pd.DataFrame([{"column": value, ...}])
pred = b["pipe"].predict(row)[0]
# classification: b["classes"][pred] is the label`}</pre>
          </details>
        </G>
      </div>
      <div className="col-lg-7 d-grid gap-3">
        {hasShap && (
          <div className="small">
            <span className="opacity-75 me-2">Explanation method</span>
            <span
              className={`chip ${method === "whatif" ? "on" : ""}`}
              onClick={() => chooseMethod("whatif")}
            >
              What-if (fast)
            </span>
            <span
              className={`chip ${method === "shap" ? "on" : ""}`}
              onClick={() => chooseMethod("shap")}
            >
              SHAP
            </span>
            <span className="opacity-50">
              {" "}
              Click Predict again after switching.
            </span>
          </div>
        )}
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
        {why && (
          <G
            title={
              why.method === "shap"
                ? "Why this prediction? (SHAP)"
                : "Why this prediction?"
            }
          >
            <Chart h={40 + bars.length * 28}>
              <BarChart
                data={bars}
                layout="vertical"
                stackOffset="sign"
                margin={{ left: 10, right: 20 }}
              >
                <CartesianGrid stroke="#ffffff18" />
                <XAxis type="number" {...ax} />
                <YAxis type="category" dataKey="label" {...ax} width={narrow ? 95 : 140} />
                {Tip}
                <Bar dataKey="pos" stackId="a" fill="#22d3ee" />
                <Bar dataKey="neg" stackId="a" fill="#f472b6" />
              </BarChart>
            </Chart>
            {why.method === "shap" ? (
              <div className="small opacity-75 mt-2">
                SHAP shares the gap between the average prediction (
                {why.baseline}) and this one ({why.full}) fairly among the
                inputs, measured as {why.measure}. Right pushes it up, left
                pushes it down. The baseline is the average over a sample of
                unseen test rows.
              </div>
            ) : (
              <div className="small opacity-75 mt-2">
                Each bar shows how much the {why.measure} changes when that
                input is swapped for a typical value (the median, or the most
                common category). Right pushes it up, left pushes it down. With
                typical inputs it would be {why.baseline}; with yours it is{" "}
                {why.full}. This is an approximation, because overlapping
                columns can double-count.
              </div>
            )}
          </G>
        )}
        <G title="Which columns matter most overall">
          <button
            className="btn btn-outline-light btn-sm mb-2"
            onClick={importance}
          >
            {imp ? "Recalculate" : "Calculate"}
          </button>
          {imp ? (
            <>
              <Chart h={40 + imp.items.length * 26}>
                <BarChart
                  data={imp.items}
                  layout="vertical"
                  margin={{ left: 10, right: 20 }}
                >
                  <CartesianGrid stroke="#ffffff18" />
                  <XAxis type="number" {...ax} />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    {...ax}
                    width={narrow ? 85 : 110}
                  />
                  {Tip}
                  <Bar
                    dataKey="importance"
                    fill="#34d399"
                    radius={[0, 6, 6, 0]}
                  />
                </BarChart>
              </Chart>
              {imp.method === "shap" ? (
                <div className="small opacity-75 mt-2">
                  Average absolute SHAP impact of each column on {imp.rows}{" "}
                  unseen rows ({imp.metric}). A bigger bar means that column
                  moves the predictions more.
                </div>
              ) : (
                <div className="small opacity-75 mt-2">
                  How much the {imp.metric} drops on {imp.rows} unseen test rows
                  when this column's values are shuffled. A bigger bar means the
                  model relies on that column more. Near zero means it barely
                  matters.
                </div>
              )}
            </>
          ) : (
            <div className="small opacity-75">
              Shuffles each column in turn (What-if) or averages SHAP values
              (SHAP) to show which columns matter most. Takes a few seconds.
            </div>
          )}
        </G>
      </div>
    </div>
  );
}
