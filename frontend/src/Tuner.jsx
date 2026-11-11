import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { logExp, tuneRecord } from "./experiments.js";
const BASE = import.meta.env.VITE_API_URL || "";
const CID = (() => {
  try {
    let v = sessionStorage.getItem("ml-cid");
    if (!v) {
      v = crypto.randomUUID();
      sessionStorage.setItem("ml-cid", v);
    }
    return v;
  } catch {
    return crypto.randomUUID();
  }
})();
export const call = async (url, body) => {
  const r = await fetch(BASE + "/api" + url, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", "X-Client-Id": CID },
    body: body === undefined ? undefined : JSON.stringify(body),
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
const fmt = (v) => (typeof v === "number" ? +v.toPrecision(3) : String(v));

export default function Tuner({
  ds,
  task,
  target,
  algos,
  scale,
  dedupe,
  ts,
  balance,
}) {
  const [trials, setTrials] = useState(20);
  const [secs, setSecs] = useState(30);
  const [job, setJob] = useState(null);
  const [err, setErr] = useState("");
  const [sel, setSel] = useState(null);
  const running = job?.status === "running";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(async () => {
      try {
        const j = await call("/job/" + job.id);
        setJob(j);
        if (j.status !== "running") logExp(tuneRecord(j, ds, task, target));
      } catch (e) {
        setErr(e.message);
        setJob(null);
      }
    }, 1000);
    return () => clearInterval(t);
  }, [job?.id, running]);
  const start = async () => {
    setErr("");
    setJob(null);
    setSel(null);
    try {
      const { job_id } = await call("/tune", {
        dataset_id: ds.id,
        task,
        target,
        algos,
        trials,
        seconds: secs,
        test_size: ts,
        scale,
        drop_duplicates: dedupe,
        balance,
      });
      setJob({ id: job_id, status: "running", pct: 0, message: "Starting..." });
    } catch (e) {
      setErr(e.message);
    }
  };
  const res = job?.result;
  const shown =
    res?.models.find((m) => m.model === sel && m.history) ||
    res?.models.find((m) => m.history);
  const better = res?.metric === "r2" ? "R²" : "accuracy";
  return (
    <div className="glass p-4 mb-3">
      <h6 className="fw-semibold mb-1">Hyperparameter tuning</h6>
      <div className="small opacity-75 mb-3">
        Searches for better settings for the algorithms ticked on the left,
        using cross-validation on the training data only. Stay on this tab while
        it runs.
      </div>
      <div className="d-flex flex-wrap align-items-end gap-4">
        <div>
          <label className="small opacity-75 d-block">
            Trials per model: {trials}
          </label>
          <input
            type="range"
            className="form-range"
            min="5"
            max="30"
            value={trials}
            disabled={running}
            onChange={(e) => setTrials(+e.target.value)}
          />
        </div>
        <div>
          <label className="small opacity-75 d-block">
            Max seconds per model: {secs}
          </label>
          <input
            type="range"
            className="form-range"
            min="10"
            max="90"
            step="10"
            value={secs}
            disabled={running}
            onChange={(e) => setSecs(+e.target.value)}
          />
        </div>
        <button
          className="btn btn-glow"
          disabled={!algos.length || running}
          onClick={start}
        >
          Tune {algos.length || ""} selected model
          {algos.length === 1 ? "" : "s"}
        </button>
      </div>
      {err && (
        <div className="alert alert-danger mt-3 mb-0 py-2 small">{err}</div>
      )}
      {running && (
        <div className="mt-3">
          <div className="d-flex justify-content-between small mb-1">
            <span>{job.message}</span>
            <span>{job.pct}%</span>
          </div>
          <div className="progress" style={{ height: 8 }}>
            <div
              className="progress-bar progress-bar-striped progress-bar-animated"
              style={{
                width: `${job.pct}%`,
                background: "linear-gradient(90deg,#22d3ee,#8b5cf6)",
              }}
            />
          </div>
          <button
            className="btn btn-outline-light btn-sm mt-2"
            onClick={() => call("/job/" + job.id + "/cancel", {})}
          >
            Cancel
          </button>
        </div>
      )}
      {job?.status === "error" && (
        <div className="alert alert-danger mt-3 mb-0 py-2 small">
          {job.error}
        </div>
      )}
      {res && (
        <div className="mt-3">
          <div className="small opacity-75 mb-2">
            {job.status === "cancelled"
              ? "Cancelled early. Showing what finished. "
              : ""}
            Optimizer: {res.optimizer}. Scores are {better} on the test set,
            which tuning never saw. Click a row to see its search curve.
          </div>
          <div style={{ overflow: "auto" }}>
            <table className="table table-sm">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Before</th>
                  <th>After</th>
                  <th>Change</th>
                  <th>CV before → after</th>
                  <th>Trials</th>
                  <th>Best settings</th>
                </tr>
              </thead>
              <tbody>
                {res.models.map((m) => {
                  const d = m.tuned !== undefined ? m.tuned - m.baseline : null;
                  return (
                    <tr
                      key={m.model}
                      onClick={() => m.history && setSel(m.model)}
                      style={{
                        cursor: m.history ? "pointer" : "default",
                        background:
                          shown?.model === m.model
                            ? "rgba(34,211,238,.15)"
                            : "",
                      }}
                    >
                      <td>{m.model}</td>
                      <td>{m.baseline ?? "-"}</td>
                      <td>{m.tuned ?? "-"}</td>
                      <td
                        style={{
                          color:
                            d === null
                              ? "inherit"
                              : d > 0
                                ? "#34d399"
                                : d < 0
                                  ? "#f87171"
                                  : "inherit",
                        }}
                      >
                        {d === null ? "-" : (d > 0 ? "+" : "") + d.toFixed(4)}
                      </td>
                      <td>
                        {m.cv_baseline !== undefined
                          ? `${m.cv_baseline} → ${m.cv_best}`
                          : "-"}
                      </td>
                      <td>{m.trials ?? "-"}</td>
                      <td className="small">
                        {m.best_params
                          ? Object.entries(m.best_params)
                              .map(([k, v]) => `${k}=${fmt(v)}`)
                              .join(", ")
                          : m.note || m.error}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {shown && (
            <div>
              <div className="small fw-semibold mb-1">
                {shown.model}: best cross-validation score so far, per trial
              </div>
              <div style={{ height: 180 }}>
                <ResponsiveContainer>
                  <LineChart
                    data={shown.history.map((s, i) => ({
                      trial: i + 1,
                      score: s,
                    }))}
                  >
                    <CartesianGrid stroke="#ffffff18" />
                    <XAxis dataKey="trial" stroke="#a5b4fc" fontSize={11} />
                    <YAxis
                      stroke="#a5b4fc"
                      fontSize={11}
                      width={45}
                      domain={["auto", "auto"]}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0b1020",
                        border: "1px solid #ffffff33",
                        borderRadius: 10,
                      }}
                    />
                    <Line
                      dataKey="score"
                      stroke="#22d3ee"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
          <div className="small opacity-50 mt-2">
            If "After" is lower than "Before", the tuned settings fitted the
            cross-validation folds too closely. Keep the defaults in that case.
          </div>
        </div>
      )}
    </div>
  );
}
