import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from "recharts";
import { call } from "./Tuner.jsx";
import { logExp, autoRecord } from "./experiments.js";
import { useNarrow } from './useNarrow.js'

const EFFORT = { Quick: [10, 15], Balanced: [15, 20], Thorough: [25, 40] };
const fmt = (v) => (typeof v === "number" ? +v.toPrecision(3) : String(v));

export default function AutoML({
  ds,
  task,
  target,
  scale,
  dedupe,
  ts,
  balance,
}) {
  const narrow = useNarrow();
  const [effort, setEffort] = useState("Balanced");
  const [job, setJob] = useState(null);
  const [err, setErr] = useState("");
  const running = job?.status === "running";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(async () => {
      try {
        const j = await call("/job/" + job.id);
        setJob(j);
        if (j.status !== "running") logExp(autoRecord(j, ds, task, target));
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
    try {
      const [trials, seconds] = EFFORT[effort];
      const { job_id } = await call("/automl", {
        dataset_id: ds.id,
        task,
        target,
        trials,
        seconds,
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
  const lab = res?.metric === "r2" ? "R²" : "accuracy";
  const ranked = res?.models.filter((m) => m.rank) || [];
  const failed = res?.models.filter((m) => m.error) || [];
  return (
    <div className="glass p-4 mb-3">
      <h6 className="fw-semibold mb-1">AutoML</h6>
      <div className="small opacity-75 mb-3">
        Trains every {task} algorithm, ranks them on unseen test data, tunes the
        top 3 and recommends one. It uses the learning type, target and options
        from the setup panel ({task} on <b>{target}</b>), so change them there
        first. Stay on this tab while it runs.
      </div>
      <div className="d-flex flex-wrap align-items-center gap-3">
        <span className="small opacity-75">Tuning effort</span>
        {Object.keys(EFFORT).map((e) => (
          <span
            key={e}
            className={`chip ${effort === e ? "on" : ""}`}
            onClick={() => !running && setEffort(e)}
          >
            {e}
          </span>
        ))}
        <button
          className="btn btn-glow ms-auto"
          disabled={running}
          onClick={start}
        >
          Run AutoML
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
        <div className="mt-3 d-grid gap-3">
          <div
            className="p-3 rounded"
            style={{
              background:
                "linear-gradient(135deg,rgba(34,211,238,.18),rgba(139,92,246,.18))",
              border: "1px solid rgba(255,255,255,.2)",
            }}
          >
            <div className="small opacity-75">
              {job.status === "cancelled"
                ? "Cancelled early. Showing what finished."
                : "Recommended model"}
            </div>
            <div className="fs-4 fw-bold mb-2">{res.winner}</div>
            <ul className="small mb-0 ps-3">
              {res.summary.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="small fw-semibold mb-1">
              Leaderboard ({lab} on the test set)
            </div>
            <div style={{ height: 40 + ranked.length * 28 }}>
              <ResponsiveContainer>
                <BarChart
                  data={ranked}
                  layout="vertical"
                  margin={{ left: 10, right: 20 }}
                >
                  <CartesianGrid stroke="#ffffff18" />
                  <XAxis
                    type="number"
                    stroke="#a5b4fc"
                    fontSize={11}
                    domain={res.metric === 'r2' ? ['auto', 1] : [0, 1]}
                  />
                  <YAxis
                    type="category"
                    dataKey="model"
                    stroke="#a5b4fc"
                    fontSize={11}
                    width={narrow ? 95 : 130}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#0b1020",
                      border: "1px solid #ffffff33",
                      borderRadius: 10,
                    }}
                  />
                  <Bar dataKey="final" radius={[0, 6, 6, 0]}>
                    {ranked.map((m) => (
                      <Cell
                        key={m.model}
                        fill={m.model === res.winner ? "#22d3ee" : "#a78bfa"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{ overflow: "auto" }}>
            <table className="table table-sm">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Model</th>
                  <th>Default</th>
                  <th>Final</th>
                  <th>Tuned</th>
                  <th>{res.secondary}</th>
                  <th>Time (s)</th>
                  <th>Settings or note</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((m) => (
                  <tr
                    key={m.model}
                    style={{
                      background:
                        m.model === res.winner ? "rgba(34,211,238,.15)" : "",
                    }}
                  >
                    <td>{m.rank}</td>
                    <td>{m.model}</td>
                    <td>{m.default}</td>
                    <td>{m.final}</td>
                    <td>{m.tuned ? "yes" : "no"}</td>
                    <td>{m.secondary}</td>
                    <td>{m.train_time}</td>
                    <td className="small">
                      {m.best_params
                        ? Object.entries(m.best_params)
                            .map(([k, v]) => `${k}=${fmt(v)}`)
                            .join(", ")
                        : m.note || ""}
                    </td>
                  </tr>
                ))}
                {failed.map((m) => (
                  <tr key={m.model}>
                    <td>-</td>
                    <td>{m.model}</td>
                    <td colSpan="6" className="small text-warning">
                      Failed: {m.error}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="small opacity-50">
            Optimizer: {res.optimizer}. A tuned version is used only if it beat
            the defaults in cross-validation on the training data. With many
            models compared, the best test score is slightly optimistic, so
            treat it as an estimate.
          </div>
        </div>
      )}
    </div>
  );
}
