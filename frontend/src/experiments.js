export const logExp = (d) => {
  if (d) window.dispatchEvent(new CustomEvent("ml-exp", { detail: d }));
};
const base = (ds) => ({
  data: `${ds.rows} rows × ${ds.cols.length} columns`,
  detected: ds.detected?.type || "",
  steps: ds.log || [],
});
const r4 = (v) => (typeof v === "number" ? +v.toFixed(4) : v);

export const trainRecord = (r, ds, task, target, algos, ts, scale, dedupe) => {
  const key = task === "classification" ? "accuracy" : "r2";
  const b = [...r.results].sort((a, c) => c[key] - a[key])[0];
  return {
    kind: "Train",
    task,
    target,
    summary: algos.join(", "),
    scoreLabel: key === "r2" ? "R²" : "accuracy",
    score: r4(b[key]),
    headline: `Best: ${b.model}`,
    settings: `test size ${ts}, ${scale ? "scaled" : "not scaled"}, ${dedupe ? "duplicates removed" : "duplicates kept"}`,
    ...base(ds),
  };
};
export const clusterRecord = (r, ds, algo, nFeatures, usesK, auto, method) => ({
  kind: "Cluster",
  task: "clustering",
  target: "",
  summary: `${algo}${usesK ? `, K=${r.k_used} (${auto ? "auto: " + method : "manual"})` : ""}`,
  scoreLabel: r.metrics.silhouette !== undefined ? "silhouette" : "",
  score: r.metrics.silhouette,
  headline:
    r.metrics.anomalies !== undefined
      ? `${r.metrics.anomalies} anomalies found`
      : `${r.metrics.clusters ?? "?"} clusters`,
  settings: `${nFeatures} features${r.sampled_rows < r.total_rows ? `, sample of ${r.sampled_rows} rows` : ""}`,
  ...base(ds),
});
export const tuneRecord = (j, ds, task, target) => {
  const res = j.result;
  if (!res) return null;
  const t = res.models
    .filter((m) => m.tuned !== undefined)
    .sort((a, b) => b.tuned - a.tuned)[0];
  return {
    jobId: j.id,
    kind: "Tuning",
    task,
    target,
    summary: res.models.map((m) => m.model).join(", "),
    scoreLabel: res.metric === "r2" ? "R²" : "accuracy",
    score: t ? t.tuned : null,
    headline: t
      ? `Best: ${t.model} ${t.baseline} → ${t.tuned}`
      : "No model could be tuned",
    settings: res.optimizer,
    ...base(ds),
  };
};
export const autoRecord = (j, ds, task, target) => {
  const res = j.result;
  if (!res) return null;
  const w = res.models.find((m) => m.model === res.winner);
  return {
    jobId: j.id,
    kind: "AutoML",
    task,
    target,
    summary: `${res.models.length} models compared`,
    scoreLabel: res.metric === "r2" ? "R²" : "accuracy",
    score: w ? w.final : null,
    headline: `Winner: ${res.winner}`,
    settings: res.optimizer,
    ...base(ds),
  };
};
