import { call } from "./Tuner.jsx";

// Starts a background job on the server and follows its real progress until it finishes.
export const runJob = async (path, body) => {
  const { job_id } = await call(path, body);
  for (;;) {
    const j = await call("/job/" + job_id);
    window.dispatchEvent(
      new CustomEvent("ml-prog", {
        detail: { pct: j.pct, message: j.message },
      }),
    );
    if (j.status === "done") return j.result;
    if (j.status === "error" || j.status === "cancelled")
      throw new Error(j.error || "The run stopped.");
    await new Promise((r) => setTimeout(r, 500));
  }
};
