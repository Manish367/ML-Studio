import { useEffect, useState } from "react";
export default function Loader({ title = "Working", steps = [] }) {
  const [i, setI] = useState(0);
  const [t, setT] = useState(0);
  const [p, setP] = useState(null);
  useEffect(() => {
    const a = setInterval(() => setI((x) => x + 1), 1800),
      b = setInterval(() => setT((x) => x + 1), 1000);
    const h = (e) => setP(e.detail);
    window.addEventListener("ml-prog", h);
    return () => {
      clearInterval(a);
      clearInterval(b);
      window.removeEventListener("ml-prog", h);
    };
  }, []);
  return (
    <div className="glass ld-wrap">
      <div className="ld-orb">
        <i />
        <i />
        <i />
        <b />
      </div>
      <h5 className="fw-bold mb-1">{title}</h5>
      {p ? (
        <div style={{ width: "100%", maxWidth: 420 }}>
          <div className="small opacity-75 mb-1">{p.message}</div>
          <div className="progress" style={{ height: 8 }}>
            <div
              className="progress-bar"
              style={{
                width: `${p.pct}%`,
                background: "linear-gradient(90deg,#22d3ee,#8b5cf6)",
                transition: "width .4s",
              }}
            />
          </div>
          <div className="small opacity-50 mt-1">{p.pct}%</div>
        </div>
      ) : (
        <div
          key={i}
          className="ld-step small opacity-75"
          style={{ minHeight: 22 }}
        >
          {steps.length ? steps[i % steps.length] : "Please wait..."}
        </div>
      )}
      <div className="ld-bars">
        {[0, 1, 2, 3, 4].map((n) => (
          <span key={n} />
        ))}
      </div>
      <div className="small opacity-50 mt-3">{t}s elapsed</div>
    </div>
  );
}
