import { useEffect, useState } from "react";
export default function Loader({ title = "Working", steps = [] }) {
  const [i, setI] = useState(0);
  const [t, setT] = useState(0);
  useEffect(() => {
    const a = setInterval(() => setI((x) => x + 1), 1800),
      b = setInterval(() => setT((x) => x + 1), 1000);
    return () => {
      clearInterval(a);
      clearInterval(b);
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
      <div
        key={i}
        className="ld-step small opacity-75"
        style={{ minHeight: 22 }}
      >
        {steps.length ? steps[i % steps.length] : "Please wait..."}
      </div>
      <div className="ld-bars">
        {[0, 1, 2, 3, 4].map((n) => (
          <span key={n} />
        ))}
      </div>
      <div className="small opacity-50 mt-3">{t}s elapsed</div>
    </div>
  );
}
