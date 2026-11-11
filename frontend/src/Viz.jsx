import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
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
const axis = { stroke: "#a5b4fc", fontSize: 11 };

export const MultiRoc = ({ curves }) => (
  <div style={{ height: 280 }}>
    <ResponsiveContainer>
      <ScatterChart margin={{ left: 8, bottom: 18 }}>
        <CartesianGrid stroke="#ffffff18" />
        <XAxis
          dataKey="x"
          type="number"
          domain={[0, 1]}
          {...axis}
          label={{
            value: "False positive rate",
            position: "insideBottom",
            offset: -10,
            fill: "#a5b4fc",
            fontSize: 11,
          }}
        />
        <YAxis
          dataKey="y"
          type="number"
          domain={[0, 1]}
          {...axis}
          width={45}
          label={{
            value: "True positive rate",
            angle: -90,
            position: "insideLeft",
            fill: "#a5b4fc",
            fontSize: 11,
          }}
        />
        <Tooltip
          contentStyle={{
            background: "#0b1020",
            border: "1px solid #ffffff33",
            borderRadius: 10,
          }}
        />
        <Legend />
        <ReferenceLine
          segment={[
            { x: 0, y: 0 },
            { x: 1, y: 1 },
          ]}
          stroke="#ffffff55"
          strokeDasharray="5 5"
        />
        {curves.map((c, i) => (
          <Scatter
            key={c.class}
            name={c.class}
            data={c.points}
            fill={COL[i % 8]}
            line={{ stroke: COL[i % 8], strokeWidth: 2 }}
            lineType="joint"
            shape={() => null}
            legendType="line"
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  </div>
);

export const Dendrogram = ({ t }) => {
  const xs = t.i.flat(),
    ys = t.d.flat();
  const x0 = Math.min(...xs),
    x1 = Math.max(...xs),
    y1 = Math.max(...ys) || 1;
  const W = 640,
    H = 260,
    pad = 10;
  const px = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
  const py = (v) => H - 18 - (v / y1) * (H - 30);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }}>
      {t.i.map((ic, k) => (
        <polyline
          key={k}
          fill="none"
          stroke="#a5b4fc"
          strokeWidth="1.2"
          points={ic.map((x, j) => `${px(x)},${py(t.d[k][j])}`).join(" ")}
        />
      ))}
      {t.cut != null && (
        <>
          <line
            x1={pad}
            x2={W - pad}
            y1={py(t.cut)}
            y2={py(t.cut)}
            stroke="#facc15"
            strokeDasharray="6 4"
          />
          <text
            x={W - pad}
            y={py(t.cut) - 4}
            textAnchor="end"
            fill="#facc15"
            fontSize="10"
          >
            cut for the chosen K
          </text>
        </>
      )}
    </svg>
  );
};

export const SilhouettePlot = ({ s }) => {
  const all = s.clusters.flatMap((c) => c.values);
  const lo = Math.min(-0.1, ...all),
    hi = Math.max(0.1, ...all);
  const W = 600,
    L = 46,
    rh = 0.9,
    gap = 12;
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - 10);
  let y = 6;
  const blocks = s.clusters.map((c, k) => {
    const top = y;
    y += c.values.length * rh + gap;
    return { c, top, k };
  });
  return (
    <svg viewBox={`0 0 ${W} ${y}`} style={{ width: "100%", maxHeight: 440 }}>
      {blocks.map(({ c, top, k }) => (
        <g key={c.c}>
          {c.values.map((v, j) => (
            <rect
              key={j}
              x={Math.min(x(0), x(v))}
              y={top + j * rh}
              width={Math.abs(x(v) - x(0))}
              height={rh + 0.2}
              fill={COL[k % 8]}
            />
          ))}
          <text
            x={L - 6}
            y={top + (c.values.length * rh) / 2 + 3}
            textAnchor="end"
            fill="#a5b4fc"
            fontSize="10"
          >
            {c.c}
          </text>
        </g>
      ))}
      <line
        x1={x(s.avg)}
        x2={x(s.avg)}
        y1={0}
        y2={y}
        stroke="#facc15"
        strokeDasharray="5 4"
      />
      <line x1={x(0)} x2={x(0)} y1={0} y2={y} stroke="#ffffff55" />
    </svg>
  );
};
