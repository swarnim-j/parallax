"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Point = { x: number; y: number; label: string; name: string };
type ScreenHit = {
  hazard_name: string;
  embed_sim: number;
  seq_sim: number;
  differential: number;
  scale: string;
  start: number;
  end: number;
};
type ScreenResult = {
  risk_score: number;
  flagged: boolean;
  hits: ScreenHit[];
  explanation: string;
  input_type: string;
  proteins_screened: number;
};
type StructureResult = {
  input_type: "protein" | "dna";
  protein_length: number;
  protein_sequence: string;
  pdb: string;
  fold_source: string;
};
type StructureState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; data: StructureResult }
  | { status: "error"; message: string };

type MolViewer = {
  addModel: (pdb: string, format: string) => void;
  setStyle: (selection: Record<string, unknown>, style: Record<string, unknown>) => void;
  zoomTo: () => void;
  zoom: (factor: number, durationMs?: number) => void;
  spin: (axis: "x" | "y" | "z" | boolean, speed?: number) => void;
  render: () => void;
  clear: () => void;
};
type MolLibrary = {
  createViewer: (element: Element, options: { backgroundColor: string }) => MolViewer;
};

declare global {
  interface Window {
    $3Dmol?: MolLibrary;
  }
}

let molScriptPromise: Promise<MolLibrary> | null = null;

function ensureMolViewer() {
  if (typeof window === "undefined") return Promise.reject(new Error("Browser environment required"));
  if (window.$3Dmol) return Promise.resolve(window.$3Dmol);
  if (molScriptPromise) return molScriptPromise;

  molScriptPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js";
    script.async = true;
    script.onload = () => (window.$3Dmol ? resolve(window.$3Dmol) : reject(new Error("3Dmol loaded without global")));
    script.onerror = () => reject(new Error("Failed to load 3Dmol script"));
    document.head.appendChild(script);
  });

  return molScriptPromise;
}

const LEGEND = [
  { color: "#a34a4f", label: "Hazard" },
  { color: "#3f6751", label: "Benign" },
  { color: "#496891", label: "Query" },
];

export default function Home() {
  const [sequence, setSequence] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScreenResult | null>(null);
  const [structure, setStructure] = useState<StructureState>({ status: "idle" });
  const [points, setPoints] = useState<Point[]>([]);
  const [queryPoint, setQueryPoint] = useState<{ x: number; y: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const structureRef = useRef<HTMLDivElement>(null);
  const activeRequestRef = useRef(0);

  useEffect(() => {
    fetch(`${API}/api/embedding-space`)
      .then((r) => r.json())
      .then((d) => setPoints(d.points))
      .catch(() => {});
  }, []);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#d0d5dc";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "#bcc3cc";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 6; i += 1) {
      const gx = 26 + (i / 6) * (w - 52);
      const gy = 26 + (i / 6) * (h - 52);
      ctx.beginPath();
      ctx.moveTo(gx, 18);
      ctx.lineTo(gx, h - 18);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(18, gy);
      ctx.lineTo(w - 18, gy);
      ctx.stroke();
    }

    const all = [...points, ...(queryPoint ? [{ ...queryPoint, label: "query", name: "QUERY" }] : [])];
    if (all.length === 0) {
      ctx.fillStyle = "#6d7683";
      ctx.font = "11px var(--font-plex-mono), monospace";
      ctx.fillText("waiting for embedding space", w / 2 - 84, h / 2);
      return;
    }

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const point of all) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }

    const pad = 36;
    const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (w - 2 * pad);
    const sy = (y: number) => pad + ((y - minY) / (maxY - minY || 1)) * (h - 2 * pad);

    for (const point of points) {
      const x = sx(point.x);
      const y = sy(point.y);
      const color = point.label === "hazard" ? "#a34a4f" : "#3f6751";
      ctx.beginPath();
      ctx.arc(x, y, 4.7, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.76;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#6a7480";
      ctx.font = "10px var(--font-plex-mono), monospace";
      ctx.fillText(point.name.slice(0, 20), x + 8, y + 3);
    }

    if (queryPoint) {
      const x = sx(queryPoint.x);
      const y = sy(queryPoint.y);
      ctx.beginPath();
      ctx.arc(x, y, 14, 0, Math.PI * 2);
      ctx.strokeStyle = "#496891";
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.36;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#496891";
      ctx.fill();
      ctx.fillStyle = "#496891";
      ctx.font = "bold 11px var(--font-plex-mono), monospace";
      ctx.fillText("QUERY", x + 13, y + 3);
    }
  }, [points, queryPoint]);

  useEffect(() => {
    drawCanvas();
    window.addEventListener("resize", drawCanvas);
    return () => window.removeEventListener("resize", drawCanvas);
  }, [drawCanvas]);

  useEffect(() => {
    const mount = structureRef.current;
    if (!mount) return;
    mount.innerHTML = "";

    if (structure.status !== "ready") return;

    let cancelled = false;
    let viewer: MolViewer | null = null;

    ensureMolViewer()
      .then(($3Dmol) => {
        if (cancelled || !structureRef.current) return;

        viewer = $3Dmol.createViewer(structureRef.current, { backgroundColor: "rgba(0,0,0,0)" });
        viewer.addModel(structure.data.pdb, "pdb");
        viewer.setStyle({}, { cartoon: { colorscheme: "spectrum", opacity: 0.96 } });
        viewer.zoomTo();
        viewer.zoom(1.16, 500);
        viewer.spin("y", 0.45);
        viewer.render();
      })
      .catch(() => {
        if (!cancelled) {
          setStructure({ status: "error", message: "3D viewer failed to load in this browser." });
        }
      });

    return () => {
      cancelled = true;
      if (viewer) {
        try {
          viewer.spin(false);
          viewer.clear();
          viewer.render();
        } catch {}
      }
      if (mount) mount.innerHTML = "";
    };
  }, [structure]);

  async function loadStructure(sequenceText: string, requestId: number) {
    try {
      const resp = await fetch(`${API}/api/structure`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence: sequenceText }),
      });
      const payload = await resp.json().catch(() => null);

      if (requestId !== activeRequestRef.current) return;

      if (!resp.ok) {
        const message =
          payload && typeof payload.detail === "string"
            ? payload.detail
            : "Structure prediction is unavailable for this sequence.";
        setStructure({ status: "error", message });
        return;
      }

      setStructure({ status: "ready", data: payload as StructureResult });
    } catch {
      if (requestId === activeRequestRef.current) {
        setStructure({ status: "error", message: "Could not reach the folding service." });
      }
    }
  }

  async function handleScreen() {
    const cleaned = sequence.trim();
    if (!cleaned) return;

    const requestId = activeRequestRef.current + 1;
    activeRequestRef.current = requestId;

    setLoading(true);
    setResult(null);
    setQueryPoint(null);
    setStructure({ status: "loading" });
    void loadStructure(cleaned, requestId);

    try {
      const resp = await fetch(`${API}/api/screen`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence: cleaned }),
      });
      if (!resp.ok) {
        throw new Error("Screening request failed.");
      }
      const data = await resp.json();

      if (requestId !== activeRequestRef.current) return;
      setResult(data.result);
      setQueryPoint(data.query_point);
    } catch {
      if (requestId !== activeRequestRef.current) return;
      setStructure({ status: "idle" });
      alert("Screening failed. Is the API server running?");
    } finally {
      if (requestId === activeRequestRef.current) {
        setLoading(false);
      }
    }
  }

  const compactLength = useMemo(() => sequence.replace(/\s/g, "").length, [sequence]);
  const tone = result ? riskTone(result.risk_score) : "low";
  const structureStatus =
    structure.status === "loading"
      ? "[FOLDING]"
      : structure.status === "ready"
        ? "[READY]"
        : structure.status === "error"
          ? "[ERROR]"
          : "[IDLE]";
  const structureSubtitle =
    structure.status === "ready"
      ? `${structure.data.input_type.toUpperCase()} · ${structure.data.protein_length} aa`
      : "Predicted 3D Structure";

  return (
    <div className="parallax-site">
      <div className="parallax-frame">
        <header className="top-strip">
          <div className="brand-block">
            <BrandMark />
            <span className="brand-name">PARALLAX</span>
          </div>
        </header>

        <div className="split-layout">
          <section className="left-panel">
            <div className="intro-block">
              <p className="eyebrow mono-text">Biosecurity Screening Platform</p>
              <h1 className="hero-title">Parallax flags risky protein designs before they leave the model.</h1>
              <p className="hero-subtitle">
                Submit an amino acid or DNA sequence and inspect differential risk against known biological hazard space.
              </p>
            </div>

            <div className="stack-gap">
              <div className="rail-group">
                <article className="panel-card reveal">
                  <div className="card-head">
                    <span className="chip">Input</span>
                    <span className="mono-small mono-text">
                      {compactLength > 0 ? `${compactLength} residues` : "Protein or DNA sequence"}
                    </span>
                  </div>
                  <textarea
                    className="sequence-input"
                    placeholder="MVLSPADKTNVKAAWGKVGAHAGEYGAEALE..."
                    value={sequence}
                    onChange={(e) => setSequence(e.target.value)}
                  />
                  <button
                    onClick={handleScreen}
                    disabled={loading || !sequence.trim()}
                    className="screen-button"
                  >
                    {loading ? "Screening..." : "Screen Sequence"}
                  </button>
                </article>

                {result ? (
                  <article className={`panel-card score-panel reveal tone-${tone}`}>
                    <div className="card-head">
                      <span className="chip chip-tone">Risk Score</span>
                      <span className="mono-small mono-text">{result.flagged ? "[FLAGGED]" : "[PASS]"}</span>
                    </div>
                    <p className="score-value">{result.risk_score.toFixed(2)}</p>
                    <div className="score-label mono-text">
                      {tone === "high" ? "High Risk" : tone === "medium" ? "Needs Review" : "Low Risk"}
                    </div>
                  </article>
                ) : (
                  <article className="panel-card reveal">
                    <div className="card-head">
                      <span className="chip">Status</span>
                      <span className="mono-small mono-text">ready</span>
                    </div>
                    <p className="idle-copy">
                      Results and nearest-neighbor hazard comparisons will appear here after you screen a sequence.
                    </p>
                  </article>
                )}
                <div className="rail-tip" />
              </div>

              {result && (
                <>
                  <div className="stat-grid">
                    <StatBox
                      label="Seq Similarity"
                      value={result.hits[0]?.seq_sim}
                      tone={result.hits[0]?.seq_sim && result.hits[0].seq_sim > 0.3 ? "high" : "low"}
                    />
                    <StatBox
                      label="Embed Similarity"
                      value={result.hits[0]?.embed_sim}
                      tone={result.hits[0]?.embed_sim && result.hits[0].embed_sim > 0.7 ? "high" : "low"}
                    />
                  </div>

                  <article className="panel-card reveal">
                    <div className="card-head">
                      <span className="chip">Assessment</span>
                    </div>
                    <p className="explanation">{result.explanation}</p>
                  </article>

                  <article className="panel-card reveal">
                    <div className="card-head">
                      <span className="chip">Top Hits</span>
                      <span className="mono-small mono-text">{result.hits.length} returned</span>
                    </div>
                    <table className="hit-table">
                      <thead>
                        <tr>
                          <th>Hazard</th>
                          <th>Embed</th>
                          <th>Seq</th>
                          <th>Diff</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.hits.slice(0, 5).map((hit, i) => (
                          <tr key={i}>
                            <td>{hit.hazard_name}</td>
                            <td>{hit.embed_sim.toFixed(3)}</td>
                            <td>{hit.seq_sim.toFixed(3)}</td>
                            <td className={hit.differential > 0.5 ? "value-tone tone-high" : ""}>
                              {hit.differential.toFixed(3)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </article>

                  {result.hits.some((hit) => hit.scale === "window") && (
                    <article className="panel-card reveal">
                      <div className="card-head">
                        <span className="chip">Sequence Regions</span>
                      </div>
                      <div className="window-track">
                        {(() => {
                          const windows = result.hits.filter((hit) => hit.scale === "window");
                          const maxEnd = Math.max(...windows.map((hit) => hit.end));
                          return windows.map((windowHit, i) => (
                            <div
                              key={i}
                              className="window-segment"
                              style={{
                                left: `${(windowHit.start / maxEnd) * 100}%`,
                                width: `${((windowHit.end - windowHit.start) / maxEnd) * 100}%`,
                                background:
                                  windowHit.differential > 0.5
                                    ? "rgba(157, 76, 83, 0.78)"
                                    : windowHit.differential > 0.2
                                      ? "rgba(143, 117, 66, 0.72)"
                                      : "rgba(70, 107, 87, 0.74)",
                              }}
                              title={`${windowHit.hazard_name}: diff=${windowHit.differential.toFixed(3)}`}
                            />
                          ));
                        })()}
                      </div>
                    </article>
                  )}

                  <div className="meta-line mono-text">
                    {result.input_type.toUpperCase()} · {result.proteins_screened} protein(s) screened
                  </div>
                </>
              )}
            </div>
          </section>

          <section className="right-panel">
            <div className="wire-stage" aria-hidden="true">
              <WireBackdrop />
              <span className="wire-label label-a mono-text">[CYP-1]</span>
              <span className="wire-label label-b mono-text">[0.57]</span>
              <span className="wire-label label-c mono-text">[0.83]</span>
            </div>

            <div className="right-stack">
              <article className="structure-panel">
                <div className="map-head">
                  <span className="chip">Structure</span>
                  <span className="mono-small mono-text">{structureSubtitle}</span>
                  <span className="mono-small mono-text map-metric">{structureStatus}</span>
                </div>
                <div className={`structure-shell ${structure.status}`}>
                  <div ref={structureRef} className="structure-stage" />
                  {structure.status === "idle" && (
                    <div className="structure-overlay mono-text">Submit a sequence to predict a 3D fold.</div>
                  )}
                  {structure.status === "loading" && (
                    <div className="structure-overlay">
                      <div className="structure-spinner" />
                      <div className="mono-text">Folding protein model...</div>
                    </div>
                  )}
                  {structure.status === "error" && (
                    <div className="structure-overlay structure-error">{structure.message}</div>
                  )}
                </div>
                <div className="structure-foot mono-text">
                  {structure.status === "ready"
                    ? `${structure.data.fold_source} · auto-rotating model`
                    : "Real structure prediction powered by ESMFold"}
                </div>
              </article>

              <article className="map-panel">
                <div className="map-head">
                  <span className="chip">Map</span>
                  <span className="mono-small mono-text">Embedding Space (t-SNE)</span>
                  <span className="mono-small mono-text map-metric">
                    {result ? `[${result.risk_score.toFixed(2)}]` : "[--]"}
                  </span>
                </div>
                <div className="canvas-wrap">
                  <canvas ref={canvasRef} className="space-canvas" />
                  {points.length === 0 && !queryPoint && <div className="canvas-empty">loading embedding vectors</div>}
                </div>
                <div className="legend-row">
                  {LEGEND.map((item) => (
                    <div key={item.label} className="legend-item">
                      <div className="legend-dot" style={{ background: item.color }} />
                      {item.label}
                    </div>
                  ))}
                </div>
              </article>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function WireBackdrop() {
  return (
    <svg className="wire-svg" viewBox="0 0 1380 920" preserveAspectRatio="xMidYMid slice">
      <WireSphere cx={190} cy={900} r={400} />
      <WireSphere cx={975} cy={420} r={228} />
      <WireSphere cx={1105} cy={248} r={166} />
      <WireSphere cx={1198} cy={626} r={132} />
      <WireSphere cx={1090} cy={618} r={92} />
      <WireSphere cx={840} cy={282} r={104} />
      <WireSphere cx={920} cy={138} r={78} />
      <WireSphere cx={1268} cy={208} r={118} />
      <path d="M780 548 C 890 492, 1034 506, 1130 594 C 1200 658, 1210 740, 1160 784 C 1106 832, 1020 816, 954 760 C 872 688, 840 632, 780 548 Z" />
    </svg>
  );
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 34 24" role="presentation" focusable="false">
        <path className="mark-band mark-band-a" d="M3 6.2H30L27 9.5H0Z" />
        <path className="mark-band mark-band-b" d="M6.2 11.1H27.2L24.2 14.4H3.2Z" />
        <path className="mark-band mark-band-c" d="M9.4 16H24.4L21.4 19.3H6.4Z" />
      </svg>
    </div>
  );
}

function WireSphere({ cx, cy, r }: { cx: number; cy: number; r: number }) {
  return (
    <g transform={`translate(${cx} ${cy})`}>
      <circle r={r} />
      <ellipse rx={r} ry={r * 0.8} />
      <ellipse rx={r} ry={r * 0.58} />
      <ellipse rx={r} ry={r * 0.34} />
      <ellipse rx={r * 0.26} ry={r} />
      <ellipse rx={r * 0.48} ry={r} />
      <ellipse rx={r * 0.7} ry={r} />
      <ellipse rx={r * 0.88} ry={r} />
    </g>
  );
}

function riskTone(score: number) {
  if (score >= 0.5) return "high";
  if (score >= 0.2) return "medium";
  return "low";
}

function StatBox({ label, value, tone }: { label: string; value?: number; tone: "high" | "low" }) {
  return (
    <article className={`panel-card stat-box ${tone === "high" ? "tone-high" : "tone-low"}`}>
      <p className="stat-label">{label}</p>
      <p className={`stat-value ${tone === "high" ? "value-tone" : ""}`}>{value !== undefined ? value.toFixed(2) : "—"}</p>
    </article>
  );
}
