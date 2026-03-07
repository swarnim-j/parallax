"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

export default function Home() {
  const [sequence, setSequence] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScreenResult | null>(null);
  const [structure, setStructure] = useState<StructureState>({ status: "idle" });
  const structureRef = useRef<HTMLDivElement>(null);
  const activeRequestRef = useRef(0);

  useEffect(() => {
    const mount = structureRef.current;
    if (!mount) return;
    mount.innerHTML = "";

    if (structure.status !== "ready") return;

    let cancelled = false;
    let viewer: MolViewer | null = null;
    const hasMountSize = (el: HTMLDivElement) => el.clientWidth > 0 && el.clientHeight > 0;
    const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

    async function waitForMountSize(el: HTMLDivElement) {
      if (hasMountSize(el)) return true;
      const timeoutMs = 1800;
      const start = performance.now();
      while (!cancelled && performance.now() - start < timeoutMs) {
        await nextFrame();
        if (hasMountSize(el)) return true;
      }

      // Fallback for delayed stylesheets/HMR races where the mount starts at 0px height.
      if (!cancelled && !hasMountSize(el)) {
        el.style.minHeight = "320px";
        el.style.height = "320px";
      }
      return hasMountSize(el);
    }

    const timer = setTimeout(() => {
      ensureMolViewer()
        .then(async ($3Dmol) => {
          if (cancelled || !structureRef.current) return;
          const el = structureRef.current;
          const ready = await waitForMountSize(el);
          if (!ready || cancelled) return;

          viewer = $3Dmol.createViewer(el, { backgroundColor: "0xd0d5dc" });
          viewer.addModel(structure.data.pdb, "pdb");
          viewer.setStyle({}, { cartoon: { color: "spectrum" } });
          const maybeResize = viewer as MolViewer & { resize?: () => void };
          if (typeof maybeResize.resize === "function") {
            maybeResize.resize();
          }
          viewer.zoomTo();
          viewer.zoom(1.16, 500);
          viewer.spin("y", 0.45);
          viewer.render();
          requestAnimationFrame(() => {
            if (!cancelled && viewer) {
              const v = viewer as MolViewer & { resize?: () => void };
              if (typeof v.resize === "function") {
                v.resize();
              }
              viewer.zoomTo();
              viewer.render();
            }
          });
        })
        .catch((err) => {
          console.error("3Dmol error:", err);
          if (!cancelled) {
            setStructure({ status: "error", message: String(err) });
          }
        });
    }, 100);

    return () => {
      cancelled = true;
      clearTimeout(timer);
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
            <article className="structure-panel-full">
              <div className={`structure-shell-full ${structure.status}`}>
                <div className="structure-viewbox">
                  <div ref={structureRef} className="structure-stage-full" />
                </div>
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
              <div className="structure-head mono-text">
                <span>{structureSubtitle}</span>
                <span>{structureStatus}</span>
              </div>
            </article>
          </section>
        </div>
      </div>
    </div>
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
