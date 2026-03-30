import { useState, useRef } from "react";
import "./Detector.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── API ─────────────────────────────────────────────────────────────────────
// Ab sirf yeh rakhho:
async function analyzeText(text) {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

// ── Word highlighter ─────────────────────────────────────────────────────────
function HighlightedText({ text, fakeWords, realWords }) {
  if (!text || (!fakeWords.length && !realWords.length)) {
    return <p className="article-text">{text}</p>;
  }

  const fakeSet = new Set(fakeWords.map((w) => w.toLowerCase()));
  const realSet = new Set(realWords.map((w) => w.toLowerCase()));

  // Simple word-boundary highlight (stemmed words may not match exactly – best effort)
  const words = text.split(/(\s+)/);
  return (
    <p className="article-text">
      {words.map((segment, i) => {
        const clean = segment.replace(/[^a-zA-Z]/g, "").toLowerCase();
        if (fakeSet.has(clean))
          return (
            <mark key={i} className="mark-fake" title="Fake indicator">
              {segment}
            </mark>
          );
        if (realSet.has(clean))
          return (
            <mark key={i} className="mark-real" title="Real indicator">
              {segment}
            </mark>
          );
        return <span key={i}>{segment}</span>;
      })}
    </p>
  );
}

// ── Gauge ────────────────────────────────────────────────────────────────────
function ConfidenceGauge({ confidence, isFake }) {
  const pct = Math.round(confidence * 100);
  const angle = -90 + (confidence * 180);
  const color = isFake ? "var(--fake)" : "var(--real)";

  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 200 110" className="gauge-svg">
        {/* Track */}
        <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="var(--surface-2)" strokeWidth="14" strokeLinecap="round" />
        {/* Fill */}
        <path
          d="M20,100 A80,80 0 0,1 180,100"
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray="251.3"
          strokeDashoffset={251.3 - 251.3 * confidence}
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(.4,0,.2,1), stroke 0.4s" }}
        />
        {/* Needle */}
        <g transform={`rotate(${angle}, 100, 100)`} style={{ transition: "transform 1s cubic-bezier(.4,0,.2,1)" }}>
          <line x1="100" y1="100" x2="100" y2="28" stroke={color} strokeWidth="3" strokeLinecap="round" />
          <circle cx="100" cy="100" r="5" fill={color} />
        </g>
        {/* Labels */}
        <text x="14" y="115" className="gauge-label">FAKE</text>
        <text x="170" y="115" className="gauge-label">REAL</text>
      </svg>
      <div className="gauge-pct" style={{ color }}>{pct}%</div>
      <div className="gauge-sub">confidence</div>
    </div>
  );
}

// ── Influence bar ────────────────────────────────────────────────────────────
function InfluenceBar({ word, score, direction }) {
  const maxScore = 0.5;
  const width = Math.min((score / maxScore) * 100, 100);
  return (
    <div className="influence-row">
      <span className="influence-word">{word}</span>
      <div className="influence-track">
        <div
          className={`influence-fill ${direction}`}
          style={{ width: `${width}%`, transition: "width 0.8s cubic-bezier(.4,0,.2,1)" }}
        />
      </div>
      <span className={`influence-tag ${direction}`}>{direction}</span>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function Detector() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("words");
  const textRef = useRef(null);
  const [url, setUrl] = useState("");

  const handleAnalyze = async () => {
    if (!text.trim() && !url.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyzeText(text || "Article from URL", url || null);
      setResult(data);
      setActiveTab("words");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText("");
    setUrl("");
    setResult(null);
    setError(null);
    textRef.current?.focus();
  };

  const isFake = result?.prediction === "Fake";
  const fakeWords = result?.influential_words.filter((w) => w.direction === "fake").map((w) => w.word) || [];
  const realWords = result?.influential_words.filter((w) => w.direction === "real").map((w) => w.word) || [];

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⚠</span>
            <div>
              <h1 className="logo-title">TruthLens</h1>
              <p className="logo-sub">AI-Powered Fake News Detector</p>
            </div>
          </div>
          <div className="header-badge">
            <span className="badge-dot" />
            ML Model Active
          </div>
        </div>
      </header>

      <main className="main">
        {/* ── Input panel ── */}
        <section className="input-panel">
          <div className="panel-header">
            <h2>Analyze Article</h2>
          </div>

          <div className="textarea-wrap">
            <textarea
              ref={textRef}
              className="news-textarea"
              placeholder="Paste news article text here to analyze…&#10;&#10;The model will detect fake news patterns, highlight suspicious words, and explain its reasoning."
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={10}
            />
            <div className="textarea-meta">
              <span>{text.split(/\s+/).filter(Boolean).length} words</span>
              <span>{text.length} characters</span>
            </div>
          </div>

          <div className="action-row">
            <button className="btn-clear" onClick={handleClear} disabled={loading}>
              Clear
            </button>
            <button
              className="btn-analyze"
              onClick={handleAnalyze}
              disabled={loading || (!text.trim() && !url.trim())}
            >
              {loading ? (
                <span className="btn-loading">
                  <span className="spinner" />
                  Analyzing…
                </span>
              ) : (
                "🔍 Detect Fake News"
              )}
            </button>
          </div>

          {error && (
            <div className="error-banner">
              <strong>Error:</strong> {error}
            </div>
          )}
        </section>

        {/* ── Results panel ── */}
        {/* ── Results panel ── */}
        {result && (
          <section className={`results-panel ${isFake ? "is-fake" : "is-real"}`}>
            
            {/* Verdict row */}
            <div className="verdict-row">
              <div className="verdict-badge" data-verdict={result.prediction}>
                <span className="verdict-icon">{isFake ? "⚠" : "✓"}</span>
                <div>
                  <div className="verdict-label">VERDICT</div>
                  <div className="verdict-text">{result.prediction} News</div>
                </div>
              </div>
              <ConfidenceGauge confidence={result.confidence} isFake={isFake} />
              <div className="stats-mini">
                <div className="stat">
                  <span className="stat-val">{result.word_count}</span>
                  <span className="stat-key">words</span>
                </div>
                <div className="stat">
                  <span className="stat-val">{result.processing_time_ms}ms</span>
                  <span className="stat-key">processing</span>
                </div>
                <div className="stat">
                  <span className="stat-val">{Math.round(result.fake_probability * 100)}%</span>
                  <span className="stat-key fake-label">fake prob.</span>
                </div>
                <div className="stat">
                  <span className="stat-val">{Math.round(result.real_probability * 100)}%</span>
                  <span className="stat-key real-label">real prob.</span>
                </div>
              </div>
            </div>

            {/* ── Model Comparison ── */}
            {result.model_predictions &&
              Object.keys(result.model_predictions).length > 0 && (
                <div className="models-comparison">
                  <h3>🤖 Model Comparison</h3>
                  <div className="models-grid">
                    {Object.entries(result.model_predictions).map(([name, pred]) => {
                      const isFakePred = pred.prediction === "Fake";
                      const conf = Math.round(
                        Math.max(pred.fake_probability, pred.real_probability) * 100
                      );
                      return (
                        <div
                          key={name}
                          className={`model-card ${isFakePred ? "model-fake" : "model-real"}`}
                        >
                          <div className="model-name">{name}</div>
                          <div className="model-verdict">
                            {isFakePred ? "⚠" : "✓"} {pred.prediction}
                          </div>
                          <div className="model-conf">{conf}%</div>
                          <div className="model-bar-wrap">
                            <div
                              className={`model-bar ${isFakePred ? "fake" : "real"}`}
                              style={{ width: `${conf}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
            )}

            {/* Explanation */}
            <div className="explanation-box">
              <h3>💡 Explanation</h3>
              <p>{result.explanation}</p>
            </div>

            {/* Suspicious patterns */}
            {result.suspicious_patterns.length > 0 && (
              <div className="patterns-box">
                <h3>🚨 Suspicious Patterns Detected</h3>
                <div className="pattern-chips">
                  {result.suspicious_patterns.map((p, i) => (
                    <span key={i} className="pattern-chip">{p}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Tabs */}
            <div className="tabs">
              <button
                className={`tab ${activeTab === "words" ? "active" : ""}`}
                onClick={() => setActiveTab("words")}
              >
                Word Influence
              </button>
              <button
                className={`tab ${activeTab === "highlight" ? "active" : ""}`}
                onClick={() => setActiveTab("highlight")}
              >
                Highlighted Text
              </button>
            </div>

            {activeTab === "words" && (
              <div className="influence-section">
                <p className="influence-desc">
                  Top words influencing the prediction (red = fake indicator, green = real indicator):
                </p>
                <div className="influence-list">
                  {result.influential_words.map((w, i) => (
                    <InfluenceBar key={i} {...w} />
                  ))}
                </div>
              </div>
            )}

            {activeTab === "highlight" && (
              <div className="highlight-section">
                <div className="highlight-legend">
                  <span className="legend-fake">■ Fake indicator</span>
                  <span className="legend-real">■ Real indicator</span>
                </div>
                <div className="highlighted-article">
                  <HighlightedText text={text} fakeWords={fakeWords} realWords={realWords} />
                </div>
              </div>
            )}

          </section>
        )}
      </main>

      <footer className="footer">
        <p>TruthLens — For educational use only. Always verify news with multiple sources.</p>
      </footer>
    </div>
  );
}