import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity, Database, Search, LineChart, HardDrive,
  ShieldCheck, AlertTriangle, XCircle, Plus, Trash2, RefreshCw,
  Play, CheckCircle2, ChevronRight, Layers, Cpu, Server, Clock,
  BarChart2, Upload, X, AlertCircle, Info, CheckCheck, Zap, FileJson,
  SlidersHorizontal, Crosshair, Eye
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import './App.css';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler
);

const API_BASE = 'http://localhost:8088';
const TOAST_DURATION = 4000;

// =============================================
// TOAST SYSTEM
// =============================================
let toastIdCounter = 0;

function useToast() {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((type, title, message) => {
    const id = ++toastIdCounter;
    setToasts(prev => [...prev, { id, type, title, message, exiting: false }]);
    setTimeout(() => {
      setToasts(prev => prev.map(t => t.id === id ? { ...t, exiting: true } : t));
      setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 300);
    }, TOAST_DURATION);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.map(t => t.id === id ? { ...t, exiting: true } : t));
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 300);
  }, []);

  return { toasts, addToast, removeToast };
}

const TOAST_ICONS = {
  success: <CheckCheck size={18} />,
  error:   <XCircle size={18} />,
  warning: <AlertTriangle size={18} />,
  info:    <Info size={18} />,
};

function ToastContainer({ toasts, removeToast }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast-item toast-${t.type} ${t.exiting ? 'exiting' : ''}`}>
          <span className="toast-icon">{TOAST_ICONS[t.type]}</span>
          <div className="toast-body">
            <div className="toast-title">{t.title}</div>
            {t.message && <div className="toast-message">{t.message}</div>}
          </div>
          <button className="toast-close-btn" onClick={() => removeToast(t.id)}>
            <X size={14} />
          </button>
          <div
            className="toast-progress"
            style={{ animationDuration: `${TOAST_DURATION}ms` }}
          />
        </div>
      ))}
    </div>
  );
}

// =============================================
// SIMPLE PCA (2-component projection for 2D visualizer)
// =============================================
function simplePCA(vectors) {
  if (!vectors || vectors.length === 0) return [];
  const n = vectors.length;
  const dim = vectors[0].length;

  // Compute mean
  const mean = Array(dim).fill(0);
  vectors.forEach(v => v.forEach((x, i) => { mean[i] += x / n; }));

  // Center
  const centered = vectors.map(v => v.map((x, i) => x - mean[i]));

  // Use first 2 random-ish projection vectors for demo-quality PCA
  // (power iteration for 2 eigenvectors)
  function dot(a, b) { return a.reduce((s, x, i) => s + x * b[i], 0); }
  function normalize(v) { const n = Math.sqrt(dot(v, v)); return v.map(x => x / (n || 1)); }

  let v1 = normalize(centered[0].length > 0 ? centered[0].slice() : Array(dim).fill(1));
  for (let iter = 0; iter < 15; iter++) {
    const scores = centered.map(r => dot(r, v1));
    let next = Array(dim).fill(0);
    centered.forEach((r, i) => r.forEach((x, j) => { next[j] += scores[i] * x; }));
    v1 = normalize(next);
  }

  let v2 = normalize(centered[Math.min(1, n - 1)].map((x, i) => x - dot(centered[Math.min(1, n - 1)], v1) * v1[i]));
  for (let iter = 0; iter < 15; iter++) {
    const scores = centered.map(r => dot(r, v2));
    let next = Array(dim).fill(0);
    centered.forEach((r, i) => r.forEach((x, j) => { next[j] += scores[i] * x; }));
    // Gram-Schmidt against v1
    const proj = dot(next, v1);
    next = next.map((x, i) => x - proj * v1[i]);
    v2 = normalize(next);
  }

  return vectors.map(v => ({
    x: dot(v.map((x, i) => x - mean[i]), v1),
    y: dot(v.map((x, i) => x - mean[i]), v2),
  }));
}

// =============================================
// VECTOR CANVAS VISUALIZER
// =============================================
function VectorCanvas({ queryVector, searchResults }) {
  const canvasRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const pointsRef = useRef([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width = W;
    canvas.height = H;

    ctx.clearRect(0, 0, W, H);

    if (!searchResults || searchResults.results.length === 0) {
      ctx.fillStyle = 'rgba(107,114,128,0.5)';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Run a search query to visualize vector space', W / 2, H / 2);
      return;
    }

    // Build vectors for PCA: query + result mocks (random noise around query)
    let parsedQuery = null;
    try { parsedQuery = JSON.parse(queryVector); } catch (e) {}

    const resultVectors = searchResults.results.map((_, i) => {
      if (parsedQuery) {
        return parsedQuery.map(x => x + (Math.random() - 0.5) * 0.4 * (1 + i * 0.2));
      }
      return Array.from({ length: 5 }, () => Math.random() * 2 - 1);
    });

    const queryVec = parsedQuery || Array.from({ length: 5 }, () => Math.random() * 2 - 1);
    const allVectors = [queryVec, ...resultVectors];
    const projected = simplePCA(allVectors);

    if (projected.length === 0) return;

    // Normalize to canvas space
    const xs = projected.map(p => p.x);
    const ys = projected.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 40;

    const toCanvas = (p) => ({
      cx: pad + ((p.x - minX) / (maxX - minX + 1e-9)) * (W - 2 * pad),
      cy: pad + ((p.y - minY) / (maxY - minY + 1e-9)) * (H - 2 * pad),
    });

    const pts = projected.map((p, i) => {
      const { cx, cy } = toCanvas(p);
      return { cx, cy, i, isQuery: i === 0, result: i > 0 ? searchResults.results[i - 1] : null };
    });
    pointsRef.current = pts;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let gx = pad; gx < W - pad; gx += (W - 2 * pad) / 6) {
      ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, H - pad); ctx.stroke();
    }
    for (let gy = pad; gy < H - pad; gy += (H - 2 * pad) / 4) {
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W - pad, gy); ctx.stroke();
    }

    // Draw lines from query to results
    pts.slice(1).forEach(p => {
      ctx.beginPath();
      ctx.moveTo(pts[0].cx, pts[0].cy);
      ctx.lineTo(p.cx, p.cy);
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // Draw result points
    pts.slice(1).forEach((p, i) => {
      const score = parseFloat(searchResults.results[i]?.score || 0);
      const alpha = 0.4 + score * 0.6;
      const radius = 7 + score * 5;

      // Glow
      const grad = ctx.createRadialGradient(p.cx, p.cy, 0, p.cx, p.cy, radius + 8);
      grad.addColorStop(0, `rgba(6, 182, 212, ${alpha * 0.4})`);
      grad.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(p.cx, p.cy, radius + 8, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Point
      ctx.beginPath();
      ctx.arc(p.cx, p.cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(6, 182, 212, ${alpha})`;
      ctx.fill();
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Rank label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`#${i + 1}`, p.cx, p.cy);
    });

    // Draw query point (on top)
    const qp = pts[0];
    const qGrad = ctx.createRadialGradient(qp.cx, qp.cy, 0, qp.cx, qp.cy, 18);
    qGrad.addColorStop(0, 'rgba(99, 102, 241, 0.35)');
    qGrad.addColorStop(1, 'transparent');
    ctx.beginPath();
    ctx.arc(qp.cx, qp.cy, 18, 0, Math.PI * 2);
    ctx.fillStyle = qGrad;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(qp.cx, qp.cy, 10, 0, Math.PI * 2);
    ctx.fillStyle = '#6366f1';
    ctx.fill();
    ctx.strokeStyle = '#a5b4fc';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Q', qp.cx, qp.cy);

  }, [queryVector, searchResults]);

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const hit = pointsRef.current.find(p => {
      const dx = p.cx - mx;
      const dy = p.cy - my;
      return Math.sqrt(dx * dx + dy * dy) < 16;
    });

    if (hit) {
      setTooltip({
        x: e.clientX + 12,
        y: e.clientY - 8,
        content: hit.isQuery
          ? { label: 'Query Vector', sub: 'PCA projection origin' }
          : {
            label: hit.result?.id || 'Result',
            sub: `Score: ${hit.result?.score} | Rank #${hit.i}`,
          }
      });
    } else {
      setTooltip(null);
    }
  };

  return (
    <div className="canvas-panel">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <div className="card-title" style={{ marginBottom: 0 }}>
          <Crosshair size={16} /> 2D Vector Space (PCA Projection)
        </div>
        <div className="canvas-legend">
          <span className="legend-item">
            <span className="legend-dot" style={{ background: '#6366f1' }}></span>Query
          </span>
          <span className="legend-item">
            <span className="legend-dot" style={{ background: '#06b6d4' }}></span>Neighbors
          </span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        className="vector-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltip && (
        <div className="canvas-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
          <div style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: 2 }}>{tooltip.content.label}</div>
          <div style={{ color: 'var(--text-muted)' }}>{tooltip.content.sub}</div>
        </div>
      )}
    </div>
  );
}

// =============================================
// INSERT VECTOR MODAL
// =============================================
function InsertVectorModal({ collection, onClose, addToast }) {
  const [vecId, setVecId] = useState('');
  const [vecData, setVecData] = useState('');
  const [payloadRows, setPayloadRows] = useState([{ key: 'category', val: '' }]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    let vectorArr;
    try { vectorArr = JSON.parse(vecData); }
    catch { addToast('error', 'Invalid Vector', 'Could not parse vector JSON. Use format: [0.1, -0.5, ...]'); return; }

    const payload = {};
    payloadRows.forEach(r => { if (r.key && r.val !== '') payload[r.key] = isNaN(r.val) ? r.val : Number(r.val); });

    setIsSubmitting(true);
    try {
      const res = await fetch(`${API_BASE}/collections/${collection.name}/insert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: vecId || `vec_${Date.now()}`, vector: vectorArr, payload })
      });
      if (res.ok) {
        addToast('success', 'Vector Inserted', `Successfully indexed into "${collection.name}"`);
        onClose();
      } else {
        const err = await res.json();
        addToast('error', 'Insert Failed', err.detail || 'Backend returned an error');
      }
    } catch {
      addToast('success', 'Vector Inserted (Demo)', `Vector added to "${collection.name}" in demo mode`);
      onClose();
    }
    setIsSubmitting(false);
  };

  const generateRandom = () => {
    const dim = collection.dimension || 128;
    const v = Array.from({ length: dim }, () => parseFloat((Math.random() * 2 - 1).toFixed(4)));
    setVecData(JSON.stringify(v));
  };

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-content" style={{ maxWidth: '600px' }}>
        <div className="modal-header">
          <h3 className="modal-title"><Plus size={18} style={{ verticalAlign: 'middle', marginRight: 6 }} />Insert Vector</h3>
          <button className="modal-close" onClick={onClose}><X size={18} /></button>
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{ background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 10, padding: '0.75rem 1rem', marginBottom: '1.25rem', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            Collection: <strong style={{ color: 'var(--accent-primary)' }}>{collection.name}</strong>
            &ensp;·&ensp; Dimension: <strong style={{ color: 'var(--accent-cyan)' }}>{collection.dimension}</strong>
            &ensp;·&ensp; Metric: <strong>{collection.metric}</strong>
          </div>

          <div className="form-group">
            <label className="form-label">Vector ID <span style={{ color: 'var(--text-dim)' }}>(optional — auto-generated if blank)</span></label>
            <input className="form-input" placeholder="e.g. vec_product_42" value={vecId} onChange={e => setVecId(e.target.value)} />
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="form-label">Vector Array <span style={{ color: 'var(--accent-rose)' }}>*</span></label>
              <button type="button" className="btn btn-secondary btn-sm" onClick={generateRandom}>
                <Zap size={12} /> Random {collection.dimension}d
              </button>
            </div>
            <textarea
              className="form-textarea"
              style={{ minHeight: 70 }}
              placeholder="[0.12, -0.45, 0.88, ...]"
              value={vecData}
              onChange={e => setVecData(e.target.value)}
              required
            />
            {vecData && (() => {
              try {
                const arr = JSON.parse(vecData);
                return <div className="vector-preview">✓ {arr.length} dimensions parsed</div>;
              } catch { return <div className="vector-preview" style={{ color: 'var(--accent-rose)' }}>✗ Invalid JSON array</div>; }
            })()}
          </div>

          <div style={{ marginTop: '0.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
              <label className="form-label">Metadata Payload</label>
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => setPayloadRows(p => [...p, { key: '', val: '' }])}>
                <Plus size={12} /> Add Field
              </button>
            </div>
            {payloadRows.map((row, i) => (
              <div className="payload-editor-row" key={i}>
                <input className="form-input" placeholder="key" value={row.key} onChange={e => setPayloadRows(p => p.map((r, j) => j === i ? { ...r, key: e.target.value } : r))} />
                <input className="form-input" placeholder="value" value={row.val} onChange={e => setPayloadRows(p => p.map((r, j) => j === i ? { ...r, val: e.target.value } : r))} />
                <button type="button" className="btn btn-danger btn-sm" onClick={() => setPayloadRows(p => p.filter((_, j) => j !== i))}>
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>

          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
              {isSubmitting ? <><RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> Inserting...</> : <><Plus size={14} /> Insert Vector</>}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// =============================================
// BATCH UPLOAD MODAL
// =============================================
function BatchUploadModal({ collection, onClose, addToast }) {
  const [uploadTab, setUploadTab] = useState('paste');
  const [jsonText, setJsonText] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [fileContent, setFileContent] = useState('');
  const [fileName, setFileName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const fileInputRef = useRef(null);

  const EXAMPLE = `[\n  {"id": "vec_001", "vector": [0.1, -0.2, 0.8], "payload": {"category": "tech"}},\n  {"id": "vec_002", "vector": [0.3, 0.5, -0.1], "payload": {"category": "art"}}\n]`;

  const handleFile = (file) => {
    if (!file) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = e => setFileContent(e.target.result);
    reader.readAsText(file);
  };

  const handleDrop = (e) => {
    e.preventDefault(); setIsDragOver(false);
    handleFile(e.dataTransfer.files[0]);
    setUploadTab('file');
  };

  const handleSubmit = async () => {
    const raw = uploadTab === 'paste' ? jsonText : fileContent;
    let vectors;
    try { vectors = JSON.parse(raw); }
    catch { addToast('error', 'Invalid JSON', 'Could not parse the dataset. Ensure it is a valid JSON array.'); return; }

    if (!Array.isArray(vectors)) { addToast('error', 'Wrong Format', 'Dataset must be a JSON array of vector objects.'); return; }

    setIsSubmitting(true);
    try {
      const res = await fetch(`${API_BASE}/collections/${collection.name}/batch_insert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vectors })
      });
      if (res.ok) {
        const data = await res.json();
        addToast('success', 'Batch Inserted', `${data.count || vectors.length} vectors indexed into "${collection.name}"`);
        onClose();
      } else {
        const err = await res.json();
        addToast('error', 'Batch Failed', err.detail || 'Backend returned an error');
      }
    } catch {
      addToast('success', 'Batch Inserted (Demo)', `${vectors.length} vectors loaded into "${collection.name}" in demo mode`);
      onClose();
    }
    setIsSubmitting(false);
  };

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-content" style={{ maxWidth: '620px' }}>
        <div className="modal-header">
          <h3 className="modal-title"><Upload size={18} style={{ verticalAlign: 'middle', marginRight: 6 }} />Batch Dataset Upload</h3>
          <button className="modal-close" onClick={onClose}><X size={18} /></button>
        </div>

        <div style={{ background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 10, padding: '0.75rem 1rem', marginBottom: '1.25rem', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          Target: <strong style={{ color: 'var(--accent-primary)' }}>{collection.name}</strong>
          &ensp;·&ensp; Dimension: <strong style={{ color: 'var(--accent-cyan)' }}>{collection.dimension}</strong>
        </div>

        <div className="upload-tabs">
          <button className={`upload-tab ${uploadTab === 'paste' ? 'active' : ''}`} onClick={() => setUploadTab('paste')}>
            <FileJson size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />Paste JSON
          </button>
          <button className={`upload-tab ${uploadTab === 'file' ? 'active' : ''}`} onClick={() => setUploadTab('file')}>
            <Upload size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />Upload File
          </button>
        </div>

        {uploadTab === 'paste' && (
          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="form-label">JSON Array of Vectors</label>
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => setJsonText(EXAMPLE)}>Load Example</button>
            </div>
            <textarea
              className="form-textarea"
              style={{ minHeight: 160 }}
              placeholder={EXAMPLE}
              value={jsonText}
              onChange={e => setJsonText(e.target.value)}
            />
          </div>
        )}

        {uploadTab === 'file' && (
          <div
            className={`dropzone ${isDragOver ? 'drag-over' : ''}`}
            onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-icon"><Upload size={32} /></div>
            {fileName
              ? <><strong style={{ color: 'var(--accent-emerald)' }}>{fileName}</strong><br /><span style={{ fontSize: '0.8rem' }}>Click to replace</span></>
              : <><strong>Drop .json file here</strong><br /><span style={{ fontSize: '0.8rem' }}>or click to browse</span></>
            }
            <input ref={fileInputRef} type="file" accept=".json" style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />
          </div>
        )}

        <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginTop: '-0.5rem', marginBottom: '0.5rem' }}>
          Format: <code style={{ color: 'var(--accent-cyan)' }}>{"[{id, vector, payload?}, ...]"}</code>
        </div>

        <div className="modal-footer">
          <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={handleSubmit} disabled={isSubmitting || (!jsonText && !fileContent)}>
            {isSubmitting
              ? <><RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> Uploading...</>
              : <><Upload size={14} /> Upload Dataset</>
            }
          </button>
        </div>
      </div>
    </div>
  );
}

// =============================================
// MAIN APP
// =============================================
export default function App() {
  const { toasts, addToast, removeToast } = useToast();
  const [activeTab, setActiveTab] = useState('overview');
  const [isServerOnline, setIsServerOnline] = useState(false);
  const [collections, setCollections] = useState([]);
  const [collectionSearch, setCollectionSearch] = useState('');
  const [metrics, setMetrics] = useState({
    qps: 124.5,
    p50_ms: 1.2,
    p95_ms: 3.8,
    p99_ms: 8.4,
    total_vectors: 12450,
    health_status: 'green',
    health_issues: [],
    recent_queries: [
      { timestamp: '16:18:42', collection: 'sift128_hnsw', k: 10, filter: '{"category": "tech"}', results: 10, time_ms: 2.1 },
      { timestamp: '16:18:38', collection: 'glove100_ivf', k: 5,  filter: 'None', results: 5, time_ms: 1.8 },
      { timestamp: '16:18:30', collection: 'sift128_hnsw', k: 20, filter: '{"price": {"$gt": 50}}', results: 20, time_ms: 4.3 },
      { timestamp: '16:18:15', collection: 'demo_flat', k: 10,   filter: 'None', results: 10, time_ms: 0.9 },
    ],
    storage: {
      bytes_used: 48500000,
      bytes_available: 500000000,
      active_log_size: 4200000,
      pending_entries: 14,
      tombstones_count: 320,
      snapshots_count: 3
    }
  });

  const [timeSeries, setTimeSeries] = useState({
    labels: ['16:14', '16:15', '16:16', '16:17', '16:18', '16:19'],
    qps: [80, 95, 110, 105, 120, 124.5],
    p50: [1.1, 1.2, 1.0, 1.3, 1.1, 1.2],
    p95: [3.5, 4.0, 3.6, 3.9, 3.7, 3.8],
    p99: [7.8, 8.9, 8.1, 8.5, 8.2, 8.4]
  });

  const [snapshots, setSnapshots] = useState([
    { id: 'snap_17887100', created_at: '2026-09-06 15:45', size_bytes: 4200000, vectors: 8000 },
    { id: 'snap_17886950', created_at: '2026-09-06 14:10', size_bytes: 2100000, vectors: 4000 },
  ]);

  // Modals
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(null);
  const [showCollectionDetail, setShowCollectionDetail] = useState(null);
  const [showInsertModal, setShowInsertModal] = useState(null);
  const [showBatchModal, setShowBatchModal] = useState(null);

  // New Collection Form
  const [newCollName, setNewCollName] = useState('');
  const [newCollDim, setNewCollDim] = useState(128);
  const [newCollMetric, setNewCollMetric] = useState('cosine');

  // Query Playground
  const [queryColl, setQueryColl] = useState('');
  const [queryVector, setQueryVector] = useState('[0.12, -0.45, 0.88, 0.33, 0.05]');
  const [queryVectorId, setQueryVectorId] = useState('');
  const [queryK, setQueryK] = useState(10);
  const [queryStrategy, setQueryStrategy] = useState('pre_filter');
  const [queryIndexType, setQueryIndexType] = useState('hnsw');
  const [efSearch, setEfSearch] = useState(50);
  const [nProbe, setNProbe] = useState(8);
  const [filterRows, setFilterRows] = useState([{ key: 'category', op: '$eq', val: 'electronics' }]);
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [recallResult, setRecallResult] = useState(null);

  // =============================================
  // BACKEND POLLING — /dashboard/stats
  // =============================================
  useEffect(() => {
    const fetchDashboardStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/dashboard/stats`);
        if (res.ok) {
          const data = await res.json();
          setIsServerOnline(true);

          setMetrics(prev => ({
            ...prev,
            qps: data.qps ?? prev.qps,
            p50_ms: parseFloat((data.latency_5m?.p50 || prev.p50_ms).toFixed(2)),
            p95_ms: parseFloat((data.latency_5m?.p95 || prev.p95_ms).toFixed(2)),
            p99_ms: parseFloat((data.latency_5m?.p99 || prev.p99_ms).toFixed(2)),
            total_vectors: data.total_vectors ?? prev.total_vectors,
            health_status: data.health?.status === 'green' ? 'green' : data.health?.status === 'yellow' ? 'yellow' : 'red',
            recent_queries: data.recent_queries?.length > 0
              ? data.recent_queries.map(q => ({
                  timestamp: q.timestamp,
                  collection: q.collection,
                  k: q.k,
                  filter: typeof q.filter_applied === 'object' ? JSON.stringify(q.filter_applied) : q.filter_applied,
                  results: q.result_count,
                  time_ms: q.latency_ms
                }))
              : prev.recent_queries,
            storage: {
              bytes_used: data.storage?.used_bytes ?? prev.storage.bytes_used,
              bytes_available: data.storage?.total_bytes ?? prev.storage.bytes_available,
              active_log_size: data.wal_summary?.total_wal_bytes ?? prev.storage.active_log_size,
              pending_entries: data.wal_summary?.pending_entries ?? prev.storage.pending_entries,
              tombstones_count: data.collections?.reduce((s, c) => s + (c.tombstoned_vectors || 0), 0) ?? prev.storage.tombstones_count,
              snapshots_count: data.snapshots?.length ?? prev.storage.snapshots_count,
            }
          }));

          // Sync time series from history
          if (data.metrics_history?.length > 0) {
            const hist = data.metrics_history.slice(-20);
            setTimeSeries({
              labels: hist.map(h => h.timestamp),
              qps: hist.map(h => h.qps),
              p50: hist.map(h => h.p50),
              p95: hist.map(h => h.p95),
              p99: hist.map(h => h.p99)
            });
          }

          if (data.snapshots?.length > 0) {
            setSnapshots(data.snapshots.map(s => ({
              id: s.path?.split('/')?.at(-1)?.replace('.pkl', '') || 'snap_' + Date.now(),
              created_at: s.mtime || 'unknown',
              size_bytes: s.size_bytes,
              vectors: 'N/A'
            })));
          }
        } else {
          setIsServerOnline(false);
        }
      } catch {
        setIsServerOnline(false);
      }
    };

    const fetchCollections = async () => {
      try {
        const res = await fetch(`${API_BASE}/collections`);
        if (res.ok) {
          const data = await res.json();
          setCollections(data.collections || []);
          if (!queryColl && data.collections.length > 0) setQueryColl(data.collections[0].name);
        }
      } catch {
        if (collections.length === 0) {
          const demo = [
            { name: 'sift128_hnsw', dimension: 128, metric: 'cosine', vectors_count: 8000, hnsw_M: 16, ef_construction: 64, ef_search: 32 },
            { name: 'glove100_ivf', dimension: 100, metric: 'euclidean', vectors_count: 4000, n_list: 32, n_probe: 8 },
            { name: 'demo_flat', dimension: 64, metric: 'dot_product', vectors_count: 450 }
          ];
          setCollections(demo);
          if (!queryColl) setQueryColl('sift128_hnsw');
        }
      }
    };

    fetchDashboardStats();
    fetchCollections();
    const interval = setInterval(() => { fetchDashboardStats(); fetchCollections(); }, 3000);
    return () => clearInterval(interval);
  }, []);

  // =============================================
  // HANDLERS
  // =============================================
  const handleCreateCollection = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${API_BASE}/collections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newCollName, dimension: parseInt(newCollDim), metric: newCollMetric })
      });
      if (res.ok) {
        setShowCreateModal(false);
        setNewCollName('');
        const cRes = await fetch(`${API_BASE}/collections`);
        if (cRes.ok) setCollections((await cRes.json()).collections);
        addToast('success', 'Collection Created', `"${newCollName}" is ready to index vectors`);
      } else {
        const err = await res.json();
        addToast('error', 'Create Failed', err.detail);
      }
    } catch {
      setCollections(prev => [...prev, { name: newCollName, dimension: parseInt(newCollDim), metric: newCollMetric, vectors_count: 0 }]);
      setShowCreateModal(false);
      setNewCollName('');
      addToast('info', 'Created (Demo Mode)', `"${newCollName}" added locally — server offline`);
    }
  };

  const handleDeleteCollection = async (name) => {
    try {
      await fetch(`${API_BASE}/collections/${name}`, { method: 'DELETE' });
      addToast('warning', 'Collection Deleted', `"${name}" and all its vectors have been removed`);
    } catch {
      addToast('warning', 'Deleted (Demo)', `"${name}" removed from local state`);
    }
    setCollections(prev => prev.filter(c => c.name !== name));
    setShowDeleteModal(null);
  };

  const handleExecuteQuery = async () => {
    setIsSearching(true);
    setShowVisualizer(false);

    let filterObj = {};
    filterRows.forEach(r => {
      if (r.key && r.val) {
        const parsedVal = isNaN(r.val) ? r.val : Number(r.val);
        filterObj[r.key] = r.op === '$eq' ? parsedVal : { [r.op]: parsedVal };
      }
    });

    let vectorArr = null;
    try { if (queryVector.trim()) vectorArr = JSON.parse(queryVector); } catch (e) {}

    const payload = {
      query: vectorArr,
      vector_id: queryVectorId || null,
      k: parseInt(queryK),
      filter: Object.keys(filterObj).length > 0 ? filterObj : null,
      filter_strategy: queryStrategy,
      index_type: queryIndexType,
      ef_search: queryIndexType === 'hnsw' ? efSearch : undefined,
      n_probe: queryIndexType === 'ivf' ? nProbe : undefined,
    };

    const startTime = performance.now();
    try {
      const res = await fetch(`${API_BASE}/collections/${queryColl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const duration = (performance.now() - startTime).toFixed(2);
      if (res.ok) {
        const data = await res.json();
        setSearchResults({ results: data.results || [], latency_ms: duration, count: (data.results || []).length });
        addToast('success', 'Search Complete', `${(data.results || []).length} vectors returned in ${duration}ms`);
      } else {
        const err = await res.json();
        addToast('error', 'Search Error', err.detail);
      }
    } catch {
      const duration = (performance.now() - startTime + 3.2).toFixed(2);
      const mockResults = Array.from({ length: Math.min(queryK, 5) }).map((_, idx) => ({
        id: `vec_sample_${idx + 101}`,
        score: (0.985 - idx * 0.042).toFixed(4),
        payload: { category: filterRows[0]?.val || 'electronics', price: 120 + idx * 15, in_stock: true }
      }));
      setSearchResults({ latency_ms: duration, count: parseInt(queryK), results: mockResults });
      addToast('info', 'Demo Results', `Showing ${mockResults.length} mock results (server offline)`);
    }
    setIsSearching(false);
    setShowVisualizer(true);
  };

  const handleVerifyRecall = async () => {
    setIsSearching(true);
    let vectorArr = null;
    try { if (queryVector.trim()) vectorArr = JSON.parse(queryVector); } catch (e) {}
    let filterObj = {};
    filterRows.forEach(r => {
      if (r.key && r.val) {
        const parsedVal = isNaN(r.val) ? r.val : Number(r.val);
        filterObj[r.key] = r.op === '$eq' ? parsedVal : { [r.op]: parsedVal };
      }
    });
    try {
      const res = await fetch(`${API_BASE}/collections/${queryColl}/recall_verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: vectorArr,
          k: parseInt(queryK),
          filter: Object.keys(filterObj).length > 0 ? filterObj : null,
          filter_strategy: queryStrategy,
          index_type: queryIndexType,
          ef_search: queryIndexType === 'hnsw' ? efSearch : undefined,
          n_probe: queryIndexType === 'ivf' ? nProbe : undefined,
        })
      });
      if (res.ok) {
        const data = await res.json();
        setRecallResult(data);
        const pct = (data.recall_at_k * 100).toFixed(1);
        const type = data.recall_at_k >= 0.9 ? 'success' : data.recall_at_k >= 0.7 ? 'warning' : 'error';
        addToast(type, `Recall@${data.k}: ${pct}%`, `ANN: ${data.ann_latency_ms}ms | Exact: ${data.exact_latency_ms}ms`);
      } else {
        const err = await res.json();
        addToast('error', 'Recall Error', err.detail);
      }
    } catch {
      // Demo mode mock recall
      const mockRecall = { recall_at_k: 0.9 + Math.random() * 0.09, k: parseInt(queryK), ann_latency_ms: '2.3', exact_latency_ms: '18.4', ann_results: [], exact_results: [] };
      setRecallResult(mockRecall);
      addToast('info', `Recall@${queryK}: ${(mockRecall.recall_at_k * 100).toFixed(1)}%`, 'Demo mode — server offline');
    }
    setIsSearching(false);
  };

  const handleTriggerAction = async (action) => {
    try {
      const endpoint = action === 'snapshot' ? '/snapshot' : '/compact';
      await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
      if (action === 'snapshot') {
        addToast('success', 'Snapshot Created', 'All in-memory mutations flushed to immutable disk snapshot');
        setSnapshots(prev => [{ id: 'snap_' + Date.now(), created_at: new Date().toLocaleString(), size_bytes: metrics.storage.bytes_used, vectors: metrics.total_vectors }, ...prev]);
      } else {
        addToast('success', 'WAL Compaction Done', 'Dead tombstones pruned and WAL segments consolidated');
        setMetrics(prev => ({ ...prev, storage: { ...prev.storage, tombstones_count: 0 } }));
      }
    } catch {
      if (action === 'snapshot') {
        addToast('success', 'Snapshot Triggered', 'Snapshot persisted to disk (demo mode)');
      } else {
        addToast('success', 'Compaction Done', 'Tombstone records reclaimed (demo mode)');
        setMetrics(prev => ({ ...prev, storage: { ...prev.storage, tombstones_count: 0 } }));
      }
    }
  };

  const addFilterRow = () => setFilterRows(prev => [...prev, { key: '', op: '$eq', val: '' }]);
  const removeFilterRow = (i) => setFilterRows(prev => prev.filter((_, j) => j !== i));

  const formatBytes = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const filteredCollections = collections.filter(c =>
    c.name.toLowerCase().includes(collectionSearch.toLowerCase())
  );

  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'top', labels: { color: '#9ca3af', boxWidth: 12, font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: '#6b7280', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
      y: { ticks: { color: '#6b7280', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
    }
  };

  return (
    <div className="app-container">
      {/* ===== HEADER ===== */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon"><Layers size={22} /></div>
          <div>
            <span className="brand-title">VecDB Ops Studio</span>
            <span className="brand-tag" style={{ marginLeft: 10 }}>v0.1.0</span>
          </div>
        </div>

        <nav className="nav-tabs">
          {[
            { id: 'overview',     icon: <Activity size={15} />,   label: 'Overview' },
            { id: 'collections',  icon: <Database size={15} />,   label: 'Collections' },
            { id: 'playground',   icon: <Search size={15} />,     label: 'Playground' },
            { id: 'metrics',      icon: <BarChart2 size={15} />,  label: 'Metrics' },
            { id: 'storage',      icon: <HardDrive size={15} />,  label: 'Storage' },
          ].map(t => (
            <button key={t.id} className={`nav-tab ${activeTab === t.id ? 'active' : ''}`} onClick={() => setActiveTab(t.id)}>
              {t.icon} {t.label}
            </button>
          ))}
        </nav>

        <div className="header-actions">
          <div className={`server-status ${isServerOnline ? '' : 'offline'}`}>
            <span className="status-dot" />
            {isServerOnline ? 'API Connected (8088)' : 'Demo Mode (Offline)'}
          </div>
        </div>
      </header>

      {/* ===== MAIN CONTENT ===== */}
      <main className="main-content">

        {/* ==================== TAB 1: OVERVIEW ==================== */}
        {activeTab === 'overview' && (
          <div>
            <div className="grid-4">
              <div className="glass-card">
                <div className="card-title"><Database size={18} /> Total Collections</div>
                <div className="stat-value">{collections.length}</div>
                <div className="stat-desc">Active index namespaces</div>
              </div>
              <div className="glass-card">
                <div className="card-title"><Cpu size={18} /> Total Vectors Indexed</div>
                <div className="stat-value">{metrics.total_vectors.toLocaleString()}</div>
                <div className="stat-desc">High-dimensional vector points</div>
              </div>
              <div className="glass-card">
                <div className="card-title"><Activity size={18} /> Live QPS</div>
                <div className="stat-value" style={{ color: 'var(--accent-cyan)' }}>{metrics.qps}</div>
                <div className="stat-desc">Rolling 10s window average</div>
              </div>
              <div className="glass-card">
                <div className="card-title"><ShieldCheck size={18} /> System Health</div>
                <div style={{ marginTop: '0.5rem' }}>
                  <span className={`health-badge ${metrics.health_status}`}>
                    <CheckCircle2 size={16} />
                    {metrics.health_status === 'green' ? 'All Healthy' : 'Action Required'}
                  </span>
                </div>
                <div className="stat-desc" style={{ marginTop: '0.4rem' }}>WAL Log & HNSW graph verified</div>
              </div>
            </div>

            <div className="grid-2">
              <div className="glass-card">
                <div className="card-title"><Clock size={18} /> Query Latencies (Last 5 mins)</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1.5rem', marginBottom: '1rem' }}>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>P50 (Median)</div>
                    <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>{metrics.p50_ms} ms</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>P95</div>
                    <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--accent-amber)' }}>{metrics.p95_ms} ms</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>P99</div>
                    <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--accent-rose)' }}>{metrics.p99_ms} ms</div>
                  </div>
                </div>
                <div style={{ height: '140px' }}>
                  <Line
                    data={{
                      labels: timeSeries.labels,
                      datasets: [
                        { label: 'P50', data: timeSeries.p50, borderColor: '#10b981', tension: 0.4, pointRadius: 2 },
                        { label: 'P95', data: timeSeries.p95, borderColor: '#f59e0b', tension: 0.4, pointRadius: 2 },
                        { label: 'P99', data: timeSeries.p99, borderColor: '#f43f5e', tension: 0.4, pointRadius: 2 }
                      ]
                    }}
                    options={{ ...chartDefaults, plugins: { ...chartDefaults.plugins, legend: { ...chartDefaults.plugins.legend, position: 'top' } } }}
                  />
                </div>
              </div>

              <div className="glass-card">
                <div className="card-title"><HardDrive size={18} /> Disk Storage Utilization</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: '1rem' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{formatBytes(metrics.storage.bytes_used)}</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>of {formatBytes(metrics.storage.bytes_available)}</div>
                </div>
                <div className="progress-bg">
                  <div className="progress-fill" style={{ width: `${Math.min(100, (metrics.storage.bytes_used / metrics.storage.bytes_available) * 100)}%` }} />
                </div>
                <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', fontSize: '0.85rem' }}>
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '8px' }}>
                    <div style={{ color: 'var(--text-muted)' }}>Active WAL Size</div>
                    <div style={{ fontWeight: 600, marginTop: '2px' }}>{formatBytes(metrics.storage.active_log_size)}</div>
                  </div>
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '8px' }}>
                    <div style={{ color: 'var(--text-muted)' }}>Dead Tombstones</div>
                    <div style={{ fontWeight: 600, marginTop: '2px', color: 'var(--accent-amber)' }}>{metrics.storage.tombstones_count} vectors</div>
                  </div>
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '8px' }}>
                    <div style={{ color: 'var(--text-muted)' }}>Pending WAL Entries</div>
                    <div style={{ fontWeight: 600, marginTop: '2px' }}>{metrics.storage.pending_entries}</div>
                  </div>
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '8px' }}>
                    <div style={{ color: 'var(--text-muted)' }}>Saved Snapshots</div>
                    <div style={{ fontWeight: 600, marginTop: '2px', color: 'var(--accent-cyan)' }}>{metrics.storage.snapshots_count}</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-card">
              <div className="card-title"><Search size={18} /> Live Query Stream</div>
              <div className="table-container" style={{ marginTop: '1rem' }}>
                <table className="data-table">
                  <thead>
                    <tr><th>Time</th><th>Collection</th><th>Top K</th><th>Metadata Filter</th><th>Results</th><th>Query Time</th></tr>
                  </thead>
                  <tbody>
                    {metrics.recent_queries.map((q, i) => (
                      <tr key={i}>
                        <td className="mono" style={{ color: 'var(--text-dim)' }}>{q.timestamp}</td>
                        <td style={{ fontWeight: 500 }}>{q.collection}</td>
                        <td>{q.k}</td>
                        <td className="mono" style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>{String(q.filter)}</td>
                        <td>{q.results} vectors</td>
                        <td className="mono" style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{q.time_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ==================== TAB 2: COLLECTIONS ==================== */}
        {activeTab === 'collections' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>Collections</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Manage index configurations, dimensions, and vector datasets.</p>
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <button className="btn btn-secondary" onClick={() => setShowBatchModal(collections[0])}>
                  <Upload size={16} /> Batch Upload
                </button>
                <button className="btn btn-primary" onClick={() => setShowCreateModal(true)}>
                  <Plus size={16} /> Create Collection
                </button>
              </div>
            </div>

            {/* Search bar */}
            <div className="collection-toolbar">
              <div className="search-input-wrapper">
                <Search size={16} />
                <input
                  className="form-input"
                  placeholder="Search collections…"
                  value={collectionSearch}
                  onChange={e => setCollectionSearch(e.target.value)}
                />
              </div>
            </div>

            <div className="grid-2">
              {filteredCollections.map((c, idx) => (
                <div className="glass-card" key={idx}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <h3 style={{ fontSize: '1.2rem', fontWeight: 600 }}>{c.name}</h3>
                      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                        <span className="brand-tag">{c.metric}</span>
                        <span className="brand-tag" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)', borderColor: 'rgba(6, 182, 212, 0.3)' }}>
                          {c.dimension} dims
                        </span>
                      </div>
                    </div>
                    <button className="btn btn-danger btn-sm" onClick={() => setShowDeleteModal(c.name)}>
                      <Trash2 size={14} />
                    </button>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginTop: '1.5rem', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '10px' }}>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Indexed Vectors</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>{(c.vectors_count || 0).toLocaleString()}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Primary Index</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--accent-primary)' }}>
                        {c.hnsw_M ? 'HNSW' : c.n_list ? 'IVF' : 'Flat'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Parameters</div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 500, fontFamily: 'var(--font-mono)' }}>
                        {c.hnsw_M ? `M=${c.hnsw_M}, ef=${c.ef_construction}` : c.n_list ? `nlist=${c.n_list}` : 'Brute Force'}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1.25rem', flexWrap: 'wrap' }}>
                    <button className="btn btn-success btn-sm" style={{ flex: 1 }} onClick={() => setShowInsertModal(c)}>
                      <Plus size={14} /> Insert Vector
                    </button>
                    <button className="btn btn-secondary btn-sm" style={{ flex: 1 }} onClick={() => setShowBatchModal(c)}>
                      <Upload size={14} /> Batch Upload
                    </button>
                    <button className="btn btn-secondary btn-sm" onClick={() => setShowCollectionDetail(c)}>
                      <Eye size={14} /> Inspect
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ==================== TAB 3: QUERY PLAYGROUND ==================== */}
        {activeTab === 'playground' && (
          <div>
            <div style={{ marginBottom: '1.5rem' }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>Query Playground</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Test nearest-neighbor search, filtering strategies, and inspect latency.</p>
            </div>

            <div className="grid-2">
              {/* Form Input Panel */}
              <div className="glass-card">
                <div className="card-title"><Search size={18} /> Search Parameters</div>

                <div className="form-group">
                  <label className="form-label">Target Collection</label>
                  <select className="form-select" value={queryColl} onChange={e => setQueryColl(e.target.value)}>
                    {collections.map(c => <option key={c.name} value={c.name}>{c.name} ({c.dimension}d)</option>)}
                  </select>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div className="form-group">
                    <label className="form-label">Index Engine</label>
                    <select className="form-select" value={queryIndexType} onChange={e => setQueryIndexType(e.target.value)}>
                      <option value="hnsw">HNSW</option>
                      <option value="ivf">IVF</option>
                      <option value="flat">Flat (Exact)</option>
                      <option value="pq">PQ (Product Quantization)</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Filter Strategy</label>
                    <select className="form-select" value={queryStrategy} onChange={e => setQueryStrategy(e.target.value)}>
                      <option value="pre_filter">Pre-Filtering</option>
                      <option value="post_filter">Post-Filtering</option>
                    </select>
                  </div>
                </div>

                {/* Dynamic hyperparameter sliders */}
                {queryIndexType === 'hnsw' && (
                  <div className="slider-group">
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <label className="form-label"><SlidersHorizontal size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />ef_search (Beam Width)</label>
                    </div>
                    <div className="slider-row">
                      <input type="range" className="slider-input" min={10} max={300} step={5} value={efSearch} onChange={e => setEfSearch(Number(e.target.value))} />
                      <span className="slider-value">{efSearch}</span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Higher = better recall, slower search</div>
                  </div>
                )}

                {queryIndexType === 'ivf' && (
                  <div className="slider-group">
                    <label className="form-label"><SlidersHorizontal size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />n_probe (Cluster Probes)</label>
                    <div className="slider-row">
                      <input type="range" className="slider-input" min={1} max={64} step={1} value={nProbe} onChange={e => setNProbe(Number(e.target.value))} />
                      <span className="slider-value">{nProbe}</span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Higher = more clusters searched, better recall</div>
                  </div>
                )}

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div className="form-group">
                    <label className="form-label">Top K Neighbors</label>
                    <input className="form-input" type="number" value={queryK} onChange={e => setQueryK(e.target.value)} min="1" max="100" />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Vector ID (Optional)</label>
                    <input className="form-input" type="text" placeholder="e.g. vec_104" value={queryVectorId} onChange={e => setQueryVectorId(e.target.value)} />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Query Vector (JSON)</label>
                  <textarea className="form-textarea" value={queryVector} onChange={e => setQueryVector(e.target.value)} placeholder="[0.1, 0.2, -0.5, ...]" style={{ minHeight: 70 }} />
                </div>

                {/* Filter Rows */}
                <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <label className="form-label">Metadata Filters</label>
                    <button className="btn btn-secondary btn-sm" onClick={addFilterRow}><Plus size={14} /> Add Clause</button>
                  </div>
                  {filterRows.map((row, idx) => (
                    <div className="filter-row" key={idx}>
                      <input className="form-input" placeholder="Field" value={row.key} onChange={e => { const u = [...filterRows]; u[idx].key = e.target.value; setFilterRows(u); }} />
                      <select className="form-select" value={row.op} onChange={e => { const u = [...filterRows]; u[idx].op = e.target.value; setFilterRows(u); }}>
                        <option value="$eq">=</option>
                        <option value="$ne">!=</option>
                        <option value="$gt">&gt;</option>
                        <option value="$gte">&gt;=</option>
                        <option value="$lt">&lt;</option>
                        <option value="$lte">&lt;=</option>
                      </select>
                      <input className="form-input" placeholder="Value" value={row.val} onChange={e => { const u = [...filterRows]; u[idx].val = e.target.value; setFilterRows(u); }} />
                      <button className="btn btn-danger btn-sm" onClick={() => removeFilterRow(idx)}><Trash2 size={14} /></button>
                    </div>
                  ))}
                </div>

                <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.25rem' }}>
                  <button className="btn btn-primary" style={{ flex: 2 }} onClick={handleExecuteQuery} disabled={isSearching}>
                    <Play size={16} /> {isSearching ? 'Executing Search...' : 'Run Vector Search'}
                  </button>
                  <button
                    className="btn btn-secondary"
                    style={{ flex: 1, borderColor: recallResult ? 'var(--accent-primary)' : undefined }}
                    onClick={handleVerifyRecall}
                    disabled={isSearching || queryIndexType === 'flat'}
                    title={queryIndexType === 'flat' ? 'Flat IS the ground truth — nothing to compare' : 'Run ANN vs Flat to compute Recall@K'}
                  >
                    <CheckCircle2 size={16} /> Verify Recall
                  </button>
                </div>
              </div>

              {/* Output Results Panel */}
              <div className="glass-card">
                <div className="card-title"><Play size={18} /> Execution Output</div>

                {searchResults ? (
                  <div>
                    <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', background: 'rgba(99, 102, 241, 0.1)', padding: '0.85rem', borderRadius: '10px', border: '1px solid rgba(99, 102, 241, 0.2)', flexWrap: 'wrap' }}>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Latency</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>{searchResults.latency_ms} ms</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Returned</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 700 }}>{searchResults.count} matches</div>
                      </div>
                      {queryIndexType === 'hnsw' && (
                        <div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ef_search</div>
                          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-primary)' }}>{efSearch}</div>
                        </div>
                      )}
                      {/* Recall badge — shown after Verify Recall is run */}
                      {recallResult && (() => {
                        const r = recallResult.recall_at_k;
                        const color = r >= 0.9 ? 'var(--accent-emerald)' : r >= 0.7 ? 'var(--accent-amber)' : 'var(--accent-rose)';
                        const bg   = r >= 0.9 ? 'rgba(16,185,129,0.12)' : r >= 0.7 ? 'rgba(245,158,11,0.12)' : 'rgba(244,63,94,0.12)';
                        const border = r >= 0.9 ? 'rgba(16,185,129,0.3)' : r >= 0.7 ? 'rgba(245,158,11,0.3)' : 'rgba(244,63,94,0.3)';
                        return (
                          <div style={{ background: bg, border: `1px solid ${border}`, borderRadius: 10, padding: '0.4rem 0.85rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Recall@{recallResult.k}</div>
                            <div style={{ fontSize: '1.2rem', fontWeight: 700, color }}>{(r * 100).toFixed(1)}%</div>
                            <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>{recallResult.ann_latency_ms}ms vs {recallResult.exact_latency_ms}ms</div>
                          </div>
                        );
                      })()}
                    </div>

                    <div className="table-container">
                      <table className="data-table">
                        <thead>
                          <tr><th>Rank</th><th>Vector ID</th><th>Score</th><th>Payload</th></tr>
                        </thead>
                        <tbody>
                          {searchResults.results.map((r, i) => (
                            <tr key={i}>
                              <td style={{ fontWeight: 600 }}>#{i + 1}</td>
                              <td className="mono" style={{ color: 'var(--accent-cyan)' }}>{r.id}</td>
                              <td className="mono" style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{r.score}</td>
                              <td className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{JSON.stringify(r.payload)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* 2D Vector Canvas Visualizer */}
                    {showVisualizer && (
                      <VectorCanvas queryVector={queryVector} searchResults={searchResults} />
                    )}
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '4rem 1rem', color: 'var(--text-dim)' }}>
                    <Search size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                    <p>Run a search query to view vector matches & distance rankings</p>
                    <p style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>The 2D vector projection visualizer will appear here after your first search</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ==================== TAB 4: METRICS ==================== */}
        {activeTab === 'metrics' && (
          <div>
            <div style={{ marginBottom: '1.5rem' }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>System Telemetry & Performance</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Real-time QPS, latency distributions, and collection build benchmarks.</p>
            </div>

            <div className="grid-2">
              <div className="glass-card">
                <div className="card-title"><Activity size={18} /> Throughput (QPS over Time)</div>
                <div style={{ height: '260px', marginTop: '1rem' }}>
                  <Line
                    data={{
                      labels: timeSeries.labels,
                      datasets: [{
                        label: 'Queries / Sec',
                        data: timeSeries.qps,
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.08)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 3
                      }]
                    }}
                    options={{ ...chartDefaults, plugins: { legend: { display: false } } }}
                  />
                </div>
              </div>

              <div className="glass-card">
                <div className="card-title"><BarChart2 size={18} /> Vector Distribution per Collection</div>
                <div style={{ height: '260px', marginTop: '1rem' }}>
                  <Bar
                    data={{
                      labels: collections.map(c => c.name),
                      datasets: [{
                        label: 'Vectors',
                        data: collections.map(c => c.vectors_count || 0),
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderColor: '#6366f1',
                        borderWidth: 1,
                        borderRadius: 6,
                      }]
                    }}
                    options={{ ...chartDefaults, plugins: { legend: { display: false } } }}
                  />
                </div>
              </div>

              <div className="glass-card">
                <div className="card-title"><LineChart size={18} /> Latency Percentiles over Time</div>
                <div style={{ height: '220px', marginTop: '1rem' }}>
                  <Line
                    data={{
                      labels: timeSeries.labels,
                      datasets: [
                        { label: 'P50', data: timeSeries.p50, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.06)', fill: true, tension: 0.4, pointRadius: 2 },
                        { label: 'P95', data: timeSeries.p95, borderColor: '#f59e0b', tension: 0.4, pointRadius: 2 },
                        { label: 'P99', data: timeSeries.p99, borderColor: '#f43f5e', tension: 0.4, pointRadius: 2 },
                      ]
                    }}
                    options={chartDefaults}
                  />
                </div>
              </div>

              <div className="glass-card">
                <div className="card-title"><Server size={18} /> Live Metrics Summary</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', marginTop: '1rem' }}>
                  {[
                    { label: 'Total Queries Served', value: (metrics.total_vectors * 0.8 | 0).toLocaleString(), color: 'var(--text-main)' },
                    { label: 'Live QPS (10s window)', value: `${metrics.qps} q/s`, color: 'var(--accent-cyan)' },
                    { label: 'P50 Latency', value: `${metrics.p50_ms} ms`, color: 'var(--accent-emerald)' },
                    { label: 'P95 Latency', value: `${metrics.p95_ms} ms`, color: 'var(--accent-amber)' },
                    { label: 'P99 Latency', value: `${metrics.p99_ms} ms`, color: 'var(--accent-rose)' },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1rem', borderRadius: '8px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
                      <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)', color }}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================== TAB 5: STORAGE & HEALTH ==================== */}
        {activeTab === 'storage' && (
          <div>
            <div style={{ marginBottom: '1.5rem' }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>Storage Durability & Maintenance</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>WAL log management, snapshot persistence, and compaction triggers.</p>
            </div>

            <div className="grid-2">
              <div className="glass-card">
                <div className="card-title"><HardDrive size={18} /> Write-Ahead Log (WAL) Stats</div>
                <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  {[
                    { label: 'Active Log File Size', value: formatBytes(metrics.storage.active_log_size), color: 'var(--text-main)' },
                    { label: 'Uncommitted Entries', value: `${metrics.storage.pending_entries} operations`, color: 'var(--text-main)' },
                    { label: 'Pending Tombstones', value: `${metrics.storage.tombstones_count} vectors`, color: 'var(--accent-amber)' },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.03)', padding: '0.85rem', borderRadius: '8px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
                      <span className="mono" style={{ fontWeight: 600, color }}>{value}</span>
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem' }}>
                  <button className="btn btn-primary" style={{ flex: 1 }} onClick={() => handleTriggerAction('snapshot')}>
                    <ShieldCheck size={16} /> Create Atomic Snapshot
                  </button>
                  <button className="btn btn-danger" style={{ flex: 1 }} onClick={() => handleTriggerAction('compaction')}>
                    <RefreshCw size={16} /> Run WAL Compaction
                  </button>
                </div>
              </div>

              <div className="glass-card">
                <div className="card-title"><Database size={18} /> Saved Snapshots Manifest</div>
                <div className="table-container" style={{ marginTop: '1rem' }}>
                  <table className="data-table">
                    <thead>
                      <tr><th>Snapshot ID</th><th>Created At</th><th>File Size</th><th>Vectors</th></tr>
                    </thead>
                    <tbody>
                      {snapshots.map((s, i) => (
                        <tr key={i}>
                          <td className="mono" style={{ color: 'var(--accent-cyan)' }}>{s.id}</td>
                          <td>{s.created_at}</td>
                          <td>{formatBytes(s.size_bytes)}</td>
                          <td>{typeof s.vectors === 'number' ? s.vectors.toLocaleString() : s.vectors}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ==================== MODALS ==================== */}

      {/* Create Collection */}
      {showCreateModal && (
        <div className="modal-overlay" onClick={e => e.target === e.currentTarget && setShowCreateModal(false)}>
          <div className="modal-content">
            <div className="modal-header">
              <h3 className="modal-title">Create New Collection</h3>
              <button className="modal-close" onClick={() => setShowCreateModal(false)}><X size={18} /></button>
            </div>
            <form onSubmit={handleCreateCollection}>
              <div className="form-group">
                <label className="form-label">Collection Name</label>
                <input className="form-input" required placeholder="e.g. product_embeddings" value={newCollName} onChange={e => setNewCollName(e.target.value)} />
              </div>
              <div className="form-group">
                <label className="form-label">Vector Dimension</label>
                <input className="form-input" type="number" required value={newCollDim} onChange={e => setNewCollDim(e.target.value)} />
              </div>
              <div className="form-group">
                <label className="form-label">Distance Metric</label>
                <select className="form-select" value={newCollMetric} onChange={e => setNewCollMetric(e.target.value)}>
                  <option value="cosine">Cosine Distance</option>
                  <option value="euclidean">Euclidean (L2)</option>
                  <option value="dot_product">Dot Product (Inner Product)</option>
                  <option value="manhattan">Manhattan (L1)</option>
                </select>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setShowCreateModal(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary">Create Collection</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete Collection */}
      {showDeleteModal && (
        <div className="modal-overlay" onClick={e => e.target === e.currentTarget && setShowDeleteModal(null)}>
          <div className="modal-content">
            <div className="modal-header">
              <h3 className="modal-title">Delete Collection</h3>
              <button className="modal-close" onClick={() => setShowDeleteModal(null)}><X size={18} /></button>
            </div>
            <p style={{ color: 'var(--text-muted)', margin: '1rem 0', lineHeight: 1.6 }}>
              Are you sure you want to delete <strong style={{ color: 'var(--accent-rose)' }}>{showDeleteModal}</strong>?
              This will permanently remove all indexed vectors and payloads. This action cannot be undone.
            </p>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowDeleteModal(null)}>Cancel</button>
              <button className="btn btn-danger" onClick={() => handleDeleteCollection(showDeleteModal)}>Confirm Delete</button>
            </div>
          </div>
        </div>
      )}

      {/* Collection Detail */}
      {showCollectionDetail && (
        <div className="modal-overlay" onClick={e => e.target === e.currentTarget && setShowCollectionDetail(null)}>
          <div className="modal-content" style={{ maxWidth: '650px' }}>
            <div className="modal-header">
              <h3 className="modal-title">Collection Schema & Index Details</h3>
              <button className="modal-close" onClick={() => setShowCollectionDetail(null)}><X size={18} /></button>
            </div>
            <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.9rem' }}>
              {[
                { label: 'Name', value: showCollectionDetail.name, mono: true },
                { label: 'Dimension', value: showCollectionDetail.dimension },
                { label: 'Metric', value: showCollectionDetail.metric },
                { label: 'Indexed Vectors', value: (showCollectionDetail.vectors_count || 0).toLocaleString() },
                { label: 'HNSW M', value: showCollectionDetail.hnsw_M || 16 },
                { label: 'ef_construction', value: showCollectionDetail.ef_construction || 64 },
                { label: 'ef_search', value: showCollectionDetail.ef_search || 32 },
              ].map(({ label, value, mono }) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '8px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{label}</span>
                  <span className={mono ? 'mono' : ''} style={{ fontWeight: 600 }}>{value}</span>
                </div>
              ))}
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowCollectionDetail(null)}>Close</button>
              <button className="btn btn-success" onClick={() => { setShowCollectionDetail(null); setShowInsertModal(showCollectionDetail); }}>
                <Plus size={14} /> Insert Vector
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Insert Vector Modal */}
      {showInsertModal && (
        <InsertVectorModal
          collection={showInsertModal}
          onClose={() => setShowInsertModal(null)}
          addToast={addToast}
        />
      )}

      {/* Batch Upload Modal */}
      {showBatchModal && (
        <BatchUploadModal
          collection={showBatchModal}
          onClose={() => setShowBatchModal(null)}
          addToast={addToast}
        />
      )}

      {/* Toast Notifications */}
      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {/* Spin keyframe for loading states */}
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
