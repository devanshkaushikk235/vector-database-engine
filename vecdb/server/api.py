import os
import shutil
import time
from typing import Any
from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from vecdb.core.distance import DistanceMetric
from vecdb.core.query_engine import FilterStrategy, IndexType
from vecdb.db.collection import Collection
from vecdb.server.metrics import metrics_registry

app = FastAPI(
    title="VecDB REST API & Dashboard Server",
    description="Vector Database built from first principles in Python",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory collections registry
collections: dict[str, Collection] = {}


# ── Pydantic Request Models ──────────────────────────────────────────────────

class CreateCollectionRequest(BaseModel):
    name: str
    dimension: int
    metric: str = "cosine"
    m_subspaces: int | None = None  # PQ sub-spaces; None → auto default (min(8, dim//16))


class VectorInsertRequest(BaseModel):
    id: str
    vector: list[float]
    payload: dict[str, Any] | None = None


class BatchVectorInsertRequest(BaseModel):
    vectors: list[VectorInsertRequest]


class SearchRequest(BaseModel):
    """Search request.  index_type: 'flat' | 'ivf' | 'hnsw' | 'pq'."""
    query: list[float] | None = None
    vector_id: str | None = None
    k: int = 10
    filter_query: dict[str, Any] | None = Field(default=None, alias="filter")
    filter_strategy: str = "pre_filter"
    index_type: str = "hnsw"
    ef_search: int | None = None
    n_probe: int | None = None


class RecallVerifyRequest(BaseModel):
    """Request body for recall verification against exact Flat search."""
    query: list[float] | None = None
    vector_id: str | None = None
    k: int = 10
    filter_query: dict[str, Any] | None = Field(default=None, alias="filter")
    filter_strategy: str = "pre_filter"
    index_type: str = "hnsw"   # ANN index to evaluate: 'hnsw' | 'ivf' | 'pq'
    ef_search: int | None = None
    n_probe: int | None = None


@app.get("/health")
def health_check():
    total_vecs = sum(len(c.vector_store) for c in collections.values())
    return {
        "status": "healthy",
        "collections_count": len(collections),
        "total_vectors": total_vecs
    }


@app.get("/collections")
def list_collections():
    items = []
    for name, coll in collections.items():
        pq = coll.query_engine.pq_index
        items.append({
            "name": name,
            "dimension": coll.dimension,
            "metric": coll.metric.value if hasattr(coll.metric, "value") else str(coll.metric),
            "vectors_count": len(coll.vector_store),
            "hnsw_M": coll.query_engine.hnsw_index.M,
            "ef_construction": coll.query_engine.hnsw_index.ef_construction,
            "ef_search": coll.query_engine.hnsw_index.ef_search,
            "m_subspaces": coll.m_subspaces,
            "pq_trained": pq.is_trained,
        })
    return {"collections": items}


@app.post("/collections", status_code=201)
def create_collection(req: CreateCollectionRequest):
    if req.name in collections:
        raise HTTPException(status_code=400, detail=f"Collection '{req.name}' already exists.")
    try:
        metric = DistanceMetric(req.metric.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid distance metric: {req.metric}")

    try:
        kwargs: dict[str, Any] = {}
        if req.m_subspaces is not None:
            kwargs["m_subspaces"] = req.m_subspaces
        coll = Collection(name=req.name, dimension=req.dimension, metric=metric, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    collections[req.name] = coll
    return {
        "status": "success",
        "message": f"Collection '{req.name}' created.",
        "m_subspaces": coll.m_subspaces,
    }


@app.delete("/collections/{name}")
def delete_collection(name: str):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    del collections[name]
    return {"status": "success", "message": f"Collection '{name}' deleted."}


@app.post("/collections/{name}/vectors")
def insert_vector(name: str, req: VectorInsertRequest):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    coll = collections[name]
    try:
        coll.insert(req.id, req.vector, req.payload)
    except ValueError as e:
        metrics_registry.record_error()
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "id": req.id}


@app.post("/collections/{name}/vectors/batch")
def insert_vectors_batch(name: str, req: BatchVectorInsertRequest):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    coll = collections[name]
    inserted_ids = []
    for item in req.vectors:
        coll.insert(item.id, item.vector, item.payload)
        inserted_ids.append(item.id)
    return {"status": "success", "count": len(inserted_ids), "ids": inserted_ids}


@app.get("/collections/{name}/vectors/{vector_id}")
def get_vector(name: str, vector_id: str):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    coll = collections[name]
    data = coll.get(vector_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Vector ID '{vector_id}' not found.")
    return data


@app.delete("/collections/{name}/vectors/{vector_id}")
def delete_vector(name: str, vector_id: str):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    coll = collections[name]
    success = coll.delete(vector_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Vector ID '{vector_id}' not found.")
    return {"status": "success", "id": vector_id}


@app.post("/collections/{name}/search")
def search_collection(name: str, req: SearchRequest):
    if name not in collections:
        metrics_registry.record_error()
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    coll = collections[name]

    # Handle search by example vector_id
    query_vec = req.query
    if query_vec is None and req.vector_id:
        v_data = coll.get(req.vector_id)
        if not v_data:
            metrics_registry.record_error()
            raise HTTPException(status_code=404, detail=f"Example vector ID '{req.vector_id}' not found.")
        query_vec = v_data["vector"]

    if query_vec is None:
        metrics_registry.record_error()
        raise HTTPException(status_code=400, detail="Must supply either 'query' vector array or 'vector_id'.")
    
    start_time = time.time()
    try:
        strategy = FilterStrategy(req.filter_strategy.lower())
        idx_type = IndexType(req.index_type.lower())
    except ValueError as e:
        metrics_registry.record_error()
        raise HTTPException(status_code=400, detail=str(e))

    total_candidates_before_filter = len(coll.vector_store)

    results = coll.search(
        query=query_vec,
        k=req.k,
        filter_query=req.filter_query,
        filter_strategy=strategy,
        index_type=idx_type,
        ef_search=req.ef_search,
        n_probe=req.n_probe
    )
    latency = time.time() - start_time
    metrics_registry.record_query(name, req.k, req.filter_query, len(results), latency)

    return {
        "results": results,
        "count": len(results),
        "total_candidates": total_candidates_before_filter,
        "latency_ms": round(latency * 1000, 2)
    }


@app.post("/collections/{name}/recall_verify")
def recall_verify(name: str, req: RecallVerifyRequest):
    """Compute Recall@K by comparing ANN results against exact Flat search.
    
    Returns recall_at_k, both result sets, and both latencies so the caller
    can inspect exactly which ground-truth neighbours the ANN index missed.
    """
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    coll = collections[name]

    query_vec = req.query
    if query_vec is None and req.vector_id:
        v_data = coll.get(req.vector_id)
        if not v_data:
            raise HTTPException(status_code=404, detail=f"Vector ID '{req.vector_id}' not found.")
        query_vec = v_data["vector"]
    if query_vec is None:
        raise HTTPException(status_code=400, detail="Must supply either 'query' or 'vector_id'.")

    try:
        strategy  = FilterStrategy(req.filter_strategy.lower())
        idx_type  = IndexType(req.index_type.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = coll.verify_recall(
        query=query_vec,
        k=req.k,
        index_type=idx_type,
        filter_query=req.filter_query,
        filter_strategy=strategy,
        ef_search=req.ef_search,
        n_probe=req.n_probe,
    )
    return result


@app.get("/collections/{name}/stats")
def get_collection_stats(name: str):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    return collections[name].get_stats()


@app.post("/collections/{name}/snapshot")
def create_snapshot(name: str):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    collections[name].create_snapshot()
    return {"status": "success", "message": f"Snapshot created for '{name}'."}


@app.post("/collections/{name}/compact")
def compact_collection(name: str):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    collections[name].compact()
    return {"status": "success", "message": f"Compaction completed for '{name}'."}


@app.post("/collections/{name}/rebuild")
def rebuild_collection(name: str, background_tasks: BackgroundTasks):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    
    def do_rebuild():
        collections[name].rebuild_index()

    background_tasks.add_task(do_rebuild)
    return {"status": "success", "message": f"Index rebuild started in background for '{name}'."}


@app.get("/metrics")
def get_metrics():
    stats_list = [coll.get_stats() for coll in collections.values()]
    content = metrics_registry.generate_prometheus_output(stats_list)
    return Response(content=content, media_type="text/plain")


@app.get("/ui/stats")
def get_dashboard_stats():
    """Aggregated JSON endpoint powering Phase 6 single-page Dashboard UI."""
    coll_stats = [coll.get_stats() for coll in collections.values()]
    total_vectors = sum(c["total_vectors"] for c in coll_stats)
    tombstone_count = sum(c["tombstoned_vectors"] for c in coll_stats)

    # Health status determination
    health_status = "green"
    health_msg = "All collections healthy"

    if tombstone_count > total_vectors * 0.2 and total_vectors > 50:
        health_status = "yellow"
        health_msg = f"Compaction recommended ({tombstone_count} tombstones pending)"
    
    # Disk usage
    total_disk_used = sum(c["storage"]["total_bytes"] for c in coll_stats)
    storage_info = shutil.disk_usage(".")
    
    wal_bytes = sum(c["storage"]["wal_bytes"] for c in coll_stats)
    pending_wal_entries = sum(c["storage"]["wal_entries_pending"] for c in coll_stats)
    
    snapshots_list = []
    segments_list = []
    for c in coll_stats:
        if c["storage"]["snapshot_bytes"] > 0:
            snapshots_list.append({
                "collection": c["name"],
                "path": c["name"] + "/snapshot.pkl",
                "size_bytes": c["storage"]["snapshot_bytes"],
                "mtime": c["storage"]["snapshot_mtime"]
            })
        segments_list.extend(c["storage"]["segments"])

    latencies_5m = metrics_registry.get_rolling_latencies_ms(window_seconds=300)
    qps = metrics_registry.get_rolling_qps(window_seconds=10)

    return {
        "total_collections": len(collections),
        "total_vectors": total_vectors,
        "health": {"status": health_status, "message": health_msg},
        "qps": round(qps, 2),
        "latency_5m": latencies_5m,
        "recent_queries": metrics_registry.get_recent_queries_feed(limit=15),
        "storage": {
            "used_bytes": total_disk_used,
            "total_bytes": storage_info.total,
            "free_bytes": storage_info.free
        },
        "collections": coll_stats,
        "wal_summary": {
            "total_wal_bytes": wal_bytes,
            "pending_entries": pending_wal_entries,
            "last_flush": time.strftime("%H:%M:%S")
        },
        "snapshots": snapshots_list,
        "segments": segments_list,
        "metrics_history": metrics_registry.time_series_history
    }


@app.get("/ui", response_class=HTMLResponse)
def get_dashboard_ui():
    """Phase 6: Single-Page Dashboard UI with 5 Tabs, Chart.js, Query Playground, and Health Monitoring."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VecDB Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg: #0f172a;
                --card-bg: #1e293b;
                --border: #334155;
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --primary: #38bdf8;
                --primary-hover: #0284c7;
                --green: #22c55e;
                --yellow: #eab308;
                --red: #ef4444;
            }
            body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
            .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 15px; margin-bottom: 20px; }
            .title-area h1 { margin: 0; color: var(--primary); font-size: 24px; font-weight: 700; }
            .title-area p { margin: 4px 0 0 0; color: var(--text-muted); font-size: 13px; }
            
            /* Tabs */
            .tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
            .tab-btn { background: transparent; color: var(--text-muted); border: none; padding: 10px 18px; font-size: 14px; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; }
            .tab-btn:hover { color: var(--text); }
            .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }
            .tab-content { display: none; }
            .tab-content.active { display: block; }

            /* Cards & Grid */
            .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 20px; }
            .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .card { background: var(--card-bg); border-radius: 8px; border: 1px solid var(--border); padding: 20px; }
            .card-title { color: var(--text-muted); font-size: 13px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
            .card-value { font-size: 28px; font-weight: 700; color: var(--text); }

            /* Health Badges */
            .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
            .badge-green { background: rgba(34, 197, 94, 0.2); color: var(--green); border: 1px solid var(--green); }
            .badge-yellow { background: rgba(234, 179, 8, 0.2); color: var(--yellow); border: 1px solid var(--yellow); }
            .badge-red { background: rgba(239, 68, 68, 0.2); color: var(--red); border: 1px solid var(--red); }

            /* Storage Progress Bar */
            .progress-bar-bg { background: var(--border); border-radius: 6px; height: 10px; overflow: hidden; margin-top: 10px; }
            .progress-bar-fill { background: var(--primary); height: 100%; width: 0%; transition: width 0.4s ease; }

            /* Tables */
            table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }
            th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); }
            th { color: var(--text-muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }
            tbody tr:hover { background: rgba(255, 255, 255, 0.03); cursor: pointer; }

            /* Forms & Buttons */
            input, select, textarea { background: #0f172a; border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 4px; font-size: 14px; width: 100%; box-sizing: border-box; }
            input:focus, select:focus, textarea:focus { outline: none; border-color: var(--primary); }
            .btn { background: var(--primary-hover); color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: 600; cursor: pointer; font-size: 14px; transition: background 0.2s; }
            .btn:hover { background: #0284c7; }
            .btn-danger { background: #dc2626; }
            .btn-danger:hover { background: #b91c1c; }
            .btn-secondary { background: var(--border); color: var(--text); }
            .btn-secondary:hover { background: #475569; }

            /* Playground Dynamic Filter Rows */
            .filter-row { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; }
            .filter-row input, .filter-row select { width: auto; flex: 1; }

            /* JSON view */
            pre { background: #090d16; padding: 12px; border-radius: 6px; border: 1px solid var(--border); overflow-x: auto; font-size: 13px; color: #a5f3fc; }
            
            /* Modal */
            .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); justify-content: center; align-items: center; z-index: 1000; }
            .modal-content { background: var(--card-bg); border-radius: 8px; border: 1px solid var(--border); padding: 24px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title-area">
                <h1>VecDB Monitoring & Operations Dashboard</h1>
                <p>First-Principles Vector Database Engine — Phase 6 Dashboard</p>
            </div>
            <div>
                <span id="health-badge" class="badge badge-green">Health: Green</span>
            </div>
        </div>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab(event, 'overview')">Overview</button>
            <button class="tab-btn" onclick="switchTab(event, 'collections')">Collections</button>
            <button class="tab-btn" onclick="switchTab(event, 'playground')">Query Playground</button>
            <button class="tab-btn" onclick="switchTab(event, 'metrics')">Metrics</button>
            <button class="tab-btn" onclick="switchTab(event, 'storage')">Storage & Health</button>
        </div>

        <!-- TAB 1: OVERVIEW -->
        <div id="tab-overview" class="tab-content active">
            <div class="grid-4">
                <div class="card">
                    <div class="card-title">Total Collections</div>
                    <div class="card-value" id="ov-collections">0</div>
                </div>
                <div class="card">
                    <div class="card-title">Total Vectors Indexed</div>
                    <div class="card-value" id="ov-vectors">0</div>
                </div>
                <div class="card">
                    <div class="card-title">Live QPS (10s window)</div>
                    <div class="card-value" id="ov-qps">0.0</div>
                </div>
                <div class="card">
                    <div class="card-title">p95 Latency (5m)</div>
                    <div class="card-value" id="ov-p95">0.00 ms</div>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <div class="card-title">Rolling Latency Panel (Last 5 Minutes)</div>
                    <p><strong>p50:</strong> <span id="lat-p50">0.00 ms</span> | <strong>p95:</strong> <span id="lat-p95">0.00 ms</span> | <strong>p99:</strong> <span id="lat-p99">0.00 ms</span></p>
                    <div class="card-title" style="margin-top: 15px;">Storage Usage</div>
                    <p id="storage-text">0 MB used of 0 GB</p>
                    <div class="progress-bar-bg">
                        <div id="storage-bar" class="progress-bar-fill"></div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Recent Query Feed</div>
                    <table>
                        <thead>
                            <tr><th>Time</th><th>Collection</th><th>k</th><th>Results</th><th>Latency</th></tr>
                        </thead>
                        <tbody id="feed-table-body">
                            <tr><td colspan="5" style="color:var(--text-muted)">No query history recorded yet.</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 2: COLLECTIONS -->
        <div id="tab-collections" class="tab-content">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h2 style="margin:0; font-size:18px;">Managed Collections</h2>
                <button class="btn" onclick="openCreateModal()">+ Create Collection</button>
            </div>
            <div class="card">
                <table>
                    <thead>
                        <tr><th>Name</th><th>Vectors</th><th>Dimension</th><th>Metric</th><th>Tombstones</th><th>Created At</th><th>Actions</th></tr>
                    </thead>
                    <tbody id="collections-table-body"></tbody>
                </table>
            </div>

            <div id="coll-detail-card" class="card" style="margin-top: 20px; display: none;">
                <h3 id="detail-title" style="color:var(--primary); margin-top:0;">Collection Detail</h3>
                <div class="grid-2">
                    <div>
                        <h4>HNSW Configuration</h4>
                        <p><strong>M:</strong> <span id="dt-m">-</span> | <strong>ef_construction:</strong> <span id="dt-efc">-</span> | <strong>ef_search:</strong> <span id="dt-efs">-</span></p>
                        <h4>Disk & Segment Stats</h4>
                        <p><strong>Segment Count:</strong> <span id="dt-segcount">1</span> | <strong>Total Size:</strong> <span id="dt-size">0 KB</span></p>
                    </div>
                    <div>
                        <h4>Inferred Payload Schema</h4>
                        <pre id="dt-schema">{}</pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 3: QUERY PLAYGROUND -->
        <div id="tab-playground" class="tab-content">
            <div class="grid-2">
                <div class="card">
                    <h3 style="margin-top:0;">Execute Vector Search Query</h3>
                    <div style="margin-bottom: 12px;">
                        <label>Collection Target</label>
                        <select id="pg-coll-select"></select>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <label>Query Input Method</label>
                        <select id="pg-input-type" onchange="toggleQueryInput()">
                            <option value="raw">Raw Vector JSON Array</option>
                            <option value="id">Vector ID (Search by Example)</option>
                        </select>
                    </div>
                    <div id="input-raw-container" style="margin-bottom: 12px;">
                        <label>Raw Float Array (e.g. [0.1, 0.2, ...])</label>
                        <textarea id="pg-vector-json" rows="3" placeholder="[0.1, 0.2, ...]"></textarea>
                    </div>
                    <div id="input-id-container" style="margin-bottom: 12px; display:none;">
                        <label>Target Vector ID</label>
                        <input type="text" id="pg-vector-id" placeholder="e.g. item_42">
                    </div>
                    <div style="display: flex; gap: 10px; margin-bottom: 12px;">
                        <div style="flex:1;">
                            <label>Top-k</label>
                            <input type="number" id="pg-k" value="10">
                        </div>
                        <div style="flex:1;">
                            <label>Strategy</label>
                            <select id="pg-strategy">
                                <option value="pre_filter">Pre-Filter</option>
                                <option value="post_filter">Post-Filter</option>
                            </select>
                        </div>
                        <div style="flex:1;">
                            <label>Index Algorithm</label>
                            <select id="pg-index">
                                <option value="hnsw">HNSW</option>
                                <option value="ivf">IVF</option>
                                <option value="flat">Flat (Exact)</option>
                            </select>
                        </div>
                    </div>

                    <!-- Metadata Filter Builder -->
                    <div style="margin-bottom: 12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <label>Metadata Filter Builder</label>
                            <button class="btn btn-secondary" onclick="addFilterRow()" style="padding:4px 8px; font-size:12px;">+ Add Field Condition</button>
                        </div>
                        <div id="filter-builder-rows"></div>
                    </div>

                    <button class="btn" onclick="executeSearch()" style="width:100%; margin-top:10px;">Run Query</button>
                </div>

                <div class="card">
                    <h3 style="margin-top:0;">Query Results & Performance</h3>
                    <p><strong>Candidates before filter:</strong> <span id="res-candidates">-</span> | <strong>Results after filter:</strong> <span id="res-count">-</span> | <strong>Query time:</strong> <span id="res-latency" style="color:var(--green)">- ms</span></p>
                    <table>
                        <thead>
                            <tr><th>Rank</th><th>Vector ID</th><th>Distance Score</th><th>Payload JSON</th></tr>
                        </thead>
                        <tbody id="pg-results-body">
                            <tr><td colspan="4" style="color:var(--text-muted)">Run a search query to view results.</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 4: METRICS -->
        <div id="tab-metrics" class="tab-content">
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">QPS Over Time (Rolling 10s Window)</div>
                    <canvas id="chart-qps"></canvas>
                </div>
                <div class="card">
                    <div class="card-title">Rolling Latencies (p50 / p95 / p99)</div>
                    <canvas id="chart-latency"></canvas>
                </div>
            </div>
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">Vectors Per Collection</div>
                    <canvas id="chart-vectors"></canvas>
                </div>
                <div class="card">
                    <div class="card-title">Last Index Build Time (Seconds)</div>
                    <canvas id="chart-builds"></canvas>
                </div>
            </div>
        </div>

        <!-- TAB 5: STORAGE & HEALTH -->
        <div id="tab-storage" class="tab-content">
            <div class="grid-2">
                <div class="card">
                    <h3 style="margin-top:0;">Write-Ahead Log (WAL) Status</h3>
                    <p><strong>Total WAL Disk Size:</strong> <span id="wal-bytes">0 KB</span></p>
                    <p><strong>Pending Unflushed Entries:</strong> <span id="wal-pending">0</span></p>
                    <p><strong>Last Flush Timestamp:</strong> <span id="wal-flush">-</span></p>
                </div>
                <div class="card">
                    <h3 style="margin-top:0;">Global Administrative Actions</h3>
                    <p style="color:var(--text-muted); font-size:13px;">Write actions require explicit user confirmation.</p>
                    <div style="display:flex; gap:10px; margin-top:15px;">
                        <button class="btn" onclick="confirmGlobalSnapshot()">Trigger Manual Snapshot</button>
                        <button class="btn btn-secondary" onclick="confirmGlobalCompaction()">Trigger Full Compaction</button>
                    </div>
                </div>
            </div>

            <div class="card" style="margin-top:20px;">
                <h3 style="margin-top:0;">Index Storage Segments</h3>
                <table>
                    <thead>
                        <tr><th>Segment ID</th><th>Vector Count</th><th>Size Bytes</th><th>Status</th></tr>
                    </thead>
                    <tbody id="segments-table-body"></tbody>
                </table>
            </div>

            <div class="card" style="margin-top:20px;">
                <h3 style="margin-top:0;">Persisted Snapshots</h3>
                <table>
                    <thead>
                        <tr><th>Collection</th><th>Snapshot Path</th><th>Size Bytes</th><th>Last Modified</th></tr>
                    </thead>
                    <tbody id="snapshots-table-body"></tbody>
                </table>
            </div>
        </div>

        <!-- CREATE COLLECTION MODAL -->
        <div id="create-modal" class="modal-overlay">
            <div class="modal-content">
                <h3 style="margin-top:0; color:var(--primary);">Create New Collection</h3>
                <div style="margin-bottom:12px;">
                    <label>Collection Name</label>
                    <input type="text" id="new-coll-name" placeholder="e.g. products">
                </div>
                <div style="margin-bottom:12px;">
                    <label>Dimension Size</label>
                    <input type="number" id="new-coll-dim" value="128">
                </div>
                <div style="margin-bottom:16px;">
                    <label>Distance Metric</label>
                    <select id="new-coll-metric">
                        <option value="cosine">Cosine</option>
                        <option value="euclidean">Euclidean</option>
                        <option value="dot_product">Dot Product</option>
                        <option value="manhattan">Manhattan</option>
                    </select>
                </div>
                <div style="display:flex; justify-content:flex-end; gap:10px;">
                    <button class="btn btn-secondary" onclick="closeCreateModal()">Cancel</button>
                    <button class="btn" onclick="submitCreateCollection()">Create</button>
                </div>
            </div>
        </div>

        <script>
            let dashboardData = {};
            let chartQPS, chartLatency, chartVectors, chartBuilds;

            function switchTab(evt, tabId) {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                if (evt && evt.target) evt.target.classList.add('active');
                const targetTab = document.getElementById('tab-' + tabId);
                if (targetTab) targetTab.classList.add('active');
                if (tabId === 'metrics') renderMetricsCharts();
            }

            function toggleQueryInput() {
                const type = document.getElementById('pg-input-type').value;
                document.getElementById('input-raw-container').style.display = type === 'raw' ? 'block' : 'none';
                document.getElementById('input-id-container').style.display = type === 'id' ? 'block' : 'none';
            }

            function addFilterRow() {
                const container = document.getElementById('filter-builder-rows');
                const row = document.createElement('div');
                row.className = 'filter-row';
                row.innerHTML = `
                    <input type="text" placeholder="Key (e.g. price)" class="f-key">
                    <select class="f-op">
                        <option value="eq">eq (=)</option>
                        <option value="neq">neq (!=)</option>
                        <option value="gt">gt (&gt;)</option>
                        <option value="gte">gte (&ge;)</option>
                        <option value="lt">lt (&lt;)</option>
                        <option value="lte">lte (&le;)</option>
                        <option value="in">in</option>
                        <option value="nin">nin</option>
                    </select>
                    <input type="text" placeholder="Value (e.g. 100)" class="f-val">
                    <button class="btn btn-danger" style="padding:4px 8px; font-size:12px;" onclick="this.parentElement.remove()">X</button>
                `;
                container.appendChild(row);
            }

            async function fetchStats() {
                try {
                    const res = await fetch('/ui/stats');
                    dashboardData = await res.json();
                    renderOverview();
                    renderCollectionsTable();
                    renderStorageTab();
                    updatePlaygroundDropdown();
                } catch(e) {
                    console.error("Error fetching stats:", e);
                }
            }

            function renderOverview() {
                document.getElementById('ov-collections').innerText = dashboardData.total_collections || 0;
                document.getElementById('ov-vectors').innerText = (dashboardData.total_vectors || 0).toLocaleString();
                document.getElementById('ov-qps').innerText = dashboardData.qps || 0.0;
                
                const lats = dashboardData.latency_5m || {p50:0, p95:0, p99:0};
                document.getElementById('ov-p95').innerText = lats.p95 + ' ms';
                document.getElementById('lat-p50').innerText = lats.p50 + ' ms';
                document.getElementById('lat-p95').innerText = lats.p95 + ' ms';
                document.getElementById('lat-p99').innerText = lats.p99 + ' ms';

                // Health Badge
                const hBadge = document.getElementById('health-badge');
                const h = dashboardData.health || {status:'green', message:'Healthy'};
                hBadge.className = 'badge badge-' + h.status;
                hBadge.innerText = 'Health: ' + h.status.toUpperCase() + ' (' + h.message + ')';

                // Storage Bar
                const st = dashboardData.storage || {used_bytes:0, total_bytes:1};
                const usedMB = (st.used_bytes / (1024*1024)).toFixed(2);
                const totalGB = (st.total_bytes / (1024*1024*1024)).toFixed(1);
                const pct = Math.min(100, ((st.used_bytes / st.total_bytes) * 100).toFixed(1));
                document.getElementById('storage-text').innerText = `${usedMB} MB used of ${totalGB} GB disk space (${pct}%)`;
                document.getElementById('storage-bar').style.width = pct + '%';

                // Recent Feed
                const tbody = document.getElementById('feed-table-body');
                if (dashboardData.recent_queries && dashboardData.recent_queries.length > 0) {
                    tbody.innerHTML = dashboardData.recent_queries.map(q => `
                        <tr>
                            <td>${q.timestamp}</td>
                            <td>${q.collection}</td>
                            <td>${q.k}</td>
                            <td>${q.result_count}</td>
                            <td style="color:var(--green)">${q.latency_ms} ms</td>
                        </tr>
                    `).join('');
                }
            }

            function renderCollectionsTable() {
                const tbody = document.getElementById('collections-table-body');
                if (!dashboardData.collections || dashboardData.collections.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="7" style="color:var(--text-muted)">No active collections found.</td></tr>`;
                    return;
                }

                tbody.innerHTML = dashboardData.collections.map(c => `
                    <tr onclick="viewCollectionDetail('${c.name}')">
                        <td><strong>${c.name}</strong></td>
                        <td>${c.total_vectors.toLocaleString()}</td>
                        <td>${c.dimension}</td>
                        <td>${c.metric}</td>
                        <td>${c.tombstoned_vectors}</td>
                        <td>${c.created_at}</td>
                        <td onclick="event.stopPropagation()">
                            <button class="btn btn-secondary" style="padding:4px 8px; font-size:12px;" onclick="triggerRebuild('${c.name}')">Rebuild</button>
                            <button class="btn btn-danger" style="padding:4px 8px; font-size:12px;" onclick="confirmDeleteCollection('${c.name}')">Delete</button>
                        </td>
                    </tr>
                `).join('');
            }

            function viewCollectionDetail(cName) {
                const c = dashboardData.collections.find(x => x.name === cName);
                if (!c) return;
                document.getElementById('coll-detail-card').style.display = 'block';
                document.getElementById('detail-title').innerText = `Collection Detail: ${c.name}`;
                document.getElementById('dt-m').innerText = c.hnsw_config.M;
                document.getElementById('dt-efc').innerText = c.hnsw_config.ef_construction;
                document.getElementById('dt-efs').innerText = c.hnsw_config.ef_search;
                document.getElementById('dt-segcount').innerText = c.storage.segment_count;
                document.getElementById('dt-size').innerText = (c.storage.total_bytes / 1024).toFixed(1) + ' KB';
                document.getElementById('dt-schema').innerText = JSON.stringify(c.payload_schema, null, 2);
            }

            function renderStorageTab() {
                const wal = dashboardData.wal_summary || {};
                document.getElementById('wal-bytes').innerText = ((wal.total_wal_bytes || 0) / 1024).toFixed(1) + ' KB';
                document.getElementById('wal-pending').innerText = wal.pending_entries || 0;
                document.getElementById('wal-flush').innerText = wal.last_flush || '-';

                // Segments
                const segBody = document.getElementById('segments-table-body');
                if (dashboardData.segments && dashboardData.segments.length > 0) {
                    segBody.innerHTML = dashboardData.segments.map(s => `
                        <tr>
                            <td>${s.id}</td>
                            <td>${s.vector_count.toLocaleString()}</td>
                            <td>${(s.size_bytes / 1024).toFixed(1)} KB</td>
                            <td><span class="badge badge-green">${s.status}</span></td>
                        </tr>
                    `).join('');
                }

                // Snapshots
                const snapBody = document.getElementById('snapshots-table-body');
                if (dashboardData.snapshots && dashboardData.snapshots.length > 0) {
                    snapBody.innerHTML = dashboardData.snapshots.map(s => `
                        <tr>
                            <td>${s.collection}</td>
                            <td>${s.path}</td>
                            <td>${(s.size_bytes / 1024).toFixed(1)} KB</td>
                            <td>${s.mtime}</td>
                        </tr>
                    `).join('');
                } else {
                    snapBody.innerHTML = `<tr><td colspan="4" style="color:var(--text-muted)">No snapshots stored on disk.</td></tr>`;
                }
            }

            function updatePlaygroundDropdown() {
                const select = document.getElementById('pg-coll-select');
                const curr = select.value;
                if (dashboardData.collections) {
                    select.innerHTML = dashboardData.collections.map(c => `<option value="${c.name}">${c.name} (${c.dimension}-d)</option>`).join('');
                    if (curr) select.value = curr;
                }
            }

            async function executeSearch() {
                const collName = document.getElementById('pg-coll-select').value;
                if (!collName) { alert("Please select a collection!"); return; }

                const inputType = document.getElementById('pg-input-type').value;
                const k = parseInt(document.getElementById('pg-k').value);
                const strategy = document.getElementById('pg-strategy').value;
                const indexType = document.getElementById('pg-index').value;

                // Build filter object from rows
                const filterObj = {};
                document.querySelectorAll('.filter-row').forEach(row => {
                    const key = row.querySelector('.f-key').value.trim();
                    const op = row.querySelector('.f-op').value;
                    let val = row.querySelector('.f-val').value.trim();

                    if (key) {
                        // Parse float/int/json if possible
                        if (!isNaN(val) && val !== '') val = parseFloat(val);
                        else if (val.startsWith('[') || val.startsWith('{')) {
                            try { val = JSON.parse(val); } catch(e){}
                        }
                        filterObj[key] = op === 'eq' ? val : { [op]: val };
                    }
                });

                const body = {
                    k: k,
                    filter: Object.keys(filterObj).length > 0 ? filterObj : null,
                    filter_strategy: strategy,
                    index_type: indexType
                };

                if (inputType === 'raw') {
                    const rawStr = document.getElementById('pg-vector-json').value.trim();
                    if (!rawStr) { alert("Please paste a vector JSON array!"); return; }
                    try { body.query = JSON.parse(rawStr); } catch(e) { alert("Invalid JSON array!"); return; }
                } else {
                    const vid = document.getElementById('pg-vector-id').value.trim();
                    if (!vid) { alert("Please enter a target Vector ID!"); return; }
                    body.vector_id = vid;
                }

                try {
                    const res = await fetch(`/collections/${collName}/search`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(body)
                    });
                    const data = await res.json();

                    if (res.status !== 200) { alert("Search Error: " + data.detail); return; }

                    document.getElementById('res-candidates').innerText = data.total_candidates;
                    document.getElementById('res-count').innerText = data.count;
                    document.getElementById('res-latency').innerText = data.latency_ms + ' ms';

                    const tbody = document.getElementById('pg-results-body');
                    if (data.results.length === 0) {
                        tbody.innerHTML = `<tr><td colspan="4" style="color:var(--text-muted)">No matching vectors found.</td></tr>`;
                    } else {
                        tbody.innerHTML = data.results.map((r, idx) => `
                            <tr>
                                <td><strong>#${idx + 1}</strong></td>
                                <td>${r.id}</td>
                                <td>${r.score.toFixed(4)}</td>
                                <td><pre style="margin:0; padding:4px;">${JSON.stringify(r.payload)}</pre></td>
                            </tr>
                        `).join('');
                    }
                } catch(e) {
                    alert("Execution failed: " + e);
                }
            }

            // Write Actions with Confirmation
            async function confirmDeleteCollection(cName) {
                if (confirm(`CAUTION: Are you sure you want to permanently delete collection '${cName}'?`)) {
                    await fetch(`/collections/${cName}`, { method: 'DELETE' });
                    fetchStats();
                }
            }

            async function triggerRebuild(cName) {
                if (confirm(`Trigger non-blocking index rebuild for collection '${cName}'?`)) {
                    await fetch(`/collections/${cName}/rebuild`, { method: 'POST' });
                    alert(`Index rebuild started in background for '${cName}'.`);
                    fetchStats();
                }
            }

            async function confirmGlobalSnapshot() {
                if (confirm("Persist manual binary snapshot for all active collections?")) {
                    for (let c of dashboardData.collections || []) {
                        await fetch(`/collections/${c.name}/snapshot`, { method: 'POST' });
                    }
                    alert("Snapshot trigger completed!");
                    fetchStats();
                }
            }

            async function confirmGlobalCompaction() {
                if (confirm("Trigger tombstone compaction and index rebuild for all collections?")) {
                    for (let c of dashboardData.collections || []) {
                        await fetch(`/collections/${c.name}/compact`, { method: 'POST' });
                    }
                    alert("Compaction completed!");
                    fetchStats();
                }
            }

            // Modal Controls
            function openCreateModal() { document.getElementById('create-modal').style.display = 'flex'; }
            function closeCreateModal() { document.getElementById('create-modal').style.display = 'none'; }
            
            async function submitCreateCollection() {
                const name = document.getElementById('new-coll-name').value.trim();
                const dim = parseInt(document.getElementById('new-coll-dim').value);
                const metric = document.getElementById('new-coll-metric').value;

                if (!name || !dim) { alert("Name and dimension are required!"); return; }

                const res = await fetch('/collections', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ name: name, dimension: dim, metric: metric })
                });

                if (res.status === 201) {
                    closeCreateModal();
                    fetchStats();
                } else {
                    const err = await res.json();
                    alert("Error: " + err.detail);
                }
            }

            // Render Chart.js
            function renderMetricsCharts() {
                if (typeof Chart === 'undefined') return;
                const history = dashboardData.metrics_history || [];
                const labels = history.map(h => h.timestamp);

                if (!chartQPS) {
                    chartQPS = new Chart(document.getElementById('chart-qps'), {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [{ label: 'QPS', data: history.map(h => h.qps), borderColor: '#38bdf8', tension: 0.3 }]
                        },
                        options: { responsive: true, scales: { y: { beginAtZero: true } } }
                    });
                } else {
                    chartQPS.data.labels = labels;
                    chartQPS.data.datasets[0].data = history.map(h => h.qps);
                    chartQPS.update();
                }

                if (!chartLatency) {
                    chartLatency = new Chart(document.getElementById('chart-latency'), {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [
                                { label: 'p50', data: history.map(h => h.p50), borderColor: '#22c55e' },
                                { label: 'p95', data: history.map(h => h.p95), borderColor: '#eab308' },
                                { label: 'p99', data: history.map(h => h.p99), borderColor: '#ef4444' }
                            ]
                        },
                        options: { responsive: true, scales: { y: { beginAtZero: true } } }
                    });
                } else {
                    chartLatency.data.labels = labels;
                    chartLatency.data.datasets[0].data = history.map(h => h.p50);
                    chartLatency.data.datasets[1].data = history.map(h => h.p95);
                    chartLatency.data.datasets[2].data = history.map(h => h.p99);
                    chartLatency.update();
                }

                // Bar Charts
                const colls = dashboardData.collections || [];
                const collNames = colls.map(c => c.name);

                if (!chartVectors) {
                    chartVectors = new Chart(document.getElementById('chart-vectors'), {
                        type: 'bar',
                        data: {
                            labels: collNames,
                            datasets: [{ label: 'Vector Count', data: colls.map(c => c.total_vectors), backgroundColor: '#38bdf8' }]
                        },
                        options: { responsive: true }
                    });
                } else {
                    chartVectors.data.labels = collNames;
                    chartVectors.data.datasets[0].data = colls.map(c => c.total_vectors);
                    chartVectors.update();
                }

                if (!chartBuilds) {
                    chartBuilds = new Chart(document.getElementById('chart-builds'), {
                        type: 'bar',
                        data: {
                            labels: collNames,
                            datasets: [{ label: 'Build Time (s)', data: colls.map(c => c.last_build_duration_sec), backgroundColor: '#a855f7' }]
                        },
                        options: { responsive: true }
                    });
                } else {
                    chartBuilds.data.labels = collNames;
                    chartBuilds.data.datasets[0].data = colls.map(c => c.last_build_duration_sec);
                    chartBuilds.update();
                }
            }

            // Auto refresh stats every 5 seconds
            setInterval(fetchStats, 5000);
            fetchStats();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
