import time
from typing import Any
import numpy as np


class QueryRecord:
    """Dataclass representing a recorded query execution."""
    def __init__(self, collection: str, k: int, filter_applied: dict[str, Any] | None, result_count: int, latency_ms: float):
        self.timestamp = time.time()
        self.collection = collection
        self.k = k
        self.filter_applied = filter_applied
        self.result_count = result_count
        self.latency_ms = latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "collection": self.collection,
            "k": self.k,
            "filter_applied": self.filter_applied or "None",
            "result_count": self.result_count,
            "latency_ms": round(self.latency_ms, 2)
        }


class PrometheusMetrics:
    """Prometheus and Dashboard metrics tracking engine."""

    def __init__(self):
        self.total_queries: int = 0
        self.total_errors: int = 0
        self.query_history: list[QueryRecord] = []
        self.time_series_history: list[dict[str, Any]] = []
        self.last_ts_sample: float = time.time()

    def record_query(self, collection: str, k: int, filter_applied: dict[str, Any] | None, result_count: int, latency_seconds: float) -> None:
        """Record a search query execution."""
        self.total_queries += 1
        latency_ms = latency_seconds * 1000.0
        rec = QueryRecord(collection, k, filter_applied, result_count, latency_ms)
        self.query_history.append(rec)
        
        # Keep last 200 query records
        if len(self.query_history) > 200:
            self.query_history.pop(0)

        self._update_time_series()

    def record_error(self) -> None:
        """Record an API error occurrence."""
        self.total_errors += 1

    def _update_time_series(self) -> None:
        """Periodically sample metrics into time-series buckets (every 10 seconds)."""
        now = time.time()
        if now - self.last_ts_sample >= 10.0:
            self.last_ts_sample = now
            lats = self.get_rolling_latencies_ms(window_seconds=300)
            qps = self.get_rolling_qps(window_seconds=10)
            
            sample = {
                "timestamp": time.strftime("%H:%M:%S", time.localtime(now)),
                "qps": round(qps, 2),
                "p50": round(lats["p50"], 2),
                "p95": round(lats["p95"], 2),
                "p99": round(lats["p99"], 2),
                "error_rate": round(self.total_errors / max(1, self.total_queries), 4)
            }
            self.time_series_history.append(sample)
            if len(self.time_series_history) > 360:  # Keep 1 hour of samples
                self.time_series_history.pop(0)

    def get_rolling_qps(self, window_seconds: float = 10.0) -> float:
        """Calculate QPS over rolling window seconds."""
        now = time.time()
        count = sum(1 for q in self.query_history if now - q.timestamp <= window_seconds)
        return count / max(1.0, window_seconds)

    def get_rolling_latencies_ms(self, window_seconds: float = 300.0) -> dict[str, float]:
        """Calculate p50, p95, p99 latencies in ms for last window_seconds (default 5 min)."""
        now = time.time()
        recent_lats = [q.latency_ms for q in self.query_history if now - q.timestamp <= window_seconds]
        
        if not recent_lats:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        return {
            "p50": float(np.percentile(recent_lats, 50)),
            "p95": float(np.percentile(recent_lats, 95)),
            "p99": float(np.percentile(recent_lats, 99))
        }

    def get_recent_queries_feed(self, limit: int = 15) -> list[dict[str, Any]]:
        """Return list of recent queries for dashboard feed."""
        return [q.to_dict() for q in reversed(self.query_history[-limit:])]

    def generate_prometheus_output(self, collections_stats: list[dict[str, Any]]) -> str:
        """Generate Prometheus-formatted metric output."""
        lines = []

        lines.append("# HELP vecdb_query_count_total Total number of search queries executed")
        lines.append("# TYPE vecdb_query_count_total counter")
        lines.append(f"vecdb_query_count_total {self.total_queries}")

        lats = self.get_rolling_latencies_ms(window_seconds=300)
        lines.append("# HELP vecdb_query_latency_p95_seconds p95 query latency in seconds")
        lines.append("# TYPE vecdb_query_latency_p95_seconds gauge")
        lines.append(f"vecdb_query_latency_p95_seconds {lats['p95'] / 1000.0:.6f}")

        lines.append("# HELP vecdb_collection_vector_count Active vectors per collection")
        lines.append("# TYPE vecdb_collection_vector_count gauge")
        for stats in collections_stats:
            c_name = stats["name"]
            total_vecs = stats["total_vectors"]
            lines.append(f'vecdb_collection_vector_count{{collection="{c_name}"}} {total_vecs}')

        lines.append("# HELP vecdb_collection_tombstone_count Tombstoned vectors per collection")
        lines.append("# TYPE vecdb_collection_tombstone_count gauge")
        for stats in collections_stats:
            c_name = stats["name"]
            t_vecs = stats["tombstoned_vectors"]
            lines.append(f'vecdb_collection_tombstone_count{{collection="{c_name}"}} {t_vecs}')

        return "\n".join(lines) + "\n"


metrics_registry = PrometheusMetrics()
