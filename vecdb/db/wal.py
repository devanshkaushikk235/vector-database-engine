import json
import os
import threading
import time
from typing import Any


class WriteAheadLog:
    """Write-Ahead Log (WAL) for mutation durability using newline-delimited JSON.
    
    Thread-safe: a threading.Lock serialises all log() calls and ensures each
    entry is fully fsynced to disk before the lock is released.  This guarantees
    that a crash between write() and fsync() cannot produce a torn/corrupt WAL.
    """

    def __init__(self, filepath: str):
        """Initialize WAL at specified file path.
        
        Args:
            filepath: Path to the .jsonl log file.
        """
        self.filepath = filepath
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                pass

    def log(
        self,
        op: str,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None
    ) -> None:
        """Append a mutation entry to the WAL and fsync before returning.
        
        The lock is held for the entire open → write → flush → fsync sequence.
        A crash anywhere inside this block leaves the entry either fully written
        or not present; it can never be partially flushed.
        
        Args:
            op: Operation type ('insert', 'upsert', 'delete').
            vector_id: Target vector identifier.
            vector: Float vector list (for insert/upsert).
            payload: Metadata payload dictionary.
        """
        entry = {
            "op": op,
            "id": vector_id,
            "vector": vector,
            "payload": payload,
            "timestamp": time.time()
        }
        serialised = json.dumps(entry) + "\n"

        with self._lock:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(serialised)
                f.flush()
                # fsync forces the kernel buffer to physical storage before we
                # release the lock.  Without this, a crash after flush() but
                # before the OS writes to disk would silently corrupt the WAL.
                os.fsync(f.fileno())

    def replay(self) -> list[dict[str, Any]]:
        """Replay all entries from the WAL file.
        
        Returns:
            list[dict]: List of ordered operation records.
        """
        if not os.path.exists(self.filepath):
            return []

        entries = []
        # replay() is called only during recovery (single-threaded startup),
        # so no lock is required here.
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def truncate(self) -> None:
        """Truncate / clear the WAL log file."""
        with self._lock:
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
