"""
In-memory job store for background video processing.

YOLOv8 inference on a full video can take a long time, so the API processes
uploads on a background thread and lets the client poll for progress instead of
holding a single long HTTP request open. This store is intentionally simple
(a thread-safe dict); for multi-process / multi-worker deployments swap it for
Redis or a database.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "filename": filename,
                "status": "queued",        # queued | processing | done | error
                "progress": 0.0,           # 0.0 - 1.0
                "result": None,
                "error": None,
                "created_at": time.time(),
            }
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(fields)

    def set_progress(self, job_id: str, progress: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job["progress"] = max(0.0, min(1.0, float(progress)))

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def cleanup(self, max_age_sec: float = 3600.0) -> None:
        """Drop finished jobs older than max_age_sec to bound memory."""
        now = time.time()
        with self._lock:
            stale = [
                jid for jid, j in self._jobs.items()
                if j["status"] in ("done", "error")
                and now - j["created_at"] > max_age_sec
            ]
            for jid in stale:
                del self._jobs[jid]


# Shared singleton used by the routers.
job_store = JobStore()
