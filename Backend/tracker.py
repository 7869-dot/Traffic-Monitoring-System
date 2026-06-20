"""
Simple Centroid Tracker

A lightweight multi-object tracker used to give an *estimate* of the number of
unique vehicles in a video instead of naively summing per-frame detections
(which counts the same vehicle once per frame and massively over-counts).

It matches detections between processed frames using nearest-centroid matching
within a distance threshold, keeps a short-lived memory of recently seen tracks,
and assigns a stable integer ID to each track. The number of unique IDs ever
created per vehicle class is the unique-vehicle estimate.

This is intentionally simple (no appearance features / Kalman filter). It is a
reasonable heuristic for traffic footage, not a guarantee.
"""

from __future__ import annotations

import math
from typing import Dict, List, Any


class CentroidTracker:
    def __init__(self, max_distance: float = 120.0, max_missed: int = 8):
        """
        Args:
            max_distance: Max pixel distance between centroids to consider them
                the same object across consecutive processed frames.
            max_missed: Number of processed frames a track may go unmatched
                before it is dropped.
        """
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self._next_id = 0
        # active tracks: id -> {"centroid": (x, y), "type": str, "missed": int}
        self._tracks: Dict[int, Dict[str, Any]] = {}
        # unique vehicles ever seen, per class
        self.unique_counts: Dict[str, int] = {}
        self.total_unique = 0

    def _register(self, centroid, vehicle_type: str) -> None:
        self._tracks[self._next_id] = {
            "centroid": centroid,
            "type": vehicle_type,
            "missed": 0,
        }
        self._next_id += 1
        self.unique_counts[vehicle_type] = self.unique_counts.get(vehicle_type, 0) + 1
        self.total_unique += 1

    def update(self, detections: List[Dict[str, Any]]) -> None:
        """Update tracker state with detections from one processed frame."""
        # Build list of incoming centroids + types
        incoming = []
        for det in detections:
            center = det.get("center")
            if not center or len(center) < 2:
                continue
            incoming.append(((float(center[0]), float(center[1])),
                             det.get("vehicle_type", "unknown")))

        # Age all existing tracks; they get reset to missed=0 if matched below.
        for track in self._tracks.values():
            track["missed"] += 1

        used_track_ids = set()
        for centroid, vehicle_type in incoming:
            best_id = None
            best_dist = self.max_distance
            for track_id, track in self._tracks.items():
                if track_id in used_track_ids:
                    continue
                if track["type"] != vehicle_type:
                    continue
                tx, ty = track["centroid"]
                dist = math.hypot(centroid[0] - tx, centroid[1] - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is None:
                # New, previously unseen vehicle
                self._register(centroid, vehicle_type)
            else:
                # Matched an existing track
                self._tracks[best_id]["centroid"] = centroid
                self._tracks[best_id]["missed"] = 0
                used_track_ids.add(best_id)

        # Drop stale tracks
        stale = [tid for tid, t in self._tracks.items() if t["missed"] > self.max_missed]
        for tid in stale:
            del self._tracks[tid]
