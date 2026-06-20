"""
Video Processor Module
Handles offline video processing and vehicle counting.

Counting strategy
-----------------
Running a detector on every frame and summing the per-frame counts massively
over-counts vehicles (a car visible for 100 frames would be counted ~100 times).
Instead this module reports three honest metrics:

- ``peak_counts``    : the most vehicles of each type seen *simultaneously* in a
                       single processed frame.
- ``unique_estimate``: an estimate of distinct vehicles over the whole clip,
                       produced by a lightweight centroid tracker.
- ``avg_per_frame``  : average vehicles per processed frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

import cv2

from vehicle_detector import VehicleDetector
from tracker import CentroidTracker


@dataclass
class FrameDetectionResult:
    """Per-frame detection result."""

    frame_index: int
    timestamp_sec: float
    detections: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class VideoProcessingSummary:
    """Summary of video processing and vehicle counts."""

    video_path: str
    total_frames: int
    processed_frames: int
    frame_sample_rate: int
    duration_sec: float
    fps: float
    peak_counts: Dict[str, int]
    unique_estimate: Dict[str, int]
    avg_per_frame: Dict[str, float]
    per_frame_results: List[FrameDetectionResult] = field(default_factory=list)


class VideoProcessor:
    """Processes video files offline and counts vehicles using VehicleDetector."""

    # Vehicle classes we report on (keeps response shape stable even at zero).
    VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        frame_sample_rate: int = 5,
    ) -> None:
        """
        Args:
            detector: Optional VehicleDetector instance. If None, a new one is created.
            frame_sample_rate: Process every Nth frame (default: 5) for performance.
        """
        self.detector = detector or VehicleDetector()
        self.frame_sample_rate = max(1, int(frame_sample_rate))

    def _validate_video_path(self, video_path: str) -> None:
        if not video_path:
            raise ValueError("video_path is required")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Not a file: {video_path}")

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        collect_per_frame: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> VideoProcessingSummary:
        """
        Process a video file and count vehicles.

        Args:
            video_path: Path to the video file.
            max_frames: Optional limit on number of *processed* frames.
            collect_per_frame: If True, store per-frame detection details.
            progress_callback: Optional callable invoked with a 0.0-1.0 fraction
                as processing advances (used for async progress reporting).
        """
        self._validate_video_path(video_path)

        if not self.detector.is_ready():
            raise RuntimeError(
                "Vehicle detector model is not loaded. Ensure 'ultralytics' is "
                "installed and the YOLO weights are available."
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Keys we always report (per-class plus a total).
        count_keys = self.VEHICLE_CLASSES + ["total"]

        peak_counts: Dict[str, int] = {k: 0 for k in count_keys}
        sum_counts: Dict[str, int] = {k: 0 for k in count_keys}
        tracker = CentroidTracker()
        per_frame_results: List[FrameDetectionResult] = []

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            duration_sec = float(total_frames / fps) if fps > 0 else 0.0

            # How many frames will actually be read (for progress reporting).
            frames_to_read = total_frames
            if max_frames is not None:
                frames_to_read = min(
                    frames_to_read or (max_frames * self.frame_sample_rate),
                    max_frames * self.frame_sample_rate,
                )

            frame_index = 0
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only process every Nth frame.
                if frame_index % self.frame_sample_rate != 0:
                    frame_index += 1
                    continue

                if max_frames is not None and processed_frames >= max_frames:
                    break

                detections = self.detector.detect(frame=frame)
                counts = self.detector.count_vehicles(detections)

                # Peak = max simultaneous; sum used for averaging.
                for key in count_keys:
                    value = counts.get(key, 0)
                    peak_counts[key] = max(peak_counts[key], value)
                    sum_counts[key] += value

                # Feed the tracker for unique-vehicle estimation.
                tracker.update(detections)

                if collect_per_frame:
                    timestamp_sec = float(frame_index / fps) if fps > 0 else 0.0
                    per_frame_results.append(
                        FrameDetectionResult(
                            frame_index=frame_index,
                            timestamp_sec=round(timestamp_sec, 3),
                            detections=detections,
                            counts=counts,
                        )
                    )

                processed_frames += 1
                frame_index += 1

                if progress_callback and frames_to_read:
                    progress_callback(min(frame_index / frames_to_read, 0.99))

        finally:
            cap.release()

        if progress_callback:
            progress_callback(1.0)

        # Build unique estimate from tracker (ensure all classes present).
        unique_estimate: Dict[str, int] = {k: 0 for k in self.VEHICLE_CLASSES}
        for vtype, n in tracker.unique_counts.items():
            if vtype in unique_estimate:
                unique_estimate[vtype] += n
        unique_estimate["total"] = sum(unique_estimate.values())

        # Average vehicles per processed frame.
        avg_per_frame: Dict[str, float] = {}
        for key in count_keys:
            avg_per_frame[key] = round(
                sum_counts[key] / processed_frames, 2
            ) if processed_frames else 0.0

        return VideoProcessingSummary(
            video_path=os.path.abspath(video_path),
            total_frames=total_frames,
            processed_frames=processed_frames,
            frame_sample_rate=self.frame_sample_rate,
            duration_sec=round(duration_sec, 2),
            fps=round(fps, 2),
            peak_counts=peak_counts,
            unique_estimate=unique_estimate,
            avg_per_frame=avg_per_frame,
            per_frame_results=per_frame_results,
        )
