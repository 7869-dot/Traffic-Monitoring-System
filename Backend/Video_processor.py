"""
Video Processor Module
Handles offline video processing and vehicle counting
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import cv2
import numpy as np

from vehicle_detector import VehicleDetector


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
    overall_counts: Dict[str, int]
    per_frame_results: List[FrameDetectionResult] = field(default_factory=list)


class VideoProcessor:
    """
    Processes video files offline and counts vehicles using VehicleDetector.

    Typical usage:

        processor = VideoProcessor()
        summary = processor.process_video("path/to/video.mp4")
        print(summary.overall_counts)
    """

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        frame_sample_rate: int = 5,
    ) -> None:
        """
        Initialize VideoProcessor.

        Args:
            detector: Optional VehicleDetector instance. If None, a new one is created.
            frame_sample_rate: Process every Nth frame (default: 5) for performance.
        """
        self.detector = detector or VehicleDetector()
        self.frame_sample_rate = max(1, int(frame_sample_rate))

    def _validate_video_path(self, video_path: str) -> None:
        """Validate that the video path exists and is a file."""
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
    ) -> VideoProcessingSummary:
        """
        Process a video file and count vehicles.

        Args:
            video_path: Path to the video file.
            max_frames: Optional limit on number of frames to process.
            collect_per_frame: If True, store per-frame detection details.

        Returns:
            VideoProcessingSummary with overall and per-frame counts.
        """
        self._validate_video_path(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            duration_sec = float(total_frames / fps) if fps > 0 else 0.0

            # Overall counts aggregated across processed frames
            overall_counts: Dict[str, int] = {
                "car": 0,
                "truck": 0,
                "bus": 0,
                "motorcycle": 0,
                "bicycle": 0,
                "total": 0,
            }

            per_frame_results: List[FrameDetectionResult] = []

            frame_index = 0
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Stop if we've reached the max_frames limit
                if max_frames is not None and processed_frames >= max_frames:
                    break

                # Only process every Nth frame
                if frame_index % self.frame_sample_rate != 0:
                    frame_index += 1
                    continue

                # Run detection on this frame
                detections = self.detector.detect(frame=frame)
                counts = self.detector.count_vehicles(detections)

                # Aggregate into overall counts
                for key in overall_counts.keys():
                    if key in counts:
                        overall_counts[key] += counts[key]

                # Optionally collect per-frame info
                if collect_per_frame:
                    timestamp_sec = float(frame_index / fps) if fps > 0 else 0.0
                    per_frame_results.append(
                        FrameDetectionResult(
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            detections=detections,
                            counts=counts,
                        )
                    )

                processed_frames += 1
                frame_index += 1

        finally:
            cap.release()

        return VideoProcessingSummary(
            video_path=os.path.abspath(video_path),
            total_frames=total_frames,
            processed_frames=processed_frames,
            frame_sample_rate=self.frame_sample_rate,
            duration_sec=duration_sec,
            fps=fps,
            overall_counts=overall_counts,
            per_frame_results=per_frame_results,
        )



