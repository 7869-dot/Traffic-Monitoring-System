"""
Video Upload and Processing Router

Provides endpoints to:
- Upload a video file and process it synchronously (small clips)
- Upload a video and process it in the background, polling a job for progress
- Return vehicle counts and basic metadata
"""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import sys

# Ensure parent directory is on path so we can import backend modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Video_processor import VideoProcessor, VideoProcessingSummary  # noqa: E402
from jobs import job_store  # noqa: E402


router = APIRouter()


UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def _summary_to_dict(filename: str, summary: VideoProcessingSummary) -> dict:
    return {
        "filename": filename,
        "total_frames": summary.total_frames,
        "processed_frames": summary.processed_frames,
        "frame_sample_rate": summary.frame_sample_rate,
        "duration_sec": summary.duration_sec,
        "fps": summary.fps,
        # Most vehicles of each type seen at the same time in one frame.
        "peak_counts": summary.peak_counts,
        # Estimated number of distinct vehicles across the whole clip.
        "unique_estimate": summary.unique_estimate,
        # Average vehicles per processed frame.
        "avg_per_frame": summary.avg_per_frame,
    }


def _validate_and_save(file: UploadFile) -> str:
    """Validate the upload and persist it under a safe unique name."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    _, ext = os.path.splitext(file.filename.lower())
    if ext and ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
        )

    safe_ext = ext if ext in ALLOWED_EXT else ".mp4"
    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{safe_ext}")
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")
    finally:
        file.file.close()

    return save_path


@router.post("/upload-and-process")
async def upload_and_process_video(
    file: UploadFile = File(...),
    frame_sample_rate: int = 5,
    max_frames: Optional[int] = None,
):
    """
    Upload a video file, process it synchronously, and return vehicle counts.

    Best for short clips; for longer videos use ``/upload-async`` + ``/job``.
    """
    original_name = file.filename
    save_path = _validate_and_save(file)

    try:
        processor = VideoProcessor(frame_sample_rate=frame_sample_rate)
        summary = processor.process_video(
            video_path=save_path,
            max_frames=max_frames,
            collect_per_frame=False,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to process video: {exc}")
    finally:
        # Don't let uploaded files accumulate on disk.
        try:
            os.remove(save_path)
        except OSError:
            pass

    return JSONResponse(content=_summary_to_dict(original_name, summary))


def _run_job(job_id: str, save_path: str, original_name: str,
             frame_sample_rate: int, max_frames: Optional[int]) -> None:
    """Background worker: process the video and update the job store."""
    job_store.update(job_id, status="processing")
    try:
        processor = VideoProcessor(frame_sample_rate=frame_sample_rate)
        summary = processor.process_video(
            video_path=save_path,
            max_frames=max_frames,
            collect_per_frame=False,
            progress_callback=lambda p: job_store.set_progress(job_id, p),
        )
        job_store.update(
            job_id,
            status="done",
            progress=1.0,
            result=_summary_to_dict(original_name, summary),
        )
    except Exception as exc:  # pragma: no cover - defensive
        job_store.update(job_id, status="error", error=str(exc))
    finally:
        # Best-effort cleanup of the uploaded file.
        try:
            os.remove(save_path)
        except OSError:
            pass
        job_store.cleanup()


@router.post("/upload-async")
async def upload_async(
    file: UploadFile = File(...),
    frame_sample_rate: int = 5,
    max_frames: Optional[int] = None,
):
    """
    Upload a video and process it on a background thread.

    Returns a ``job_id``; poll ``GET /api/video/job/{job_id}`` for progress and
    the final result.
    """
    original_name = file.filename
    save_path = _validate_and_save(file)

    job_id = job_store.create(original_name)
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, save_path, original_name, frame_sample_rate, max_frames),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@router.get("/job/{job_id}")
async def get_job(job_id: str):
    """Get the status / progress / result of a background processing job."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "progress": round(job["progress"], 3),
        "result": job["result"],
        "error": job["error"],
    }
