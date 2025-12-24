"""
Video Upload and Processing Router

Provides endpoints to:
- Upload a video file
- Process it offline with VideoProcessor
- Return vehicle counts and basic metadata
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import sys

# Ensure parent directory is on path so we can import backend modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Video_processor import VideoProcessor  # noqa: E402


router = APIRouter()


UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload-and-process")
async def upload_and_process_video(
    file: UploadFile = File(...),
    frame_sample_rate: int = 5,
    max_frames: Optional[int] = None,
):
    """
    Upload a video file, process it, and return vehicle counts.

    - Saves the uploaded file to `Backend/uploads/`
    - Runs VideoProcessor over the saved file
    - Returns overall vehicle counts and basic video metadata
    """
    # Basic validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    # Simple extension check (optional)
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext and ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed_ext))}",
        )

    # Save file to uploads directory
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")
    finally:
        await file.close()

    # Process video
    try:
        processor = VideoProcessor(frame_sample_rate=frame_sample_rate)
        summary = processor.process_video(
            video_path=save_path,
            max_frames=max_frames,
            collect_per_frame=False,  # summary only for API response (lighter)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to process video: {exc}")

    response_data = {
        "video_path": summary.video_path,
        "total_frames": summary.total_frames,
        "processed_frames": summary.processed_frames,
        "frame_sample_rate": summary.frame_sample_rate,
        "duration_sec": summary.duration_sec,
        "fps": summary.fps,
        "overall_counts": summary.overall_counts,
    }

    return JSONResponse(content=response_data)



