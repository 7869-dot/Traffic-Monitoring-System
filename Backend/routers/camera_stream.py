from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os
import cv2
import numpy as np
from io import BytesIO

# Add parent directory to path for imports (so we can import backend modules)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import get_db  # noqa: E402
from camera_streamer import CameraStreamer  # noqa: E402
from vehicle_detector import VehicleDetector  # noqa: E402


router = APIRouter()

# Shared detector + streamer instances
try:
    detector = VehicleDetector()
except Exception:
    detector = None
    print("Warning: Vehicle detector not available in camera_stream router")

try:
    streamer = CameraStreamer()
except Exception:
    streamer = None
    print("Warning: Camera streamer not available")


class DetectionRequest(BaseModel):
    image_path: Optional[str] = None
    camera_id: Optional[str] = None


@router.post("/detect")
async def detect_vehicles(request: DetectionRequest):
    """Detect vehicles in an image or camera stream."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Vehicle detector not available")

    try:
        # Case 1: live camera (ESP32 or local webcam)
        if request.camera_id:
            if streamer is None:
                raise HTTPException(
                    status_code=503, detail="Camera streamer not available"
                )

            # camera_id can be:
            # - "0" or "1" for local webcam
            # - "http://<ESP_IP>/frame" for ESP32-CAM
            frame = streamer.get_snapshot(request.camera_id)

            if frame is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to capture frame from camera_id={request.camera_id}",
                )

            detections = detector.detect(frame=frame)

        # Case 2: static image on disk
        elif request.image_path:
            detections = detector.detect(image_path=request.image_path)

        else:
            raise HTTPException(
                status_code=400,
                detail="Either image_path or camera_id must be provided",
            )

        # Store detections in database
        with get_db() as conn:
            cursor = conn.cursor()
            for detection in detections:
                cursor.execute(
                    """
                    INSERT INTO vehicle_detections (vehicle_type, confidence, camera_id)
                    VALUES (?, ?, ?)
                """,
                    (
                        detection.get("vehicle_type"),
                        detection.get("confidence"),
                        request.camera_id,
                    ),
                )

        return {"detections": detections, "count": len(detections)}

    except HTTPException:
        # re-raise FastAPI HTTP errors
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CameraInfo(BaseModel):
    camera_id: str
    status: str
    resolution: Optional[str] = None
    fps: Optional[float] = None


@router.get("/status")
async def get_camera_status():
    """Get camera stream status."""
    if streamer is None:
        return {"available": False, "status": "Camera streamer not initialized"}

    try:
        cameras = streamer.get_available_cameras()
        return {"available": True, "cameras": cameras}
    except Exception as e:
        return {"available": False, "status": f"Error: {str(e)}"}


@router.get("/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """Stream video from a specific camera."""
    if streamer is None:
        raise HTTPException(status_code=503, detail="Camera streamer not available")

    try:
        def generate_frames():
            for frame in streamer.get_stream(camera_id):
                # Encode frame as JPEG
                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshot/{camera_id}")
async def get_snapshot(camera_id: str):
    """Get a single snapshot from a camera."""
    if streamer is None:
        raise HTTPException(status_code=503, detail="Camera streamer not available")

    try:
        frame = streamer.get_snapshot(camera_id)
        if frame is None:
            raise HTTPException(
                status_code=404, detail="Camera not found or unavailable"
            )

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode image")

        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras")
async def list_cameras():
    """List all available cameras."""
    if streamer is None:
        return {"cameras": []}

    try:
        cameras = streamer.get_available_cameras()
        return {"cameras": cameras}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
