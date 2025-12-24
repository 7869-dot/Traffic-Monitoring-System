from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os
import cv2
import numpy as np
from io import BytesIO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database import get_db

router = APIRouter()

# Import camera streamer
try:
    from camera_streamer import CameraStreamer
    streamer = CameraStreamer()
except ImportError:
    streamer = None
    print("Warning: Camera streamer not available")

class CameraInfo(BaseModel):
    camera_id: str
    status: str
    resolution: Optional[str] = None
    fps: Optional[float] = None

@router.get("/status")
async def get_camera_status():
    """Get camera stream status"""
    if streamer is None:
        return {"available": False, "status": "Camera streamer not initialized"}
    
    try:
        cameras = streamer.get_available_cameras()
        return {"available": True, "cameras": cameras}
    except Exception as e:
        return {"available": False, "status": f"Error: {str(e)}"}

@router.get("/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """Stream video from a specific camera"""
    if streamer is None:
        raise HTTPException(status_code=503, detail="Camera streamer not available")
    
    try:
        def generate_frames():
            for frame in streamer.get_stream(camera_id):
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/snapshot/{camera_id}")
async def get_snapshot(camera_id: str):
    """Get a single snapshot from a camera"""
    if streamer is None:
        raise HTTPException(status_code=503, detail="Camera streamer not available")
    
    try:
        frame = streamer.get_snapshot(camera_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="Camera not found or unavailable")
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cameras")
async def list_cameras():
    """List all available cameras"""
    if streamer is None:
        return {"cameras": []}
    
    try:
        cameras = streamer.get_available_cameras()
        return {"cameras": cameras}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

