"""
ESP32-CAM Frame Upload Router
Handles HTTP POST requests from ESP32-CAM with JPEG frames
"""
import cv2
import numpy as np
import logging
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Request, Header
from fastapi.responses import JSONResponse
import sys
import os
import time
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vehicle_detector import VehicleDetector

logger = logging.getLogger(__name__)

# Debug log file path
DEBUG_LOG_PATH = r"c:\Users\haani\OneDrive\Desktop\Traffic Monitoring System\.cursor\debug.log"

def debug_log(hypothesis_id, location, message, data=None):
    """Write debug log to NDJSON file"""
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write debug log: {e}")

router = APIRouter()

# Initialize vehicle detector (singleton pattern)
_detector_instance = None

def get_detector() -> VehicleDetector:
    """Get or create vehicle detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = VehicleDetector(confidence_threshold=0.5)
        logger.info("Vehicle detector initialized")
    return _detector_instance


@router.post("/upload_frame")
async def upload_frame(
    request: Request,
    x_camera_id: str = Header(default="esp32_cam_1", alias="X-Camera-Id")
):
    """
    Receive a JPEG frame from ESP32-CAM, run vehicle detection and tracking, return results.
    
    Expected request:
    - Method: POST
    - Content-Type: image/jpeg
    - Headers: X-Camera-Id (optional, defaults to "esp32_cam_1")
    - Body: Raw JPEG image bytes
    
    Returns:
    {
        "camera_id": str,
        "vehicle_count": int,
        "detections": [
            {
                "id": int,  # Track ID from DeepSORT
                "label": str,  # "car", "bus", "truck", "motorcycle", "bicycle"
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
        ]
    }
    """
    start_time = time.time()
    
    # #region agent log
    debug_log("B", f"{__file__}:{67}", "upload_frame endpoint called", {
        "camera_id": x_camera_id,
        "method": request.method,
        "url": str(request.url),
        "client": request.client.host if request.client else "unknown",
        "headers": dict(request.headers)
    })
    # #endregion
    
    try:
        # Read raw body bytes
        body_bytes = await request.body()
        frame_size = len(body_bytes)
        
        # #region agent log
        debug_log("C", f"{__file__}:{78}", "Request body read", {
            "frame_size_bytes": frame_size,
            "first_10_bytes": list(body_bytes[:10]) if len(body_bytes) >= 10 else []
        })
        # #endregion
        
        logger.info(f"Received frame from camera '{x_camera_id}': {frame_size} bytes")
        
        if frame_size == 0:
            # #region agent log
            debug_log("D", f"{__file__}:{85}", "Empty frame received", {})
            # #endregion
            raise HTTPException(status_code=400, detail="Empty frame received")
        
        # Decode JPEG image
        try:
            nparr = np.frombuffer(body_bytes, np.uint8)
            # #region agent log
            debug_log("E", f"{__file__}:{92}", "Before JPEG decode", {
                "buffer_size": len(nparr),
                "first_bytes_match_jpeg": list(body_bytes[:3]) == [0xFF, 0xD8, 0xFF] if len(body_bytes) >= 3 else False
            })
            # #endregion
            
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # #region agent log
            debug_log("E", f"{__file__}:{99}", "After JPEG decode", {
                "frame_is_none": frame is None,
                "frame_shape": list(frame.shape) if frame is not None else None
            })
            # #endregion
            
            if frame is None:
                # #region agent log
                debug_log("D", f"{__file__}:{105}", "JPEG decode failed", {
                    "body_first_20_bytes": list(body_bytes[:20])
                })
                # #endregion
                raise HTTPException(
                    status_code=400, 
                    detail="Failed to decode JPEG image. Invalid image data."
                )
            
            logger.debug(f"Decoded frame: {frame.shape[1]}x{frame.shape[0]} pixels")
        except HTTPException:
            raise
        except Exception as e:
            # #region agent log
            debug_log("D", f"{__file__}:{118}", "Exception during JPEG decode", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            # #endregion
            logger.error(f"Error decoding JPEG: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")
        
        # Get detector instance
        detector = get_detector()
        
        # #region agent log
        debug_log("F", f"{__file__}:{130}", "Detector status check", {
            "is_ready": detector.is_ready(),
            "is_tracker_loaded": getattr(detector, 'is_tracker_loaded', False)
        })
        # #endregion
        
        if not detector.is_ready():
            # #region agent log
            debug_log("D", f"{__file__}:{137}", "Detector not ready", {})
            # #endregion
            logger.error("Vehicle detector is not ready")
            raise HTTPException(
                status_code=503, 
                detail="Vehicle detector model not loaded. Please check server logs."
            )
        
        # Run detection and tracking
        try:
            # #region agent log
            debug_log("F", f"{__file__}:{148}", "Starting vehicle detection", {
                "frame_shape": list(frame.shape),
                "use_tracking": True
            })
            # #endregion
            
            detections = detector.detect(frame=frame, use_tracking=True)
            vehicle_count = len(detections)
            
            # #region agent log
            debug_log("F", f"{__file__}:{155}", "Detection complete", {
                "vehicle_count": vehicle_count,
                "detections_count": len(detections)
            })
            # #endregion
            
            # Format detections for response (already in correct format from vehicle_detector)
            formatted_detections = []
            for det in detections:
                formatted_detections.append({
                    "id": det.get("id", 0),
                    "label": det.get("label", "unknown"),
                    "confidence": det.get("confidence", 0.0),
                    "bbox": det.get("bbox", [0, 0, 0, 0])
                })
            
            inference_time = time.time() - start_time
            logger.info(
                f"Detection complete for camera '{x_camera_id}': "
                f"{vehicle_count} vehicles detected in {inference_time:.3f}s"
            )
            
            # Prepare response
            response = {
                "camera_id": x_camera_id,
                "vehicle_count": vehicle_count,
                "detections": formatted_detections,
                "processing_time_ms": round(inference_time * 1000, 2)
            }
            
            # #region agent log
            debug_log("F", f"{__file__}:{172}", "Sending response", {
                "response_size": len(str(response)),
                "vehicle_count": vehicle_count
            })
            # #endregion
            
            return JSONResponse(content=response)
            
        except Exception as e:
            # #region agent log
            debug_log("D", f"{__file__}:{180}", "Exception during detection", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            # #endregion
            logger.error(f"Error during vehicle detection: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")
    
    except HTTPException as e:
        # #region agent log
        debug_log("D", f"{__file__}:{188}", "HTTPException raised", {
            "status_code": e.status_code,
            "detail": e.detail
        })
        # #endregion
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # #region agent log
        debug_log("D", f"{__file__}:{195}", "Unexpected exception", {
            "error": str(e),
            "error_type": type(e).__name__
        })
        # #endregion
        logger.error(f"Unexpected error processing frame: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/status")
async def get_detector_status():
    """
    Get the status of the vehicle detector
    
    Returns:
    {
        "detector_ready": bool,
        "tracker_ready": bool,
        "status": str
    }
    """
    detector = get_detector()
    
    return {
        "detector_ready": detector.is_ready(),
        "tracker_ready": detector.is_tracker_loaded if hasattr(detector, 'is_tracker_loaded') else False,
        "status": "ready" if detector.is_ready() else "not ready"
    }

