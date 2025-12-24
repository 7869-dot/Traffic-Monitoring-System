from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database import get_db

router = APIRouter()

# Import vehicle detector (you'll implement this)
try:
    from vehicle_detector import VehicleDetector
    detector = VehicleDetector()
except ImportError:
    detector = None
    print("Warning: Vehicle detector not available")

class DetectionResult(BaseModel):
    vehicle_type: str
    confidence: float
    timestamp: datetime
    camera_id: Optional[str] = None

class DetectionRequest(BaseModel):
    image_path: Optional[str] = None
    camera_id: Optional[str] = None

@router.get("/status")
async def get_detector_status():
    """Get vehicle detector status"""
    if detector is None:
        return {"available": False, "status": "Vehicle detector not initialized"}
    
    try:
        is_ready = detector.is_ready()
        return {"available": is_ready, "status": "ready" if is_ready else "not ready"}
    except Exception as e:
        return {"available": False, "status": f"Error: {str(e)}"}

@router.post("/detect")
async def detect_vehicles(request: DetectionRequest):
    """Detect vehicles in an image or camera stream"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Vehicle detector not available")
    
    try:
        detections = detector.detect(
            image_path=request.image_path,
            camera_id=request.camera_id
        )
        
        # Store detections in database
        with get_db() as conn:
            cursor = conn.cursor()
            for detection in detections:
                cursor.execute("""
                    INSERT INTO vehicle_detections (vehicle_type, confidence, camera_id)
                    VALUES (?, ?, ?)
                """, (detection.get("vehicle_type"), detection.get("confidence"), request.camera_id))
        
        return {"detections": detections, "count": len(detections)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detections")
async def get_recent_detections(limit: int = 100):
    """Get recent vehicle detections"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM vehicle_detections
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        detections = [dict(row) for row in cursor.fetchall()]
    return {"detections": detections, "count": len(detections)}

@router.get("/stats")
async def get_detection_stats():
    """Get vehicle detection statistics"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Total detections
        cursor.execute("SELECT COUNT(*) as total FROM vehicle_detections")
        total = cursor.fetchone()["total"]
        
        # Detections by vehicle type
        cursor.execute("""
            SELECT vehicle_type, COUNT(*) as count
            FROM vehicle_detections
            GROUP BY vehicle_type
        """)
        by_type = {row["vehicle_type"]: row["count"] for row in cursor.fetchall()}
        
        # Recent detections (last hour)
        cursor.execute("""
            SELECT COUNT(*) as recent
            FROM vehicle_detections
            WHERE timestamp > datetime('now', '-1 hour')
        """)
        recent = cursor.fetchone()["recent"]
    
    return {
        "total_detections": total,
        "recent_detections": recent,
        "by_vehicle_type": by_type
    }

