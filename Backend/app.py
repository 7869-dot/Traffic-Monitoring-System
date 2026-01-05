from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np

from routers import arduino, vehicle_detector, camera_stream, video_upload
from database import init_db, get_db

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Monitoring System API",
    description="API for traffic monitoring with Arduino integration",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("Database initialized")

# Include routers
app.include_router(arduino.router, prefix="/api/arduino", tags=["Arduino"])
app.include_router(vehicle_detector.router, prefix="/api/vehicles", tags=["Vehicle Detection"])
app.include_router(camera_stream.router, prefix="/api/camera", tags=["Camera Stream"])
app.include_router(video_upload.router, prefix="/api/video", tags=["Video Upload"])

# Root endpoint
@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "Traffic Monitoring System API",
            "version": "1.0.0",
            "endpoints": {
                "arduino": "/api/arduino",
                "vehicles": "/api/vehicles",
                "camera": "/api/camera",
                "video": "/api/video",
                "esp32_upload": "/upload_frame",
                "esp32_test": "/esp32/test",
                "health": "/health"
            }
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ESP32 connection test endpoint
@app.get("/esp32/test")
async def esp32_test():
    """Simple endpoint for ESP32 to test connectivity"""
    return {
        "status": "ok",
        "message": "ESP32 can reach FastAPI server",
        "detector_ready": vehicle_detector.detector.is_ready() if vehicle_detector.detector else False
    }


# ESP32-CAM frame upload endpoint (root level to match Arduino code)
@app.post("/upload_frame")
async def upload_frame(request: Request):
    """
    Receive JPEG frame from ESP32-CAM and analyze it for vehicles.
    
    This endpoint:
    1. Receives raw JPEG image bytes from ESP32
    2. Decodes the image
    3. Runs vehicle detection
    4. Stores results in database
    5. Returns detection results
    
    Expected: POST with Content-Type: image/jpeg and raw JPEG bytes in body
    Optional header: X-Camera-Id to identify the camera
    """
    try:
        # Get camera ID from header (ESP32 sends "X-Camera-Id: esp32_cam_1")
        camera_id = request.headers.get("X-Camera-Id", "esp32-cam")
        
        # Read raw image bytes from request body
        image_data = await request.body()
        
        if not image_data:
            print(f"[ESP32] ERROR: No image data received from {camera_id}")
            raise HTTPException(status_code=400, detail="No image data received")
        
        print(f"[ESP32] Received frame from {camera_id}: {len(image_data)} bytes")
        
        # Convert bytes to numpy array and decode JPEG
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print(f"[ESP32] ERROR: Failed to decode JPEG from {camera_id}")
            raise HTTPException(status_code=400, detail="Failed to decode JPEG image")
        
        print(f"[ESP32] Decoded frame: {frame.shape[1]}x{frame.shape[0]} pixels")
        
        # Use detector instance from vehicle_detector router module
        detector = vehicle_detector.detector
        
        if detector is None:
            print("[ESP32] ERROR: Vehicle detector not available")
            raise HTTPException(status_code=503, detail="Vehicle detector not available")
        
        if not detector.is_ready():
            print("[ESP32] ERROR: Vehicle detector not ready")
            raise HTTPException(status_code=503, detail="Vehicle detector not ready")
        
        # Detect vehicles in the frame
        detections = detector.detect(frame=frame)
        
        # Count vehicles by type
        counts = detector.count_vehicles(detections)
        
        print(f"[ESP32] Detected {len(detections)} vehicles: {counts}")
        
        # Store detections in database
        with get_db() as conn:
            cursor = conn.cursor()
            for detection in detections:
                cursor.execute("""
                    INSERT INTO vehicle_detections (
                        vehicle_type, 
                        confidence, 
                        camera_id
                    )
                    VALUES (?, ?, ?)
                """, (
                    detection.get("vehicle_type"),
                    detection.get("confidence"),
                    camera_id
                ))
        
        # Prepare response
        response_data = {
            "success": True,
            "message": "Frame analyzed successfully",
            "detections": detections,
            "counts": counts,
            "total_vehicles": len(detections),
            "camera_id": camera_id,
            "frame_size": {
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0])
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to process frame: {str(e)}"
        print(f"[ESP32] ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

