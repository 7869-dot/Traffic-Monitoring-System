from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import logging
import socket
import json
from datetime import datetime

from routers import arduino, vehicle_detector, camera_stream, esp32
from database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# Middleware to log all incoming requests
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # #region agent log
        debug_log("A", f"{__file__}:{47}", "Incoming HTTP request", {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "headers": dict(request.headers),
            "client": request.client.host if request.client else "unknown"
        })
        # #endregion
        
        response = await call_next(request)
        
        # #region agent log
        debug_log("A", f"{__file__}:{56}", "HTTP response sent", {
            "status_code": response.status_code,
            "path": request.url.path
        })
        # #endregion
        
        return response

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Monitoring System API",
    description="API for traffic monitoring with Arduino integration",
    version="1.0.0"
)

# Add request logging middleware first
app.add_middleware(RequestLoggingMiddleware)

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
    logger.info("Database initialized")
    
    # Get server IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        server_ip = s.getsockname()[0]
        s.close()
    except:
        server_ip = local_ip
    
    # #region agent log
    debug_log("A", f"{__file__}:{75}", "Server startup", {
        "hostname": hostname,
        "local_ip": local_ip,
        "server_ip": server_ip,
        "port": 8000,
        "endpoint": f"http://{server_ip}:8000/upload_frame"
    })
    # #endregion
    
    logger.info(f"FastAPI server starting - ESP32-CAM frame upload endpoint available at POST /upload_frame")
    logger.info(f"Server accessible at: http://{server_ip}:8000")
    logger.info(f"Local access: http://localhost:8000")

# Include routers
app.include_router(arduino.router, prefix="/api/arduino", tags=["Arduino"])
app.include_router(vehicle_detector.router, prefix="/api/vehicles", tags=["Vehicle Detection"])
app.include_router(camera_stream.router, prefix="/api/camera", tags=["Camera Stream"])
# ESP32 frame upload endpoint at root level (no prefix) to match Arduino code
app.include_router(esp32.router, tags=["ESP32-CAM"])

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
                "esp32_upload_frame": "/upload_frame"
            }
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

