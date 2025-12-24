from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from routers import arduino, vehicle_detector, camera_stream
from database import init_db

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
                "camera": "/api/camera"
            }
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

