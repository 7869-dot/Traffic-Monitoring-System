import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from routers import arduino, vehicle_detector, camera_stream, video_upload
from database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    print("Database initialized")
    yield
    # Shutdown (nothing to clean up for now)


# Initialize FastAPI app
app = FastAPI(
    title="Traffic Monitoring System API",
    description="API for traffic monitoring with Arduino integration",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration.
# allow_origins=["*"] combined with allow_credentials=True is rejected by
# browsers, so we read explicit origins from the ALLOWED_ORIGINS env var
# (comma-separated). Defaults cover common local dev frontends.
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if _origins_env:
    allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            },
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
