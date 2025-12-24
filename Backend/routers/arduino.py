from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database import get_db

router = APIRouter()

# Import Arduino controller
try:
    from arduino_controller import ArduinoController
    arduino_controller = ArduinoController()
except ImportError:
    arduino_controller = None
    print("Warning: Arduino controller not available")

class ArduinoCommand(BaseModel):
    action: str
    value: Optional[int] = None
    duration: Optional[int] = None

class ArduinoStatus(BaseModel):
    connected: bool
    status: str
    last_command: Optional[str] = None

@router.get("/status")
async def get_arduino_status():
    """Get Arduino connection status"""
    if arduino_controller is None:
        return ArduinoStatus(
            connected=False,
            status="Arduino controller not initialized"
        )
    
    try:
        is_connected = arduino_controller.is_connected()
        return ArduinoStatus(
            connected=is_connected,
            status="connected" if is_connected else "disconnected"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/command")
async def send_arduino_command(command: ArduinoCommand):
    """Send command to Arduino"""
    if arduino_controller is None:
        raise HTTPException(status_code=503, detail="Arduino controller not available")
    
    try:
        result = arduino_controller.send_command(
            action=command.action,
            value=command.value,
            duration=command.duration
        )
        
        # Log the command to database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO arduino_logs (action, status, details)
                VALUES (?, ?, ?)
            """, (command.action, "success", str(result)))
        
        return {"success": True, "result": result}
    except Exception as e:
        # Log error to database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO arduino_logs (action, status, details)
                VALUES (?, ?, ?)
            """, (command.action, "error", str(e)))
        
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_arduino_logs(limit: int = 50):
    """Get recent Arduino control logs"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM arduino_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        logs = [dict(row) for row in cursor.fetchall()]
    return {"logs": logs}

