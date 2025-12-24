"""
Camera Streamer Module
Handles camera streaming and snapshot capture
"""
import cv2
from typing import List, Dict, Optional, Generator
import threading
import time

class CameraStreamer:
    """Manages camera streams for traffic monitoring"""
    
    def __init__(self):
        """Initialize camera streamer"""
        self.cameras: Dict[int, cv2.VideoCapture] = {}
        self.camera_info: Dict[int, Dict] = {}
        self.lock = threading.Lock()
    
    def get_available_cameras(self) -> List[Dict]:
        """
        Get list of available cameras
        
        Returns:
            List of camera information dictionaries
        """
        available = []
        # Test first 5 camera indices
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    available.append({
                        "camera_id": str(i),
                        "index": i,
                        "resolution": f"{width}x{height}",
                        "fps": fps,
                        "status": "available"
                    })
                cap.release()
        return available
    
    def open_camera(self, camera_id: str) -> bool:
        """
        Open a camera for streaming
        
        Args:
            camera_id: Camera identifier (usually index as string)
        
        Returns:
            True if camera opened successfully
        """
        try:
            index = int(camera_id)
            with self.lock:
                if index in self.cameras:
                    return True
                
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    self.cameras[index] = cap
                    return True
                return False
        except (ValueError, Exception) as e:
            print(f"Error opening camera {camera_id}: {e}")
            return False
    
    def close_camera(self, camera_id: str):
        """Close a camera"""
        try:
            index = int(camera_id)
            with self.lock:
                if index in self.cameras:
                    self.cameras[index].release()
                    del self.cameras[index]
        except (ValueError, Exception) as e:
            print(f"Error closing camera {camera_id}: {e}")
    
    def get_stream(self, camera_id: str) -> Generator:
        """
        Get video stream generator for a camera
        
        Args:
            camera_id: Camera identifier
        
        Yields:
            Video frames (numpy arrays)
        """
        if not self.open_camera(camera_id):
            raise ValueError(f"Camera {camera_id} not available")
        
        index = int(camera_id)
        while True:
            with self.lock:
                if index not in self.cameras:
                    break
                cap = self.cameras[index]
                ret, frame = cap.read()
            
            if not ret:
                break
            
            yield frame
            time.sleep(0.03)  # ~30 FPS
    
    def get_snapshot(self, camera_id: str) -> Optional:
        """
        Get a single snapshot from a camera
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Frame as numpy array or None if unavailable
        """
        if not self.open_camera(camera_id):
            return None
        
        index = int(camera_id)
        with self.lock:
            if index not in self.cameras:
                return None
            cap = self.cameras[index]
            ret, frame = cap.read()
        
        return frame if ret else None
    
    def __del__(self):
        """Cleanup on deletion"""
        with self.lock:
            for cap in self.cameras.values():
                cap.release()

