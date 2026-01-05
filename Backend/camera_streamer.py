"""
Camera Streamer Module
Handles camera streaming and snapshot capture

Now supports:
- Local webcams via OpenCV (camera_id like "0", "1", ...)
- ESP32-CAM HTTP endpoints (camera_id is a URL like "http://<IP>/frame")
"""

import cv2
import numpy as np
import requests
from typing import List, Dict, Optional, Generator, Any
import threading
import time


class CameraStreamer:
    """Manages camera streams for traffic monitoring"""

    def __init__(self):
        """Initialize camera streamer"""
        # internal registry: key = camera_id (string)
        # value = dict with fields:
        #   type: "local" or "esp32"
        #   cap:  cv2.VideoCapture (for local)
        #   url:  str (for esp32 HTTP frame endpoint)
        self.cameras: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def get_available_cameras(self) -> List[Dict]:
        """
        Get list of available local cameras (USB/webcams)

        Returns:
            List of camera information dictionaries
        """
        available = []
        # Test first 5 camera indices for LOCAL webcams only
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    available.append(
                        {
                            "camera_id": str(i),
                            "index": i,
                            "resolution": f"{width}x{height}",
                            "fps": fps,
                            "status": "available",
                            "type": "local",
                        }
                    )
                cap.release()
        return available

    def _is_url(self, camera_id: str) -> bool:
        """Simple check: treat camera_id as URL if it starts with http/https"""
        return camera_id.startswith("http://") or camera_id.startswith("https://")

    def open_camera(self, camera_id: str) -> bool:
        """
        Open a camera for streaming.

        Args:
            camera_id:
                - "0", "1", ...  → local webcam index
                - "http://<IP>/frame" → ESP32-CAM HTTP frame endpoint

        Returns:
            True if camera opened / registered successfully
        """
        try:
            with self.lock:
                if camera_id in self.cameras:
                    # already opened/registered
                    return True

                # --- ESP32 / HTTP camera ---
                if self._is_url(camera_id):
                    # Just store the URL; frames will be fetched via HTTP
                    self.cameras[camera_id] = {
                        "type": "esp32",
                        "url": camera_id,
                    }
                    return True

                # --- Local webcam via OpenCV ---
                index = int(camera_id)
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    self.cameras[camera_id] = {
                        "type": "local",
                        "cap": cap,
                        "index": index,
                    }
                    return True

                return False

        except (ValueError, Exception) as e:
            print(f"Error opening camera {camera_id}: {e}")
            return False

    def close_camera(self, camera_id: str):
        """Close a camera"""
        try:
            with self.lock:
                cam = self.cameras.get(camera_id)
                if not cam:
                    return

                if cam.get("type") == "local":
                    cap = cam.get("cap")
                    if cap is not None:
                        cap.release()

                # For "esp32" type, nothing special to release
                del self.cameras[camera_id]

        except Exception as e:
            print(f"Error closing camera {camera_id}: {e}")

    def _read_local_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Read a single frame from a local webcam"""
        cam = self.cameras.get(camera_id)
        if not cam or cam.get("type") != "local":
            return None

        cap = cam.get("cap")
        if cap is None:
            return None

        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def _read_esp32_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Read a single frame from ESP32-CAM HTTP endpoint.

        camera_id is expected to be a URL like "http://<IP>/frame",
        which returns a JPEG image.
        """
        cam = self.cameras.get(camera_id)
        if not cam or cam.get("type") != "esp32":
            return None

        url = cam.get("url")
        if not url:
            return None

        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code != 200:
                return None

            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"Error reading ESP32 frame from {url}: {e}")
            return None

    def get_stream(self, camera_id: str) -> Generator[np.ndarray, None, None]:
        """
        Get video stream generator for a camera

        Args:
            camera_id:
                - "0", "1", ... (local webcam)
                - "http://<IP>/frame" (ESP32-CAM)

        Yields:
            Video frames (numpy arrays)
        """
        if not self.open_camera(camera_id):
            raise ValueError(f"Camera {camera_id} not available")

        while True:
            with self.lock:
                cam = self.cameras.get(camera_id)
                if not cam:
                    break
                cam_type = cam.get("type")

            if cam_type == "local":
                frame = self._read_local_frame(camera_id)
            elif cam_type == "esp32":
                frame = self._read_esp32_frame(camera_id)
            else:
                frame = None

            if frame is None:
                # stop streaming if we cannot read frame
                break

            yield frame
            time.sleep(0.03)  # ~30 FPS target

    def get_snapshot(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get a single snapshot from a camera

        Args:
            camera_id:
                - "0", "1", ... (local webcam)
                - "http://<IP>/frame" (ESP32-CAM)

        Returns:
            Frame as numpy array or None if unavailable
        """
        if not self.open_camera(camera_id):
            return None

        with self.lock:
            cam = self.cameras.get(camera_id)
            if not cam:
                return None
            cam_type = cam.get("type")

        if cam_type == "local":
            return self._read_local_frame(camera_id)
        elif cam_type == "esp32":
            return self._read_esp32_frame(camera_id)
        else:
            return None

    def __del__(self):
        """Cleanup on deletion"""
        with self.lock:
            for cam_id, cam in list(self.cameras.items()):
                if cam.get("type") == "local":
                    cap = cam.get("cap")
                    if cap is not None:
                        cap.release()
            self.cameras.clear()
