"""
Vehicle Detector Module
Handles vehicle detection using computer vision with YOLOv8 + DeepSORT tracking
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
import os

# ---- YOLO import ----
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

# ---- DeepSORT import ----
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # pyright: ignore[reportMissingImports]
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("Warning: deep-sort-realtime not installed. Install with: pip install deep-sort-realtime")


class VehicleDetector:
    """Detects and tracks vehicles in images or video streams using YOLOv8 + DeepSORT"""

    # COCO dataset class IDs for vehicles
    # COCO classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
    VEHICLE_CLASSES = {
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        enable_tracking: bool = True,
    ):
        """
        Initialize vehicle detector

        Args:
            model_path: Path to YOLO model weights (optional, uses YOLOv8n by default)
            confidence_threshold: Minimum confidence score for detections (default: 0.5)
            enable_tracking: If True and deep-sort-realtime is installed, enable DeepSORT tracking
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
        self.is_model_loaded = False

        # DeepSORT tracker (optional)
        self.enable_tracking = enable_tracking
        self.tracker: Optional[DeepSort] = None

        if not YOLO_AVAILABLE:
            print("Error: ultralytics package not available. Please install it first.")
            return

        # Load model
        self._load_model()

        # Init tracker
        if self.enable_tracking:
            self._init_tracker()

    def _load_model(self):
        """Load YOLOv8 detection model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model if path provided
                self.model = YOLO(self.model_path)
                print(f"Loaded custom model from {self.model_path}")
            else:
                # Load pre-trained YOLOv8n (nano) model - fastest and smallest
                # Will auto-download on first use
                self.model = YOLO("yolov8n.pt")
                print("Loaded YOLOv8n pre-trained model")

            self.is_model_loaded = True
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_model_loaded = False
            self.model = None

    def _init_tracker(self):
        """Initialize DeepSORT tracker (optional)"""
        if not DEEPSORT_AVAILABLE:
            print("DeepSORT not available, tracking disabled.")
            self.tracker = None
            return

        try:
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                max_iou_distance=0.7,
                embedder="mobilenet",
                half=True,
            )
            print("DeepSORT tracker initialized")
        except Exception as e:
            print(f"Error initializing DeepSORT tracker: {e}")
            self.tracker = None

    def is_ready(self) -> bool:
        """Check if detector is ready"""
        return self.is_model_loaded and self.model is not None
    
    def is_tracking_enabled(self) -> bool:
        """Check if DeepSORT tracking is enabled and available"""
        return self.tracker is not None and DEEPSORT_AVAILABLE
    
    def get_tracking_status(self) -> Dict[str, any]:
        """Get detailed tracking status"""
        return {
            "deep_sort_available": DEEPSORT_AVAILABLE,
            "tracking_enabled": self.enable_tracking,
            "tracker_initialized": self.tracker is not None,
            "status": "active" if (self.tracker is not None and DEEPSORT_AVAILABLE) else "disabled"
        }

    def detect(
        self,
        image_path: Optional[str] = None,
        camera_id: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Detect vehicles in an image or frame

        Args:
            image_path: Path to image file
            camera_id: Camera identifier (if using camera stream)
            frame: Pre-loaded frame (numpy array)

        Returns:
            List of detection dictionaries with keys:
            - vehicle_type: str (car, truck, bus, motorcycle, bicycle)
            - confidence: float (0.0 to 1.0)
            - bbox: List[int] [x1, y1, x2, y2] (bounding box coordinates)
            - center: List[int] [cx, cy] (center point of bounding box)
            - width: int
            - height: int
            - track_id: Optional[int] (stable ID across frames if tracking enabled)
        """
        if not self.is_ready():
            print("Warning: Model not loaded. Cannot perform detection.")
            return []

        # Load image
        if frame is not None:
            img = frame.copy()
        elif image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image from {image_path}")
                return []
        elif camera_id:
            # Get frame from camera (integrate with camera_streamer if needed)
            print(f"Warning: Camera ID {camera_id} not yet implemented")
            return []
        else:
            return []

        if img is None or img.size == 0:
            return []

        # Perform detection (+ optional tracking)
        detections = self._detect_vehicles(img)

        return detections

    def _detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Internal method to detect vehicles in a frame using YOLOv8
        If DeepSORT tracker is available, returns tracked detections with track_id
        """
        detections: List[Dict] = []

        try:
            # Run YOLO inference
            results = self.model(
                frame, conf=self.confidence_threshold, verbose=False
            )

            # ---- If tracking is not enabled or tracker failed, fall back to old behaviour ----
            if self.tracker is None:
                for result in results:
                    boxes = result.boxes

                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])

                        if class_id not in self.VEHICLE_CLASSES:
                            continue

                        vehicle_type = self.VEHICLE_CLASSES[class_id]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = [int(x1), int(y1), int(x2), int(y2)]

                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        detection = {
                            "vehicle_type": vehicle_type,
                            "confidence": round(confidence, 3),
                            "bbox": bbox,
                            "center": [center_x, center_y],
                            "width": int(x2 - x1),
                            "height": int(y2 - y1),
                            "track_id": None,
                        }
                        detections.append(detection)

                return detections

            # ---- DeepSORT enabled: build detections list for tracker ----
            raw_detections = []  # ([x, y, w, h], conf, dummy_class)
            others = []          # supplementary info carrying type + conf

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id not in self.VEHICLE_CLASSES:
                        continue

                    vehicle_type = self.VEHICLE_CLASSES[class_id]

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = float(x2 - x1)
                    h = float(y2 - y1)

                    raw_detections.append(([float(x1), float(y1), w, h], confidence, 0))
                    # store full info as supplementary data
                    others.append(
                        {
                            "vehicle_type": vehicle_type,
                            "confidence": float(confidence),
                        }
                    )

            # Update DeepSORT tracker
            tracks = self.tracker.update_tracks(
                raw_detections,
                frame=frame,
                others=others,
            )

            # Convert tracks back to our detection dicts
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = int(track.track_id)
                l, t, r, b = track.to_ltrb()
                x1, y1, x2, y2 = map(int, [l, t, r, b])

                info = track.get_det_supplementary() or {}
                vehicle_type = info.get("vehicle_type", "car")
                confidence = float(info.get("confidence", 1.0))

                width = int(x2 - x1)
                height = int(y2 - y1)
                center_x = x1 + width // 2
                center_y = y1 + height // 2

                detection = {
                    "vehicle_type": vehicle_type,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "center": [center_x, center_y],
                    "width": width,
                    "height": height,
                    "track_id": track_id,
                }
                detections.append(detection)

        except Exception as e:
            print(f"Error during detection/tracking: {str(e)}")
            return []

        return detections

    def count_vehicles(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count vehicles by type from detections (per-frame)

        NOTE:
        - For videos with DeepSORT, you should use track_id-based
          counting at a higher level (e.g. VideoProcessor) to avoid
          counting the same vehicle multiple times.
        """
        counts = {vtype: 0 for vtype in self.classes}
        counts["total"] = len(detections)

        for detection in detections:
            vtype = detection.get("vehicle_type", "unknown")
            if vtype in counts:
                counts[vtype] += 1

        return counts

    def detect_with_visualization(
        self, frame: np.ndarray
    ) -> tuple[List[Dict], np.ndarray]:
        """
        Detect vehicles and return annotated frame with bounding boxes
        (uses tracking if available)
        """
        detections = self._detect_vehicles(frame)
        annotated_frame = frame.copy()

        # Draw bounding boxes and labels
        for detection in detections:
            bbox = detection["bbox"]
            vehicle_type = detection["vehicle_type"]
            confidence = detection["confidence"]
            track_id = detection.get("track_id")

            x1, y1, x2, y2 = bbox

            # Choose color based on vehicle type
            colors = {
                "car": (0, 255, 0),  # Green
                "truck": (255, 0, 0),  # Blue
                "bus": (0, 0, 255),  # Red
                "motorcycle": (255, 255, 0),  # Cyan
                "bicycle": (255, 0, 255),  # Magenta
            }
            color = colors.get(vehicle_type, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Label: type + conf + optional ID
            if track_id is not None:
                label = f"ID {track_id} | {vehicle_type} {confidence:.2f}"
            else:
                label = f"{vehicle_type} {confidence:.2f}"

            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            label_y = max(y1, label_size[1] + 10)

            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return detections, annotated_frame
