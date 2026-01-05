"""
Vehicle Detector Module
Handles vehicle detection using computer vision with YOLOv8 + DeepSORT tracking
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
import os
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("Warning: deep-sort-realtime not installed. Install with: pip install deep-sort-realtime")

logger = logging.getLogger(__name__)


class VehicleDetector:
    """Detects vehicles in images or video streams using YOLOv8"""
    
    # COCO dataset class IDs for vehicles
    # COCO classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
    VEHICLE_CLASSES = {
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize vehicle detector with YOLO and DeepSORT
        
        Args:
            model_path: Path to YOLO model weights (optional, uses YOLOv8n by default)
            confidence_threshold: Minimum confidence score for detections (default: 0.5)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tracker = None
        self.classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        self.is_model_loaded = False
        self.is_tracker_loaded = False
        
        if not YOLO_AVAILABLE:
            print("Error: ultralytics package not available. Please install it first.")
            return
        
        # Load model
        self._load_model()
        
        # Initialize DeepSORT tracker
        self._load_tracker()
    
    def _load_model(self):
        """Load YOLOv8 detection model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model if path provided
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded custom model from {self.model_path}")
            else:
                # Load pre-trained YOLOv8n (nano) model - fastest and smallest
                # Will auto-download on first use
                self.model = YOLO('yolov8n.pt')
                logger.info("Loaded YOLOv8n pre-trained model")
            
            self.is_model_loaded = True
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_model_loaded = False
            self.model = None
    
    def _load_tracker(self):
        """Initialize DeepSORT tracker"""
        if not DEEPSORT_AVAILABLE:
            logger.warning("DeepSORT not available. Tracking will be disabled.")
            self.is_tracker_loaded = False
            return
        
        try:
            # Initialize DeepSORT tracker
            # max_age: number of frames to keep a track alive without detection
            # n_init: number of consecutive detections before track is confirmed
            self.tracker = DeepSort(
                max_age=50,
                n_init=3,
                max_iou_distance=0.7,
                max_cosine_distance=0.2,
                nn_budget=100
            )
            self.is_tracker_loaded = True
            logger.info("DeepSORT tracker initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing tracker: {str(e)}")
            self.is_tracker_loaded = False
            self.tracker = None
    
    def is_ready(self) -> bool:
        """Check if detector is ready"""
        return self.is_model_loaded and self.model is not None
    
    def detect(self, image_path: Optional[str] = None, camera_id: Optional[str] = None, frame: Optional[np.ndarray] = None, use_tracking: bool = False) -> List[Dict]:
        """
        Detect vehicles in an image or frame
        
        Args:
            image_path: Path to image file
            camera_id: Camera identifier (if using camera stream)
            frame: Pre-loaded frame (numpy array)
            use_tracking: Whether to use DeepSORT tracking (default: False for backward compatibility)
        
        Returns:
            List of detection dictionaries with keys:
            - id: int (track ID if tracking enabled, otherwise index)
            - label: str (car, truck, bus, motorcycle, bicycle)
            - confidence: float (0.0 to 1.0)
            - bbox: List[int] [x1, y1, x2, y2] (bounding box coordinates)
        """
        if not self.is_ready():
            logger.warning("Model not loaded. Cannot perform detection.")
            return []
        
        # Load image
        if frame is not None:
            img = frame.copy()
        elif image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image from {image_path}")
                return []
        elif camera_id:
            # Get frame from camera (you'll need to integrate with camera_streamer)
            logger.warning(f"Camera ID {camera_id} not yet implemented")
            return []
        else:
            return []
        
        if img is None or img.size == 0:
            return []
        
        # Perform detection
        detections = self._detect_vehicles(img, use_tracking=use_tracking)
        
        return detections
    
    def _detect_vehicles(self, frame: np.ndarray, use_tracking: bool = True) -> List[Dict]:
        """
        Internal method to detect vehicles in a frame using YOLOv8 + DeepSORT
        
        Args:
            frame: Input frame (numpy array in BGR format)
            use_tracking: Whether to use DeepSORT tracking (default: True)
        
        Returns:
            List of detection dictionaries with format:
            - id: int (track ID from DeepSORT)
            - label: str (vehicle type: car, truck, bus, motorcycle, bicycle)
            - confidence: float (0.0 to 1.0)
            - bbox: List[int] [x1, y1, x2, y2] (bounding box coordinates)
        """
        detections = []
        
        try:
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Prepare detections for tracking (format: [[x1, y1, x2, y2, confidence, class_id], ...])
            detections_for_tracker = []
            detection_classes = []
            detection_confidences = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if detected object is a vehicle
                    if class_id in self.VEHICLE_CLASSES:
                        # Get bounding box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Format for DeepSORT: [x1, y1, x2, y2, confidence, class_id]
                        detections_for_tracker.append([int(x1), int(y1), int(x2), int(y2), confidence, class_id])
                        detection_classes.append(class_id)
                        detection_confidences.append(confidence)
            
            # Apply tracking if available and enabled
            if use_tracking and self.is_tracker_loaded and self.tracker is not None and len(detections_for_tracker) > 0:
                # Convert to numpy array for tracker
                detections_array = np.array(detections_for_tracker)
                
                # Update tracker with detections
                tracks = self.tracker.update_tracks(detections_array, frame=frame)
                
                # Process tracked detections
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    track_id = track.track_id
                    ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
                    
                    # Find corresponding class_id and confidence from original detections
                    # Match by bounding box overlap
                    best_match_idx = 0
                    best_iou = 0
                    
                    for idx, det in enumerate(detections_for_tracker):
                        det_bbox = det[:4]
                        iou = self._calculate_iou(ltrb, det_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = idx
                    
                    class_id = detection_classes[best_match_idx]
                    confidence = detection_confidences[best_match_idx]
                    vehicle_type = self.VEHICLE_CLASSES[class_id]
                    
                    detection = {
                        "id": int(track_id),
                        "label": vehicle_type,
                        "confidence": round(confidence, 3),
                        "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
                    }
                    detections.append(detection)
            else:
                # Fallback to detection without tracking
                for idx, det in enumerate(detections_for_tracker):
                    x1, y1, x2, y2, confidence, class_id = det
                    vehicle_type = self.VEHICLE_CLASSES[class_id]
                    
                    detection = {
                        "id": idx,  # Use index as temporary ID when tracking unavailable
                        "label": vehicle_type,
                        "confidence": round(confidence, 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    }
                    detections.append(detection)
        
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []
        
        return detections
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        
        Returns:
            IoU value (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def count_vehicles(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count vehicles by type from detections
        
        Args:
            detections: List of detection dictionaries (with 'label' or 'vehicle_type' key)
        
        Returns:
            Dictionary with vehicle counts by type
        """
        counts = {vtype: 0 for vtype in self.classes}
        counts['total'] = len(detections)
        
        for detection in detections:
            # Support both 'label' (new format) and 'vehicle_type' (legacy format)
            vtype = detection.get("label") or detection.get("vehicle_type", "unknown")
            if vtype in counts:
                counts[vtype] += 1
        
        return counts
    
    def detect_with_visualization(self, frame: np.ndarray, use_tracking: bool = False) -> tuple[List[Dict], np.ndarray]:
        """
        Detect vehicles and return annotated frame with bounding boxes
        
        Args:
            frame: Input frame (numpy array)
            use_tracking: Whether to use DeepSORT tracking (default: False)
        
        Returns:
            Tuple of (detections list, annotated frame)
        """
        detections = self._detect_vehicles(frame, use_tracking=use_tracking)
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and labels
        for detection in detections:
            bbox = detection['bbox']
            # Support both 'label' (new format) and 'vehicle_type' (legacy format)
            vehicle_type = detection.get('label') or detection.get('vehicle_type', 'unknown')
            confidence = detection['confidence']
            track_id = detection.get('id', 0)
            
            x1, y1, x2, y2 = bbox
            
            # Choose color based on vehicle type
            colors = {
                'car': (0, 255, 0),      # Green
                'truck': (255, 0, 0),    # Blue
                'bus': (0, 0, 255),      # Red
                'motorcycle': (255, 255, 0),  # Cyan
                'bicycle': (255, 0, 255)  # Magenta
            }
            color = colors.get(vehicle_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with track ID if tracking is enabled
            if use_tracking and track_id > 0:
                label = f"ID:{track_id} {vehicle_type} {confidence:.2f}"
            else:
                label = f"{vehicle_type} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y1, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return detections, annotated_frame

