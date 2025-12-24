"""
Vehicle Detector Module
Handles vehicle detection using computer vision with YOLOv8
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


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
        Initialize vehicle detector
        
        Args:
            model_path: Path to YOLO model weights (optional, uses YOLOv8n by default)
            confidence_threshold: Minimum confidence score for detections (default: 0.5)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        self.is_model_loaded = False
        
        if not YOLO_AVAILABLE:
            print("Error: ultralytics package not available. Please install it first.")
            return
        
        # Load model
        self._load_model()
    
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
                self.model = YOLO('yolov8n.pt')
                print("Loaded YOLOv8n pre-trained model")
            
            self.is_model_loaded = True
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_model_loaded = False
            self.model = None
    
    def is_ready(self) -> bool:
        """Check if detector is ready"""
        return self.is_model_loaded and self.model is not None
    
    def detect(self, image_path: Optional[str] = None, camera_id: Optional[str] = None, frame: Optional[np.ndarray] = None) -> List[Dict]:
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
            # Get frame from camera (you'll need to integrate with camera_streamer)
            print(f"Warning: Camera ID {camera_id} not yet implemented")
            return []
        else:
            return []
        
        if img is None or img.size == 0:
            return []
        
        # Perform detection
        detections = self._detect_vehicles(img)
        
        return detections
    
    def _detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Internal method to detect vehicles in a frame using YOLOv8
        
        Args:
            frame: Input frame (numpy array in BGR format)
        
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            # Run YOLO inference
            # Results will contain detected objects
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if detected object is a vehicle
                    if class_id in self.VEHICLE_CLASSES:
                        vehicle_type = self.VEHICLE_CLASSES[class_id]
                        
                        # Get bounding box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        
                        # Calculate center point
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Create detection dictionary
                        detection = {
                            "vehicle_type": vehicle_type,
                            "confidence": round(confidence, 3),
                            "bbox": bbox,
                            "center": [center_x, center_y],
                            "width": int(x2 - x1),
                            "height": int(y2 - y1)
                        }
                        
                        detections.append(detection)
        
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []
        
        return detections
    
    def count_vehicles(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count vehicles by type from detections
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Dictionary with vehicle counts by type
        """
        counts = {vtype: 0 for vtype in self.classes}
        counts['total'] = len(detections)
        
        for detection in detections:
            vtype = detection.get("vehicle_type", "unknown")
            if vtype in counts:
                counts[vtype] += 1
        
        return counts
    
    def detect_with_visualization(self, frame: np.ndarray) -> tuple[List[Dict], np.ndarray]:
        """
        Detect vehicles and return annotated frame with bounding boxes
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            Tuple of (detections list, annotated frame)
        """
        detections = self._detect_vehicles(frame)
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and labels
        for detection in detections:
            bbox = detection['bbox']
            vehicle_type = detection['vehicle_type']
            confidence = detection['confidence']
            
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
            
            # Draw label
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

