# Vehicle Detector Usage Guide

## Overview
The `VehicleDetector` class uses YOLOv8 (You Only Look Once version 8) to detect vehicles in images and video frames. It can detect:
- Cars
- Trucks
- Buses
- Motorcycles
- Bicycles

## Installation

First, install the required dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- `ultralytics` - YOLOv8 library
- `torch` - PyTorch (required by YOLO)
- `opencv-python` - Image processing
- Other dependencies

**Note**: On first run, YOLOv8 will automatically download the pre-trained model (`yolov8n.pt`) - this is about 6MB.

## Basic Usage

### 1. Initialize the Detector

```python
from vehicle_detector import VehicleDetector

# Initialize with default settings (uses YOLOv8n, confidence threshold 0.5)
detector = VehicleDetector()

# Or customize confidence threshold
detector = VehicleDetector(confidence_threshold=0.6)

# Or use a custom model
detector = VehicleDetector(model_path="path/to/custom_model.pt")
```

### 2. Detect Vehicles in an Image

```python
# Detect from image file
detections = detector.detect(image_path="path/to/image.jpg")

# Or detect from a frame (numpy array)
import cv2
frame = cv2.imread("path/to/image.jpg")
detections = detector.detect(frame=frame)

# Print results
for detection in detections:
    print(f"Vehicle: {detection['vehicle_type']}")
    print(f"Confidence: {detection['confidence']}")
    print(f"Bounding Box: {detection['bbox']}")
    print(f"Center: {detection['center']}")
    print("---")
```

### 3. Count Vehicles

```python
detections = detector.detect(image_path="image.jpg")
counts = detector.count_vehicles(detections)

print(f"Total vehicles: {counts['total']}")
print(f"Cars: {counts['car']}")
print(f"Trucks: {counts['truck']}")
print(f"Buses: {counts['bus']}")
print(f"Motorcycles: {counts['motorcycle']}")
print(f"Bicycles: {counts['bicycle']}")
```

### 4. Visualize Detections

```python
import cv2

frame = cv2.imread("path/to/image.jpg")
detections, annotated_frame = detector.detect_with_visualization(frame)

# Save annotated image
cv2.imwrite("output_with_detections.jpg", annotated_frame)

# Or display
cv2.imshow("Detections", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Detection Output Format

Each detection is a dictionary with:
```python
{
    "vehicle_type": "car",           # str: car, truck, bus, motorcycle, or bicycle
    "confidence": 0.85,              # float: confidence score (0.0 to 1.0)
    "bbox": [100, 50, 300, 200],     # List[int]: [x1, y1, x2, y2] bounding box
    "center": [200, 125],            # List[int]: [cx, cy] center point
    "width": 200,                    # int: bounding box width
    "height": 150                    # int: bounding box height
}
```

## Model Information

- **Default Model**: YOLOv8n (nano) - fastest and smallest
- **Model Size**: ~6MB
- **Speed**: ~10-30 FPS on CPU, ~100+ FPS on GPU
- **Accuracy**: Good for general vehicle detection
- **Classes Detected**: 80 COCO classes, filtered to vehicles only

### Alternative Models

You can use different YOLOv8 models for better accuracy (but slower):
- `yolov8n.pt` - Nano (fastest, default)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate, slowest)

To use a different model:
```python
detector = VehicleDetector(model_path="yolov8m.pt")
```

## Integration with API

The detector is already integrated with the FastAPI router at `Backend/routers/vehicle_detector.py`:

```python
# API endpoint: POST /api/vehicles/detect
{
    "image_path": "/path/to/image.jpg",
    "camera_id": "camera1"
}

# Response:
{
    "detections": [...],
    "count": 5
}
```

## Performance Tips

1. **Confidence Threshold**: Lower threshold (0.3-0.4) detects more vehicles but may include false positives. Higher threshold (0.6-0.7) is more accurate but may miss some vehicles.

2. **Frame Processing**: For video, process every Nth frame to improve performance:
   ```python
   frame_count = 0
   for frame in video_frames:
       if frame_count % 5 == 0:  # Process every 5th frame
           detections = detector.detect(frame=frame)
       frame_count += 1
   ```

3. **GPU Acceleration**: If you have CUDA GPU, PyTorch will automatically use it for faster inference.

## Troubleshooting

### Model Not Loading
- Ensure `ultralytics` is installed: `pip install ultralytics`
- Check internet connection (first run downloads model)
- Verify PyTorch is installed: `pip install torch torchvision`

### Low Detection Accuracy
- Try lowering confidence threshold
- Use a larger model (yolov8m or yolov8l)
- Ensure good image quality and lighting

### Slow Performance
- Use YOLOv8n (nano) model
- Process fewer frames per second
- Use GPU if available
- Reduce image resolution before detection

## Example: Complete Detection Script

```python
from vehicle_detector import VehicleDetector
import cv2

# Initialize detector
detector = VehicleDetector(confidence_threshold=0.5)

# Check if ready
if not detector.is_ready():
    print("Detector not ready!")
    exit(1)

# Load image
image_path = "traffic_image.jpg"
detections = detector.detect(image_path=image_path)

# Count vehicles
counts = detector.count_vehicles(detections)
print(f"Found {counts['total']} vehicles:")
for vehicle_type, count in counts.items():
    if vehicle_type != 'total' and count > 0:
        print(f"  {vehicle_type}: {count}")

# Visualize
frame = cv2.imread(image_path)
detections, annotated = detector.detect_with_visualization(frame)
cv2.imwrite("detected_output.jpg", annotated)
print("Saved annotated image to detected_output.jpg")
```

