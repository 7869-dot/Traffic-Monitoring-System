# Vehicle Detection System - Architectural Roadmap

## Overview
This document outlines the complete architecture and file structure needed to build a vehicle detection system that analyzes video footage and counts vehicles.

---

## 🏗️ System Architecture

```
Traffic Monitoring System
├── Backend/                    # FastAPI backend (✅ Already exists)
│   ├── app.py                  # Main FastAPI application (✅ Exists)
│   ├── database.py             # Database management (✅ Exists)
│   ├── vehicle_detector.py     # Core detection logic (⚠️ Needs implementation)
│   ├── camera_streamer.py      # Video stream handling (⚠️ Needs implementation)
│   ├── video_processor.py      # Video file processing (❌ Needs creation)
│   ├── models/                 # ML models directory (❌ Needs creation)
│   │   └── download_models.py  # Script to download YOLO models
│   ├── routers/
│   │   ├── vehicle_detector.py # API endpoints (✅ Exists, needs updates)
│   │   ├── camera_stream.py    # Camera endpoints (✅ Exists)
│   │   └── video_upload.py     # Video upload endpoints (❌ Needs creation)
│   └── requirements.txt        # Dependencies (✅ Exists, needs updates)
│
├── Frontend/                   # Web interface (❌ Needs creation)
│   ├── index.html              # Main HTML page
│   ├── styles.css              # Styling
│   ├── app.js                  # Frontend JavaScript
│   └── upload.html             # Video upload interface
│
└── Ardriuno/                   # Arduino integration (✅ Exists)
    └── traffic-light.ino
```

---

## 📋 Files to Implement/Create

### 🔴 **CRITICAL - Core Detection System**

#### 1. **Backend/vehicle_detector.py** (⚠️ Currently placeholder)
**Purpose**: Core vehicle detection logic using computer vision
**What to implement**:
- YOLO model integration (YOLOv8 recommended)
- Frame preprocessing
- Vehicle detection and classification
- Confidence thresholding
- Bounding box extraction
- Vehicle counting logic

**Key Methods Needed**:
- `_load_model()` - Load YOLO weights/config
- `_detect_vehicles()` - Run detection on frames
- `detect()` - Main detection interface
- `count_vehicles()` - Count vehicles by type
- `track_vehicles()` - Optional: vehicle tracking across frames

---

#### 2. **Backend/video_processor.py** (❌ New file)
**Purpose**: Process video files frame-by-frame
**What to implement**:
- Video file reading (MP4, AVI, MOV support)
- Frame extraction at specified FPS
- Progress tracking
- Batch processing
- Integration with VehicleDetector

**Key Methods Needed**:
- `process_video()` - Main processing function
- `extract_frames()` - Extract frames from video
- `get_video_info()` - Get video metadata (duration, FPS, resolution)
- `save_results()` - Save detection results

---

#### 3. **Backend/models/download_models.py** (❌ New file)
**Purpose**: Download and setup YOLO models
**What to implement**:
- Download YOLOv8 weights
- Verify model files
- Setup model directory structure

---

#### 4. **Backend/routers/video_upload.py** (❌ New file)
**Purpose**: Handle video file uploads and processing
**What to implement**:
- File upload endpoint (POST /api/video/upload)
- Video processing endpoint (POST /api/video/process)
- Progress tracking endpoint (GET /api/video/progress/{job_id})
- Results retrieval endpoint (GET /api/video/results/{job_id})

**Endpoints Needed**:
- `POST /api/video/upload` - Upload video file
- `POST /api/video/process` - Start processing uploaded video
- `GET /api/video/progress/{job_id}` - Get processing progress
- `GET /api/video/results/{job_id}` - Get detection results
- `GET /api/video/list` - List processed videos

---

### 🟡 **IMPORTANT - Supporting Infrastructure**

#### 5. **Backend/camera_streamer.py** (⚠️ Check if exists)
**Purpose**: Handle live camera streams
**What to implement**:
- Camera initialization
- Frame capture
- Stream management
- Multiple camera support

---

#### 6. **Backend/config.py** (❌ New file)
**Purpose**: Centralized configuration
**What to implement**:
- Model paths
- Detection thresholds
- Video processing settings
- Camera settings
- Database paths

---

#### 7. **Backend/utils.py** (❌ New file)
**Purpose**: Utility functions
**What to implement**:
- Image preprocessing helpers
- File validation
- Progress calculation
- Result formatting

---

### 🟢 **OPTIONAL - Enhanced Features**

#### 8. **Backend/vehicle_tracker.py** (❌ New file)
**Purpose**: Track vehicles across frames (avoid double counting)
**What to implement**:
- Object tracking (DeepSORT or ByteTrack)
- Vehicle ID assignment
- Trajectory tracking
- Counting logic (count when vehicle enters/exits region)

---

#### 9. **Backend/routers/analytics.py** (❌ New file)
**Purpose**: Analytics and reporting endpoints
**What to implement**:
- Time-based statistics
- Vehicle type distribution
- Peak hour analysis
- Export to CSV/JSON

---

### 🎨 **Frontend Files** (❌ Needs creation)

#### 10. **Frontend/index.html**
**Purpose**: Main dashboard
**Features**:
- Real-time vehicle count display
- Video player with detection overlay
- Statistics charts
- Camera selection

#### 11. **Frontend/upload.html**
**Purpose**: Video upload interface
**Features**:
- Drag-and-drop file upload
- Progress bar
- Results display
- Download results

#### 12. **Frontend/app.js**
**Purpose**: Frontend JavaScript logic
**Features**:
- API calls to backend
- Real-time updates
- Chart rendering
- Video playback control

#### 13. **Frontend/styles.css**
**Purpose**: Styling

---

## 🔧 Implementation Priority

### **Phase 1: Core Detection** (Must Have)
1. ✅ Update `Backend/vehicle_detector.py` with YOLO implementation
2. ✅ Create `Backend/video_processor.py` for video file processing
3. ✅ Create `Backend/routers/video_upload.py` for video upload endpoints
4. ✅ Update `Backend/requirements.txt` with YOLO dependencies

### **Phase 2: API Integration** (Must Have)
5. ✅ Update `Backend/routers/vehicle_detector.py` to work with video files
6. ✅ Create `Backend/config.py` for configuration
7. ✅ Test video upload → processing → results flow

### **Phase 3: Frontend** (Should Have)
8. ✅ Create basic frontend for video upload
9. ✅ Create dashboard for viewing results
10. ✅ Add real-time visualization

### **Phase 4: Enhancements** (Nice to Have)
11. ✅ Add vehicle tracking to avoid double counting
12. ✅ Add analytics and reporting
13. ✅ Add export functionality

---

## 📦 Required Dependencies

### Add to `Backend/requirements.txt`:
```
ultralytics>=8.0.0          # YOLOv8
torch>=2.0.0                # PyTorch (for YOLO)
torchvision>=0.15.0
pillow>=10.0.0
python-multipart>=0.0.12    # For file uploads
aiofiles>=23.0.0             # Async file operations
```

---

## 🎯 Key Implementation Details

### Vehicle Detection Approach:
1. **Model**: Use YOLOv8 (pre-trained on COCO dataset)
   - Classes: car, truck, bus, motorcycle, bicycle
   - Confidence threshold: 0.5 (configurable)

2. **Video Processing Flow**:
   ```
   Upload Video → Extract Frames (every N frames) → 
   Run Detection → Aggregate Results → Store in DB → Return Counts
   ```

3. **Counting Strategy**:
   - **Simple**: Count all detections in each frame, average across video
   - **Advanced**: Use tracking to count unique vehicles (enter/exit detection)

4. **Database Storage**:
   - Store detection results per frame
   - Store aggregated counts per video
   - Store metadata (video path, processing time, etc.)

---

## 📝 Next Steps

1. **Start with**: Update `vehicle_detector.py` with YOLO implementation
2. **Then**: Create `video_processor.py` for video file handling
3. **Then**: Create `video_upload.py` router for API endpoints
4. **Finally**: Test with a sample video file

---

## 🔍 Testing Checklist

- [ ] Upload a video file via API
- [ ] Process video and detect vehicles
- [ ] Get accurate vehicle counts
- [ ] Store results in database
- [ ] Retrieve results via API
- [ ] Handle multiple video formats
- [ ] Handle large video files
- [ ] Show progress during processing

---

## 📚 Resources

- YOLOv8 Documentation: https://docs.ultralytics.com/
- FastAPI File Uploads: https://fastapi.tiangolo.com/tutorial/request-files/
- OpenCV Video Processing: https://docs.opencv.org/

