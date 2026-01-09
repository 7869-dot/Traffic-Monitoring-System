# 🚦 Traffic Monitoring System (FastAPI + YOLO + DeepSORT)

A simple and practical traffic monitoring system built with **FastAPI**, **YOLO** for vehicle detection, and **DeepSORT** for tracking.  
You can upload an **MP4 video** and the system will process it and return results in a **JSON output** (vehicle counts / tracking data depending on your implementation).

This project was tested using videos recorded from an **ESP32-CAM** (converted/saved as MP4), but it works with *any* MP4 traffic footage.

---

## ✨ Features

- ✅ Upload MP4 video through a FastAPI endpoint (Swagger UI)
- ✅ Vehicle detection using **YOLO**
- ✅ Vehicle tracking using **DeepSORT**
- ✅ Designed to avoid counting the same vehicle multiple times (tracking-based)
- ✅ Returns results in **JSON format**

---

## 📌 Repo Branches / Versions

This repo contains multiple versions:

### ✅ `main` branch (Recommended)
- **Most stable version**
- Uses **MP4 upload** workflow
- Best option if you want something that works reliably

### ⚠️ Other branches / versions (Under Development)
- More advanced experiments (including **POST requests / frame streaming** approaches)
- Currently **unstable** and still being developed
- Use only if you want to explore or contribute

👉 If you're a new user, **stick to the `main` branch**.

---

## 🧠 How It Works (High Level)

1. You upload an MP4 video to the FastAPI server.
2. The backend reads frames from the video.
3. **YOLO** detects vehicles in each frame.
4. **DeepSORT** assigns IDs and tracks vehicles across frames.
5. The system generates a final **JSON result** (counts / tracked objects / events).

---

## 🧰 Tech Stack

- **FastAPI** (Backend API)
- **YOLO** (Object detection)
- **DeepSORT** (Multi-object tracking)
- **OpenCV** (Video frame handling)
- Python 3.x

---

## 🚀 Getting Started

### 1) Clone the repo (stable version)
```bash
git clone <YOUR_REPO_URL_HERE>
cd <YOUR_PROJECT_FOLDER>
git checkout main
````

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell)**

```bash
.venv\Scripts\Activate.ps1
```

**Mac/Linux**

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the FastAPI server

```bash
uvicorn app:app --reload
```

If that doesn’t work, try:

```bash
python -m uvicorn app:app --reload
```

---

## ✅ Usage

1. Open your browser:

   * `http://127.0.0.1:8000/docs`

2. Find the endpoint for video upload (example: `/upload_video`)

3. Click **Try it out**

4. Upload an `.mp4` file

5. Execute → You’ll receive a **JSON response** with the processed results.

> Tip: If you recorded footage from ESP32-CAM, just make sure it’s saved/converted into `.mp4` before uploading.

---

## 📁 Input / Output

### Input

* `.mp4` video file (traffic footage)

### Output

* JSON result (example contents may include):

  * total vehicles counted
  * track IDs
  * per-class counts (car, bus, truck, etc.)
  * timestamps / frame numbers (if implemented)

---

## 🧪 Notes

* If you want the newest experimental features (like POST-based frame upload), check other branches — but they are **not stable** yet.
* This project is built to be simple and accurate first, then improved over time.

---

## 🤝 Contributing

Contributions are welcome:

* bug fixes
* performance improvements
* better counting logic
* UI improvements
* optimization for real-time streams

---

## 📜 License

Add your license here (MIT / Apache 2.0 / GPL / etc.)

---

## 📫 Contact

If you have questions or want to collaborate, open an issue in this repo.

