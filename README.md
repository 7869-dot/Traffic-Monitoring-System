# 🚦 Traffic Monitoring System

A traffic monitoring system with a **FastAPI** backend (vehicle detection with
**YOLOv8** + a lightweight centroid tracker) and a simple, **deployable static
frontend**. Upload an MP4 of traffic footage and get vehicle counts back.

It was tested with footage from an **ESP32‑CAM** (saved as MP4) but works with
any MP4 traffic video.

---

## ✨ Features

- ✅ Upload an MP4/AVI/MOV/MKV video and analyze it
- ✅ Vehicle detection with **YOLOv8** (cars, trucks, buses, motorcycles, bicycles)
- ✅ Honest counting — does **not** naively sum per‑frame detections:
  - **Unique estimate** — distinct vehicles across the clip (centroid tracker)
  - **Peak simultaneous** — most vehicles seen at once in a single frame
  - **Average per frame**
- ✅ Simple static frontend (plain HTML/CSS/JS — no build step)
- ✅ Optional Arduino traffic‑light control and live camera endpoints (require hardware)

---

## 🗂 Project structure

```
Backend/            FastAPI app + detection / tracking / processing
  app.py            API entrypoint (run with uvicorn)
  Video_processor.py Frame sampling + counting logic
  vehicle_detector.py YOLOv8 wrapper
  tracker.py        Centroid tracker for unique-vehicle estimation
  routers/          API routes (video, vehicles, camera, arduino)
frontend/           Static web UI (index.html, style.css, app.js)
Ardriuno/           Arduino sketch for the traffic light
```

---

## 🚀 Run the backend

```bash
cd Backend
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m uvicorn app:app --reload
```

The API runs at `http://127.0.0.1:8000` (interactive docs at `/docs`).
On first run YOLOv8 downloads the `yolov8n.pt` weights (~6 MB) if not present.

### Configuring CORS for a deployed frontend

The backend reads allowed origins from the `ALLOWED_ORIGINS` environment
variable (comma‑separated). Defaults cover common local dev ports. For example:

```bash
ALLOWED_ORIGINS="https://your-frontend.example.com" python -m uvicorn app:app
```

---

## 🖥 Run the frontend

The frontend is fully static — serve the `frontend/` folder any way you like:

```bash
cd frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500`, set the **API base URL** to your backend
(default `http://127.0.0.1:8000`), choose a video and click **Analyze**.

---

## ☁️ Deploying (Vercel frontend + Render backend)

Deploy the **backend first** so you have its URL for the frontend config.

### 1) Backend on Render

The repo includes [`render.yaml`](render.yaml). Either create a **Blueprint**
(New + → Blueprint → pick this repo) or a **Web Service** with these settings:

| Setting | Value |
| ------- | ----- |
| Root Directory | `Backend` |
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health Check Path | `/health` |

**Environment variables (Render):**

| Key | Value | Notes |
| --- | ----- | ----- |
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app` | Your Vercel URL. Comma‑separate multiple (e.g. add preview domains). **No trailing slash.** |
| `PYTHON_VERSION` | `3.12.7` | Avoids PyTorch wheel issues on newer Python. |

Do **not** set `PORT` — Render provides it automatically.

> ⚠️ **Notes that matter for this backend:**
> - Use a **single worker** (`--workers 1`). The progress/job store is
>   in-memory, so a job created on one worker wouldn't be found by another.
> - PyTorch + YOLO is memory-heavy. Render's **free tier (512 MB) may OOM** while
>   loading the model — if the service keeps restarting/crashing, upgrade to a
>   plan with ≥ 2 GB RAM.
> - `requirements.txt` already uses **CPU-only PyTorch** and **headless OpenCV**,
>   which are required on a server (the GUI OpenCV build crashes on import).
> - The YOLO weights (`Backend/yolov8n.pt`) are committed, so no model download
>   is needed at runtime.

### 2) Frontend on Vercel

1. Edit [`frontend/config.js`](frontend/config.js) and set your Render URL:
   ```js
   window.TMS_API_BASE = "https://traffic-monitoring-api.onrender.com";
   ```
   Commit and push.
2. In Vercel: **New Project** → import this repo → set **Root Directory** to
   `frontend`. Framework preset = **Other** (no build step). Deploy.

**Environment variables (Vercel):** none. The frontend is static with no build
step, so Vercel env vars can't be injected into the JS — the backend URL lives
in `config.js` instead (and can also be changed at runtime in the UI).

### 3) Connect them

Make sure Render's `ALLOWED_ORIGINS` exactly matches your final Vercel domain.
Open the Vercel URL — the **Backend connection** badge should read *connected*.

### Database?

No separate database is required. The app auto-creates a small SQLite file on
startup, but the **video-analysis flow the frontend uses does not depend on it**
(it's only used by the optional Arduino/manual-detect endpoints). On Render the
filesystem is ephemeral, which is fine here. Add a Render Disk or external DB
only if you later need to persist detection/Arduino logs.

---

## 🔌 Key API endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/health` | Health check |
| `POST` | `/api/video/upload-and-process` | Upload a short video and get counts synchronously |
| `POST` | `/api/video/upload-async` | Upload a video; returns a `job_id` and processes in the background |
| `GET`  | `/api/video/job/{job_id}` | Poll a background job for progress and the final result |
| `GET`  | `/api/vehicles/stats` | Detection statistics |
| `GET`  | `/api/camera/cameras` | List local cameras (needs hardware) |
| `GET`  | `/api/arduino/status` | Arduino connection status (needs hardware) |

> The frontend uses the **async** flow: it uploads the video, then polls the job
> for a live progress bar so long videos don't block a single request. The
> in-memory job store works for a single worker; for multi-worker deployments
> swap it for Redis or a database.

**Example response** from `/api/video/upload-and-process`:

```json
{
  "filename": "traffic.mp4",
  "total_frames": 718,
  "processed_frames": 30,
  "frame_sample_rate": 10,
  "duration_sec": 23.96,
  "fps": 29.97,
  "peak_counts":   { "car": 8,  "truck": 0, "total": 8 },
  "unique_estimate": { "car": 22, "truck": 0, "total": 22 },
  "avg_per_frame": { "car": 3.07, "total": 3.07 }
}
```

---

## 🧠 How counting works

Detecting on every frame and summing counts over‑counts massively (a parked car
gets counted once per frame). Instead the processor samples every Nth frame,
runs YOLOv8 on each sample, and:

- tracks detections across frames with a centroid tracker to estimate the number
  of **distinct** vehicles, and
- records the **peak** number seen simultaneously.

The unique estimate is a heuristic, not a guarantee — tune `frame_sample_rate`
for your footage.

---

## 📜 License

Add your license here (MIT / Apache 2.0 / GPL / etc.)
