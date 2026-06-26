# 🎯 Real-Time Object Detection & Tracking — CodeAlpha AI Internship (Task 4)

A real-time computer vision pipeline that **detects** objects in a video stream
using **YOLOv8** and **tracks** them across frames with **Deep SORT**, assigning
each object a persistent ID even through brief occlusion. Built for the
CodeAlpha AI Internship, with several optional features added on top of the
core brief to make the project demo- and resume-ready.

## ✅ How this satisfies the task brief

| Requirement (from CodeAlpha brief) | Implementation |
|---|---|
| Real-time video input (webcam or file) | `cv2.VideoCapture` in `main.py`, source selectable via `--source` |
| Pre-trained detection model (YOLO / Faster R-CNN) | YOLOv8 (`ultralytics`) in `src/detector.py` |
| Process each frame, draw bounding boxes | `src/utils.py: draw_tracked_object()` |
| Object tracking (SORT / Deep SORT) | Deep SORT (`deep-sort-realtime`) in `src/tracker.py` |
| Display output with labels + tracking IDs in real time | Live `cv2.imshow` window with ID, class, and confidence on every box |

## ⭐ Extra features (beyond the brief, for a stronger demo)

- **Live object-count panel** — running unique count per class, overlaid on the video
- **IN/OUT line-crossing counter** (`--line-y`) — footfall/vehicle-counting style metric
- **CSV analytics logging** — every detection logged with timestamp + summary report
- **Output video export** — annotated video saved automatically
- **FPS overlay** — shows real-time processing speed
- **Streamlit web dashboard** (`app.py`) — drag-and-drop video upload, live preview, downloadable CSV — no terminal needed
- **Configurable via CLI flags or `config.py`** — confidence threshold, class filter, model size
- **Unit tests** (`tests/`) — basic correctness checks with `pytest`
- **Class-filtering** — e.g. only track people and vehicles with `--classes 0,2`

## 📁 Folder Structure

```
CodeAlpha_ObjectDetectionTracking/
├── README.md                  # you are here
├── requirements.txt           # pip dependencies
├── .gitignore
├── config.py                  # all tunable settings in one place
├── main.py                    # CLI app — run this for webcam/video file
├── app.py                     # optional Streamlit web dashboard
├── src/
│   ├── __init__.py
│   ├── detector.py            # YOLOv8 detection wrapper
│   ├── tracker.py             # Deep SORT tracking wrapper
│   ├── counter.py             # optional IN/OUT line-crossing counter
│   ├── analytics.py           # CSV logging + unique-object-count summary
│   └── utils.py                # FPS counter, drawing helpers
├── tests/
│   └── test_detector.py       # pytest sanity checks
├── data/                      # put your sample/demo videos here
├── models/                    # YOLO weights are auto-downloaded here
└── outputs/                   # annotated videos + CSV logs land here
```

## 🚀 Setup

```bash
# 1. Clone your repo (after you push this to GitHub, per the internship instructions)
git clone https://github.com/<your-username>/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> First run will auto-download the YOLOv8 weights (`yolov8n.pt`, ~6 MB) — no manual download needed.

## ▶️ Usage

### Option A — CLI (webcam or video file)

```bash
# Webcam, default settings
python main.py

# Run on a video file
python main.py --source data/sample.mp4

# Only detect people (class 0) and cars (class 2), higher confidence
python main.py --source data/sample.mp4 --classes 0,2 --conf 0.5

# Enable IN/OUT counting across a horizontal line at y=300
python main.py --source data/sample.mp4 --line-y 300

# Preview only, don't save an output video
python main.py --no-save
```

Press **`q`** in the preview window to stop. On exit you'll see:
```
[INFO] Frame-by-frame log saved to: outputs/detection_log_20260626_140230.csv
[INFO] Summary saved to:            outputs/summary_20260626_140230.csv
[INFO] Output video saved to:       outputs/output.mp4
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | `0` for webcam, or a path to a video file |
| `--model` | `yolov8n.pt` | YOLOv8 weights (`n`/`s`/`m`/`l`/`x` — bigger = more accurate, slower) |
| `--conf` | `0.4` | Detection confidence threshold |
| `--classes` | `None` (all) | Comma-separated COCO class ids to restrict detection to |
| `--output` | `outputs/output.mp4` | Where to save the annotated video |
| `--no-save` | off | Skip saving the output video |
| `--line-y` | `None` | Y pixel coordinate to enable the IN/OUT counter |
| `--no-show` | off | Run headless (no preview window) |

### Option B — Streamlit web dashboard

```bash
streamlit run app.py
```

Opens a browser tab where you can upload a video (or use your webcam),
watch detection + tracking happen live, and download the CSV log — great
for recording the **demo video** the internship instructions ask you to
post on LinkedIn.

### Running tests

```bash
pytest tests/
```

## 🧠 How it works (for your video explanation / README "Approach" section)

1. **Detection** — Each frame is passed to YOLOv8, which returns bounding
   boxes, confidence scores, and class IDs for everything it recognizes
   (80 COCO classes: people, cars, animals, etc.).
2. **Tracking** — Those boxes are handed to Deep SORT, which combines a
   **Kalman filter** (predicts where each object should be next frame)
   with a **CNN appearance embedding** (recognizes "this is the same
   person as 3 frames ago" even after a brief overlap with another
   object). This is what keeps IDs stable instead of flickering.
3. **Analytics** — Every confirmed track is logged frame-by-frame to CSV,
   and a running set of unique IDs per class gives a final "X people, Y
   cars detected" summary.
4. **Visualization** — Boxes, labels, IDs, FPS, and live counts are drawn
   directly onto the frame, then shown live and/or written to an output
   video file.

## 🛠️ Tech Stack

- **YOLOv8** (Ultralytics) — object detection
- **Deep SORT** (`deep-sort-realtime`) — multi-object tracking
- **OpenCV** — video I/O and rendering
- **Streamlit** — optional web dashboard
- **Pandas** — analytics table display
- **pytest** — testing

## 📋 Suggested README/Resume one-liner

> Built a real-time object detection & tracking system (YOLOv8 + Deep SORT)
> with live analytics, a line-crossing counter, and a Streamlit dashboard;
> achieved stable multi-object tracking with persistent IDs across
> occlusion.

## 📄 License

MIT — free to use and adapt for your own learning/portfolio.

---
Built for the **CodeAlpha Artificial Intelligence Internship** — Task 4: Object Detection and Tracking.
