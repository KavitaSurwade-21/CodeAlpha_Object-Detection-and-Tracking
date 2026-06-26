"""
Central configuration for the Object Detection & Tracking project.
Edit values here instead of hunting through the codebase.
"""

# ---------------- Detection ----------------
DEFAULT_MODEL = "yolov8n.pt"        # n = nano (fastest) -> s -> m -> l -> x (most accurate)
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
DEFAULT_CLASSES = None              # None = detect all 80 COCO classes

# ---------------- Tracking (Deep SORT) ----------------
TRACKER_MAX_AGE = 30                # frames to keep a lost track alive
TRACKER_N_INIT = 3                  # consecutive detections before confirming a track
TRACKER_MAX_COSINE_DISTANCE = 0.4   # appearance-similarity threshold
EMBEDDER = "mobilenet"              # lightweight CNN used for Deep SORT's appearance features

# ---------------- I/O ----------------
OUTPUT_DIR = "outputs"
DEFAULT_OUTPUT_VIDEO = "outputs/output.mp4"

# ---------------- UI ----------------
FONT_SCALE = 0.5
BOX_THICKNESS = 2
