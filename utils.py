"""
utils.py
--------
Small, reusable helpers: FPS measurement and all the on-screen drawing
(boxes, labels, FPS counter, live object-count panel). Keeping these out
of main.py/app.py keeps the entry points readable.
"""

import time
import cv2
import numpy as np


class FPSCounter:
    """Rolling-average FPS counter (smoother than a per-frame instant value)."""

    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.timestamps = []

    def update(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.avg_frames:
            self.timestamps.pop(0)

    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])


def get_color(class_id):
    """Deterministic, visually distinct color per class id (same class = same color every run)."""
    rng = np.random.RandomState(int(class_id) * 37 + 7)
    color = tuple(int(c) for c in rng.randint(60, 255, size=3))
    return color


def draw_tracked_object(frame, bbox, track_id, label, conf, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"ID {track_id} | {label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    y_label = max(y1, th + 10)
    cv2.rectangle(frame, (x1, y_label - th - 10), (x1 + tw + 6, y_label), color, -1)
    cv2.putText(frame, text, (x1 + 3, y_label - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def draw_count_panel(frame, counts: dict):
    """Semi-transparent panel in the top-right corner showing live unique-object counts per class."""
    if not counts:
        return
    h, w = frame.shape[:2]
    panel_w = 230
    panel_h = 30 + 22 * len(counts)
    x0, y0 = w - panel_w - 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "Object Counts", (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    for i, (cls_name, count) in enumerate(counts.items()):
        y = y0 + 22 + 22 * (i + 1)
        cv2.putText(frame, f"{cls_name}: {count}", (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
