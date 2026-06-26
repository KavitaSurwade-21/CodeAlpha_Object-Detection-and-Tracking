"""
analytics.py
------------
Logs every detection to a CSV (per-frame, per-object) and keeps a running
set of unique track IDs per class so you can report "Detected 14 unique
people and 3 cars in this video" at the end -- a nice, concrete metric for
a project write-up or LinkedIn post.
"""

import csv
import os
from datetime import datetime
from collections import defaultdict


class AnalyticsLogger:
    def __init__(self, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"detection_log_{timestamp}.csv")
        self.summary_path = os.path.join(output_dir, f"summary_{timestamp}.csv")
        self._init_csv()

        self.unique_ids_per_class = defaultdict(set)
        self.frame_count = 0

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["frame", "timestamp", "track_id", "class_name",
                 "confidence", "x1", "y1", "x2", "y2"]
            )

    def log_frame(self, frame_idx, tracked_objects, class_name_lookup):
        """
        Args:
            frame_idx: current frame number
            tracked_objects: output of ObjectTracker.update()
            class_name_lookup: function mapping class_id -> class name
                (pass detector.get_class_name)
        """
        self.frame_count = frame_idx
        timestamp = datetime.now().isoformat(timespec="seconds")

        rows = []
        for obj in tracked_objects:
            cls_name = class_name_lookup(obj["class_id"]) if obj["class_id"] is not None else "unknown"
            self.unique_ids_per_class[cls_name].add(obj["track_id"])
            x1, y1, x2, y2 = obj["bbox"]
            rows.append([frame_idx, timestamp, obj["track_id"], cls_name,
                         f"{obj['conf']:.3f}", x1, y1, x2, y2])

        if rows:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerows(rows)

    def get_live_counts(self):
        """Unique object count per class, so far."""
        return {cls: len(ids) for cls, ids in self.unique_ids_per_class.items()}

    def save_summary(self):
        with open(self.summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "unique_object_count"])
            for cls, ids in self.unique_ids_per_class.items():
                writer.writerow([cls, len(ids)])
        return self.summary_path
