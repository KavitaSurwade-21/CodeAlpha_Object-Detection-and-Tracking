"""
main.py
-------
CLI entry point for real-time object detection & tracking.

Usage examples:
    python main.py                                  # webcam, default settings
    python main.py --source data/sample.mp4         # run on a video file
    python main.py --source 0 --line-y 300           # enable IN/OUT line counter
    python main.py --classes 0,2 --conf 0.5          # only detect 'person' and 'car'
    python main.py --no-save --show                  # don't save video, just preview

Press 'q' in the preview window to stop early.
"""

import argparse
import os
import sys

import cv2

import config
from src.detector import YOLODetector
from src.tracker import ObjectTracker
from src.counter import LineCounter
from src.analytics import AnalyticsLogger
from src.utils import FPSCounter, get_color, draw_tracked_object, draw_fps, draw_count_panel


def parse_args():
    parser = argparse.ArgumentParser(
        description="CodeAlpha AI Internship - Task 4: Object Detection and Tracking"
    )
    parser.add_argument("--source", type=str, default="0",
                         help="'0' for webcam, or path to a video file")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                         help="YOLOv8 weights file (auto-downloaded if not found locally)")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                         help="Detection confidence threshold (0-1)")
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated COCO class ids to detect, e.g. '0,2,3'. Default = all")
    parser.add_argument("--output", type=str, default=config.DEFAULT_OUTPUT_VIDEO,
                         help="Path to save the annotated output video")
    parser.add_argument("--no-save", action="store_true",
                         help="Don't save an output video (preview only)")
    parser.add_argument("--line-y", type=int, default=None,
                         help="Y pixel coordinate for an IN/OUT crossing counter (optional)")
    parser.add_argument("--no-show", action="store_true",
                         help="Run headless, without a live preview window")
    return parser.parse_args()


def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source
    classes = [int(c) for c in args.classes.split(",")] if args.classes else config.DEFAULT_CLASSES

    print("[INFO] Loading YOLOv8 model... (first run downloads weights automatically)")
    detector = YOLODetector(model_path=args.model, conf_threshold=args.conf, classes=classes)
    tracker = ObjectTracker(
        max_age=config.TRACKER_MAX_AGE,
        n_init=config.TRACKER_N_INIT,
        max_cosine_distance=config.TRACKER_MAX_COSINE_DISTANCE,
        embedder=config.EMBEDDER,
    )
    fps_counter = FPSCounter()
    analytics = AnalyticsLogger(output_dir=config.OUTPUT_DIR)
    line_counter = LineCounter(args.line_y) if args.line_y is not None else None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        sys.exit(1)

    writer = None
    if not args.no_save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))

    print("[INFO] Running. Press 'q' in the preview window to stop.")
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            detections = detector.detect(frame)
            tracked_objects = tracker.update(detections, frame)
            analytics.log_frame(frame_idx, tracked_objects, detector.get_class_name)

            for obj in tracked_objects:
                color = get_color(obj["class_id"] if obj["class_id"] is not None else 0)
                label = detector.get_class_name(obj["class_id"]) if obj["class_id"] is not None else "object"
                draw_tracked_object(frame, obj["bbox"], obj["track_id"], label, obj["conf"], color)

                if line_counter:
                    line_counter.update(obj["track_id"], obj["bbox"])

            fps_counter.update()
            draw_fps(frame, fps_counter.get_fps())
            draw_count_panel(frame, analytics.get_live_counts())

            if line_counter:
                cv2.line(frame, (0, line_counter.line_y), (frame.shape[1], line_counter.line_y), (0, 0, 255), 2)
                cv2.putText(frame, f"IN: {line_counter.in_count}  OUT: {line_counter.out_count}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if writer:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow("CodeAlpha - Object Detection & Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        summary_path = analytics.save_summary()
        print(f"[INFO] Frame-by-frame log saved to: {analytics.csv_path}")
        print(f"[INFO] Summary saved to:            {summary_path}")
        if writer:
            print(f"[INFO] Output video saved to:       {args.output}")


if __name__ == "__main__":
    main()
