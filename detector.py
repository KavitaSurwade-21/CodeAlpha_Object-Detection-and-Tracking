"""
detector.py
-----------
Thin wrapper around Ultralytics YOLOv8 that turns raw model output into a
simple, framework-agnostic list of detections: ([x, y, w, h], confidence, class_id)

This format is exactly what `deep_sort_realtime` expects, which keeps
detector.py and tracker.py decoupled (you could swap YOLO for Faster R-CNN
or YOLO for SSD without touching the tracker).
"""

from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4,
                 iou_threshold=0.45, classes=None):
        """
        Args:
            model_path: path/name of a YOLOv8 .pt weights file. If the file
                doesn't exist locally, Ultralytics auto-downloads it.
            conf_threshold: minimum confidence to keep a detection.
            iou_threshold: IoU threshold used by YOLO's internal NMS.
            classes: list of COCO class ids to restrict detection to
                (e.g. [0, 2] -> person, car). None = all 80 classes.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.class_names = self.model.names  # dict: {0: 'person', 1: 'bicycle', ...}

    def detect(self, frame):
        """
        Run detection on a single BGR frame (as returned by cv2.VideoCapture).

        Returns:
            List of tuples: ([x, y, w, h], confidence, class_id)
            where (x, y) is the TOP-LEFT corner of the box.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([float(x1), float(y1), float(w), float(h)], conf, cls_id))
        return detections

    def get_class_name(self, cls_id):
        return self.class_names.get(cls_id, str(cls_id))
