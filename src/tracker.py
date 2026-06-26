"""
tracker.py
----------
Wraps `deep_sort_realtime` so the rest of the app only deals with simple
dictionaries instead of library-specific Track objects.

Deep SORT = SORT (Kalman filter + Hungarian assignment) PLUS a CNN
appearance embedding, which is what lets it keep a consistent ID on an
object even after a brief occlusion -- something plain SORT struggles with.
This directly satisfies the brief's "Apply object tracking using algorithms
like SORT or Deep SORT" requirement.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.4,
                 embedder="mobilenet"):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nms_max_overlap=1.0,
            embedder=embedder,
            half=True,
            bgr=True,
        )

    def update(self, detections, frame):
        """
        Args:
            detections: list of ([x, y, w, h], confidence, class_id) -- the
                exact output of YOLODetector.detect()
            frame: the current BGR frame (Deep SORT uses it to compute the
                appearance embedding for each box)

        Returns:
            List of dicts: {track_id, bbox (x1,y1,x2,y2), class_id, conf}
            Only CONFIRMED tracks are returned (i.e. tracks Deep SORT is
            confident are real, not noise).
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = track.to_ltrb()
            results.append({
                "track_id": track.track_id,
                "bbox": (int(l), int(t), int(r), int(b)),
                "class_id": track.get_det_class(),
                "conf": track.get_det_conf() or 0.0,
            })
        return results
