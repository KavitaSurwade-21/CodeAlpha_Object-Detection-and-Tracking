"""
counter.py
----------
Optional "wow factor" feature: a virtual line that counts how many unique
tracked objects cross it (IN vs OUT). This is the same core idea behind
real-world people-counting / vehicle-counting systems, and it's a strong
talking point for an internship demo video or resume bullet point.

It works on top of the tracker's IDs, so an object is only ever counted
once no matter how many frames it appears in.
"""


class LineCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.counted_ids = set()
        self.in_count = 0
        self.out_count = 0
        self._last_center_y = {}

    def update(self, track_id, bbox):
        """
        Call once per tracked object per frame.

        Args:
            track_id: stable ID assigned by the tracker
            bbox: (x1, y1, x2, y2) of the current frame's box

        Returns:
            "in", "out", or None (no crossing this frame)
        """
        _, y1, _, y2 = bbox
        cy = (y1 + y2) // 2

        prev_cy = self._last_center_y.get(track_id)
        self._last_center_y[track_id] = cy

        if prev_cy is None or track_id in self.counted_ids:
            return None

        crossed_down = prev_cy < self.line_y <= cy
        crossed_up = prev_cy > self.line_y >= cy

        if crossed_down:
            self.in_count += 1
            self.counted_ids.add(track_id)
            return "in"
        if crossed_up:
            self.out_count += 1
            self.counted_ids.add(track_id)
            return "out"
        return None
