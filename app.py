"""
app.py
------
Optional bonus: a Streamlit web dashboard on top of the same detector/
tracker/analytics modules used by main.py. Lets you upload a video (or use
your webcam), watch detection + tracking happen live in the browser, and
see a running table of unique object counts. Great for the demo video
required by the internship instructions.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

import config
from src.analytics import AnalyticsLogger
from src.detector import YOLODetector
from src.tracker import ObjectTracker
from src.utils import FPSCounter, draw_count_panel, draw_fps, draw_tracked_object, get_color

st.set_page_config(page_title="Object Detection & Tracking", layout="wide")
st.title("Real-Time Object Detection & Tracking")
st.caption(" YOLOv8 + Deep SORT")

with st.sidebar:
    st.header("Settings")
    source_type = st.radio("Video source", ["Upload video", "Webcam"])
    model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    conf = st.slider("Confidence threshold", 0.1, 0.9, config.CONFIDENCE_THRESHOLD, 0.05)
    uploaded_file = (
        st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if source_type == "Upload video"
        else None
    )
    run = st.button("▶ Start", use_container_width=True)
    stop = st.button("⏹ Stop", use_container_width=True)

frame_placeholder = st.empty()
col1, col2 = st.columns([1, 1])
stats_placeholder = col1.empty()
fps_placeholder = col2.empty()

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if stop:
    st.session_state.stop_requested = True

if run:
    st.session_state.stop_requested = False

    if source_type == "Upload video" and uploaded_file is None:
        st.warning("Please upload a video file first.")
        st.stop()

    if source_type == "Upload video":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        source = tfile.name
    else:
        source = 0

    detector = YOLODetector(model_path=model_name, conf_threshold=conf)
    tracker = ObjectTracker(
        max_age=config.TRACKER_MAX_AGE,
        n_init=config.TRACKER_N_INIT,
        max_cosine_distance=config.TRACKER_MAX_COSINE_DISTANCE,
        embedder=config.EMBEDDER,
    )
    fps_counter = FPSCounter()
    analytics = AnalyticsLogger(output_dir=config.OUTPUT_DIR)

    cap = cv2.VideoCapture(source)
    frame_idx = 0

    while cap.isOpened():
        if st.session_state.stop_requested:
            break

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

        fps_counter.update()
        draw_fps(frame, fps_counter.get_fps())
        draw_count_panel(frame, analytics.get_live_counts())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        fps_placeholder.metric("FPS", f"{fps_counter.get_fps():.1f}")
        counts = analytics.get_live_counts()
        if counts:
            df = pd.DataFrame(list(counts.items()), columns=["Class", "Unique Count"])
            stats_placeholder.dataframe(df, use_container_width=True, hide_index=True)

    cap.release()
    summary_path = analytics.save_summary()
    st.success(f"Done! Summary saved to `{summary_path}`")

    with open(analytics.csv_path, "rb") as f:
        st.download_button("Download full detection log (CSV)", f, file_name=os.path.basename(analytics.csv_path))

    if source_type == "Upload video":
        os.unlink(source)
else:
    st.info("Choose a video source in the sidebar and click **Start**.")
