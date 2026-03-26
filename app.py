import os
import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Fortnite Video Analyzer")

st.title("Fortnite Video Analyzer")

uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi"],
)

if uploaded_file is not None:
    video_bytes = uploaded_file.getvalue()
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    temp_file_path = ""
    output_file_path = ""

    st.video(video_bytes)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name
            
        MODEL_PATH = "best.pt"

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model file not found!")

        model = YOLO(MODEL_PATH)
        
        capture = cv2.VideoCapture(temp_file_path)
        frame_count = 0

        if not capture.isOpened():
            st.error("Unable to open the uploaded video.")
        else:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
                output_file_path = output_file.name

            writer = cv2.VideoWriter(
                output_file_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_message = st.empty()
            status_message.write("Processing...")

            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break

                results = model.track(frame, persist=True, device="cpu")
                output_frame = results[0].plot()
                frame_count += 1

                writer.write(output_frame)
                frame_placeholder.image(output_frame, channels="BGR")

                frame_index += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_index / total_frames, 1.0))

            progress_bar.progress(1.0)
            status_message.write("Processing complete.")
            st.write(f"Total frames processed: {frame_count}")

            if output_file_path and os.path.exists(output_file_path):
                with open(output_file_path, "rb") as output_video_file:
                    processed_video_bytes = output_video_file.read()

                st.video(processed_video_bytes)
                st.download_button(
                    label="Download processed video",
                    data=processed_video_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                )
    finally:
        if "capture" in locals():
            capture.release()
        if "writer" in locals():
            writer.release()
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if output_file_path and os.path.exists(output_file_path):
            os.unlink(output_file_path)
