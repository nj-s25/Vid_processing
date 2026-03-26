import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import imageio

st.title("🎮 Fortnite Video Analyzer (Cloud Version)")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    reader = imageio.get_reader(temp_file.name)
    frames = []

    st.write("Processing...")

    for i, frame in enumerate(reader):
        if i % 2 != 0:  # skip frames
            continue

        results = model(frame)
        annotated = results[0].plot()

        frames.append(annotated)

        if i > 50:  # limit for cloud
            break

    output_path = "output.mp4"
    imageio.mimsave(output_path, frames, fps=10)

    st.success("Done!")

    st.video(output_path)
