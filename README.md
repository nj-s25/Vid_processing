# 🎮 Fortnite Player Detection & Video Analyzer (YOLOv8)

An end-to-end computer vision project that detects and localizes players in Fortnite gameplay using **YOLOv8**, with a fully interactive **Streamlit video analyzer app**.

---

## 🚀 Overview

This project focuses on detecting **players and heads** in Fortnite gameplay images and videos. It combines:

* 🧠 Deep Learning (YOLOv8)
* 🎥 Video Processing (OpenCV)
* 🌐 Interactive UI (Streamlit)

The model is trained on a custom dataset and evaluated using **mAP@0.50**, a standard metric for object detection.

---

## ✨ Features

* 🎯 Detect **players and heads** in images and videos
* 🎥 Upload gameplay videos and analyze frame-by-frame
* 📦 Automatic bounding box visualization
* 📊 Model evaluation using mAP@0.50
* 💾 Download processed video output
* ⚡ Optimized for CPU inference

---

## 🧠 Model Details

* Model: **YOLOv8 (yolov8s)**
* Pretrained on: COCO dataset
* Classes:

  * `people`
  * `head`
* Framework: Ultralytics YOLOv8

---

## 📁 Project Structure

```bash
.
├── app.py                  # Streamlit video analyzer app
├── best.pt                 # Trained YOLOv8 model
├── data.yaml               # Dataset configuration
├── requirements.txt        # Dependencies
├── train/                  # Training images + annotations
├── valid/                  # Validation images + annotations
└── notebooks/              # Training & EDA notebooks
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fortnite-detection.git
cd fortnite-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the URL shown in terminal.

---

## 🎥 How It Works

```text
Upload Video → Frame Extraction → YOLO Detection → Bounding Boxes → Rebuild Video → Display & Download
```

---

## 📊 Evaluation

The model is evaluated using:

* **mAP@0.50**
* **mAP@0.50–0.95**

Example:

```python
metrics = model.val(device="cpu")
print(metrics.box.map50)
```

---

## 📦 Dataset

* Format: CSV annotations converted to YOLO format
* Classes: `people`, `head`
* Images include diverse gameplay scenarios with occlusions and motion

---

## ⚠️ Notes

* GPU (P100) is not supported due to CUDA compatibility issues
* All inference is performed on CPU
* Large videos may take time to process

---

## 🚀 Future Improvements

* 🔥 Real-time detection with GPU support
* 📊 Player tracking and analytics dashboard
* 🎯 Small-object detection optimization
* 🌍 Deployment on Streamlit Cloud / HuggingFace

---

## 🧑‍💻 Tech Stack

* Python
* Ultralytics YOLOv8
* OpenCV
* Streamlit
* PyTorch

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgements

* Ultralytics YOLOv8
* Kaggle Competition Dataset
* Roboflow Universe

---
