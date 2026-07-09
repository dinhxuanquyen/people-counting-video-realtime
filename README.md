# 👥 People Counting Video and Realtime

A Computer Vision project for detecting, tracking, and counting people in videos and realtime camera streams using YOLOv8 and DeepSORT.

## 📖 Overview

This project was developed as a Computer Vision course project at Phenikaa University.

The system is capable of:

- Detecting people in videos and realtime camera feeds.
- Tracking individuals using unique IDs.
- Counting people entering and leaving a specific area.
- Counting the number of people currently inside a monitored area.
- Logging counting results for further analysis.

## 🚀 Features

### 1. People Detection

- Detect people using YOLOv8.
- Real-time object detection.
- Bounding box visualization with confidence scores.

### 2. People Tracking

- Multi-object tracking using DeepSORT.
- Assign unique IDs to each detected person.
- Maintain tracking consistency across frames.

### 3. Entry / Exit Counting

- Define virtual zones or lines.
- Count people entering an area.
- Count people leaving an area.
- Display current occupancy in real time.

### 4. Realtime Camera Processing

- Webcam support.
- Live counting and monitoring.
- Real-time visualization.

### 5. Data Logging

- Save counting statistics to CSV.
- Track historical entry and exit records.

---

## 🛠 Technologies Used

### Computer Vision

- YOLOv8
- DeepSORT
- OpenCV

### Programming Language

- Python

### Training & Dataset

- Roboflow
- Google Colab

### Libraries

- Ultralytics
- NumPy
- Pandas
- SciPy

---

## 🧠 Model Information

### Detection Model

YOLOv8 Nano (YOLOv8n)

### Tracking Algorithm

DeepSORT

### Dataset

- 1000+ images for people detection.
- Images collected under various conditions:
  - Different viewpoints
  - Different lighting conditions
  - Different clothing styles

### Dataset Split

- Training: 70%
- Validation: 20%
- Testing: 10%

---

## 📊 Performance

Model evaluation results:

| Metric | Value |
|----------|----------|
| Precision | 92.5% |
| Recall | 85.0% |
| mAP50 | 93.1% |
| mAP50-95 | 70.8% |

Inference speed:

| Stage | Time |
|---------|---------|
| Preprocess | 0.7 ms |
| Inference | 5.4 ms |
| Postprocess | 7.4 ms |

---

## 📂 Project Structure

```text
People_counting_video_and_realtime
│
├── main.py
├── main2.py
├── main4.py
├── tracker.py
├── people_log.csv
│
├── Runs/
├── weights/
│
└── README.md
```

---

## ▶️ How To Run

### 1. Clone repository

```bash
git clone https://github.com/dinhxuanquyen/people-counting-video-realtime.git
```

### 2. Install dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pandas
```

### 3. Run project

```bash
python main.py
```

or

```bash
python main2.py
```

depending on the desired functionality.

---

## 📈 Applications

- Security monitoring
- Retail customer counting
- Employee management
- Warehouse monitoring
- Smart building analytics

---

## 👨‍💻 Team Members

- Cao Mậu Thành Đạt
- Nguyễn Trần Việt Anh
- Đinh Xuân Quyền
- Võ Quang Giáp
- Trần Đình Đàn

### Instructor

ThS. Nguyễn Thị Khánh Trâm

Phenikaa University – Faculty of Information Technology

---

## 🔮 Future Improvements

- Improve detection accuracy in low-light conditions.
- Handle heavy occlusion more effectively.
- Integrate ByteTrack or FairMOT.
- Deploy on edge devices and surveillance systems.
- Build a complete smart monitoring platform.

---

⭐ If you find this project useful, please give it a star!
