<img width="950" height="494" alt="image" src="https://github.com/user-attachments/assets/6a31d61a-d08b-4a9c-8506-22b35e07f6d2" /># 👥 People Counting Video and Realtime

![Project Banner](images/banner.png)

## 📖 Overview

This project was developed for the Computer Vision course at Phenikaa University.

The system uses YOLOv8 and DeepSORT to detect, track, and count people in video streams and realtime camera feeds.

### Main Features

- Detect people in videos and realtime cameras
- Track people using unique IDs
- Count people entering and leaving an area
- Count the number of people currently inside a zone
- Export counting data to CSV
- Realtime monitoring

---

## 🎯 Project Objectives

This project aims to:

- Automatically detect people
- Track movement across frames
- Count entries and exits accurately
- Monitor occupancy in realtime
- Support security and business analytics applications

---

# 📸 Demo

## 1️⃣ People Detection

Detect people using YOLOv8.
<img width="950" height="494" alt="image" src="https://github.com/user-attachments/assets/1fe781fa-f588-4342-86c8-63a72f1a12a8" />

## 2️⃣ Entry Counting

Count people entering a monitored area.

<img width="887" height="396" alt="image" src="https://github.com/user-attachments/assets/929ac4c4-b668-440f-b8e3-3aa1dd46afac" />
<img width="887" height="396" alt="image" src="https://github.com/user-attachments/assets/a2a8c213-a866-4dfb-b9c8-7b6cff61384c" />

## 3️⃣ Exit Counting

Count people leaving a monitored area.
<img width="555" height="289" alt="image" src="https://github.com/user-attachments/assets/a4e491f1-18ce-4df2-9259-eaac74b8bf7b" />



## 4️⃣ Realtime Camera Monitoring

Realtime detection and counting using webcam.

<img width="751" height="572" alt="image" src="https://github.com/user-attachments/assets/fb14bb4e-c665-4cd4-8463-199a26293676" />
<img width="594" height="450" alt="image" src="https://github.com/user-attachments/assets/a5df705b-2d9c-4059-bbe0-823772a8c170" />

## 5️⃣ Employee & Customer Counting

Classify and count employees and customers separately.
<img width="768" height="574" alt="image" src="https://github.com/user-attachments/assets/d5c92b50-6f8f-4032-8fda-0ac610b506c8" />
<img width="762" height="524" alt="image" src="https://github.com/user-attachments/assets/29d906b8-84ba-479b-bb43-60c2c0f34bbd" />


---

## 6️⃣ Statistics Dashboard

Counting statistics and logs.
<img width="889" height="181" alt="image" src="https://github.com/user-attachments/assets/a46224f7-3015-4703-be1f-c959e10e7206" />

<img width="893" height="269" alt="image" src="https://github.com/user-attachments/assets/8717f9ca-32d8-4890-b454-17f8391b7405" />

# 🏗 System Architecture

```text
Camera / Video
       │
       ▼
   YOLOv8
(People Detection)
       │
       ▼
   DeepSORT
(Object Tracking)
       │
       ▼
 Counting Logic
       │
       ▼
 CSV Logging
       │
       ▼
 Realtime Display
```

# 🛠 Technologies Used

## Programming Language

- Python

## Computer Vision

- YOLOv8
- DeepSORT
- OpenCV

## Data Processing

- NumPy
- Pandas
- SciPy

## Training Platform

- Roboflow
- Google Colab

---

# 📊 Model Performance

| Metric | Value |
|----------|----------|
| Precision | 92.5% |
| Recall | 85.0% |
| mAP50 | 93.1% |
| mAP50-95 | 70.8% |

### Inference Speed

| Stage | Time |
|---------|---------|
| Preprocess | 0.7 ms |
| Inference | 5.4 ms |
| Postprocess | 7.4 ms |

---

# 📂 Project Structure

```text
people-counting-video-realtime
│
├── images/
│   ├── banner.png
│   ├── detection.png
│   ├── entry-count.png
│   ├── exit-count.png
│   ├── realtime.png
│   ├── employee-customer.png
│   └── statistics.png
│
├── main.py
├── main2.py
├── main4.py
├── tracker.py
├── people_log.csv
│
└── README.md
```

---

# 🚀 Installation

Clone repository:

```bash
git clone https://github.com/dinhxuanquyen/people-counting-video-realtime.git
```

Install dependencies:

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pandas
pip install scipy
```

Run project:

```bash
python main.py
```

or

```bash
python main2.py
```

---

# 💡 Applications

- Smart retail stores
- Security monitoring
- Building occupancy analysis
- Employee attendance monitoring
- Warehouse management
- Customer flow analysis

---

# 👨‍💻 Team Members

| Student | ID |
|----------|----------|
| Cao Mậu Thành Đạt | 22010338 |
| Nguyễn Trần Việt Anh | 22010341 |
| Đinh Xuân Quyền | 22010342 |
| Võ Quang Giáp | 22010343 |
| Trần Đình Đàn | 22010053 |

### Instructor

**ThS. Nguyễn Thị Khánh Trâm**

Faculty of Information Technology  
Phenikaa University

---

# 🔮 Future Improvements

- Improve low-light performance
- Handle heavy occlusion
- Integrate ByteTrack
- Integrate FairMOT
- Deploy on edge devices
- Build a complete smart monitoring platform

---

⭐ If you like this project, please give it a star!
