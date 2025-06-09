# TTSRM: Taiwan Traffic Sign & Road Marking Dataset for YOLOv12

> **Course:** Deep Learning in Computer Vision (深度學習應用於電腦視覺)  
> **Project Type:** Individual Project  
> **Semester:** National Taiwan University 113-2 (2024-2學期)  
> **Instructor:** Prof. Rih-Teng Wu (吳日騰)  
> **Teaching Assistant:** Cheng-Ju Tsai (蔡政儒)  

---

## Overview (專案概述)

This project tackles Taiwan’s distinctive traffic environment—Traditional Chinese signage, unique local symbols, dense urban lane markings—by developing a lightweight **YOLOv12s** detection system. It simultaneously identifies traffic signs and road markings, and lays the groundwork for automated infrastructure checks (e.g. lane‐mark legality, sign placement/height, permitted travel directions).

---

## Original Dataset (原始資料集)

- **Taiwan Traffic Sign Recognition Benchmark (TTSRB)**  
  – **Author:** ExodusTW  
  – **License:** CC BY-NC-SA 3.0 (academic use only; please attribute)  
  – **Contents:**  
  &nbsp;&nbsp;• 27 common traffic‐sign classes  
  &nbsp;&nbsp;• Training set: 711 images (plus augmented 20,913 images)  
  &nbsp;&nbsp;• Test set: 377 images  
  – **Source:** Google Street View screenshots  

---

## Our Contributions (本次貢獻)

1. **Road‐mark Expansion (路面標線擴充)**  
   – Collected 76 additional Google Maps Street View images containing lane markings and multi-object scenes.  
2. **Multi-Object Annotations (多物件場景標註)**  
   – Added 66 multi-object annotations to the original 1,319 single-object images, resulting in **1,395** total.  
3. **Integrated 37-Class Dataset (37類整合資料集)**  
   – Combined traffic signs and road markings into a unified 37-class YOLO format.  
4. **CPU-Only Training Optimization (CPU平臺優化訓練)**  
   – Employed caching, early stopping, and selective augmentations to achieve **mAP@0.5 = 0.63** on CPU.  

---

## Usage (使用方式)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/YourUser/TTSRM-Taiwan-Traffic-Sign-Road-Marking.git
   cd TTSRM-Taiwan-Traffic-Sign-Road-Marking
2. **Prepare data**
   Place the original TTSRB images and your additional lane-mark images/labels under data/images/ and data/labels/.
3. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python
4. **Train the model**  
   ```bash
   python YOLO_cpu_train.py --data data/traffic.yaml --model yolov12s.yaml --device cpu --epochs 100 --batch 16 --cache True
5. **Run inference**
   ```bash
   python detect.py --weights runs/train/weights/best.pt --source examples/*.jpg

## Repository Structure (儲存庫結構)
  ```text
  .
  ├── LICENSE.md           # CC BY-NC-SA 3.0
  ├── README.md
  ├── data/
  │   ├── images/          # TTSRB + additional street-view images
  │   └── labels/          # YOLO-format annotations
  ├── code/
  │   ├── YOLO_cpu_train.py
  │   └── YOLO_detect.py
  └── yolov12s.yaml        # model configuration
  ```

## Acknowledgements (致謝)

This work was carried out as part of the “Deep Learning in Computer Vision” course (National Taiwan University Semester 113-2). We would like to thank:

- **Professor Rih-Teng Wu (吳日騰)** for his guidance and support throughout this project.
- **Teaching Assistant Cheng-Ju Tsai (蔡政儒)** for helpful discussions and feedback.
- **ExodusTW** for providing the original Taiwan Traffic Sign Recognition Benchmark dataset under CC BY-NC-SA 3.0.
- **The Ultralytics team** for the YOLOv12 framework.

