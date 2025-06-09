from ultralytics import YOLO
import numpy as np

# 1. 載入你訓練好的模型權重
model = YOLO(r"C:/Users/User/yolov12/runs/detect/train2/weights/best.pt")

# 2. 單張圖片推論
results = model.predict(
    source=r"D:/NTU CE/1132_Deep Learning/Project/data/test.png",
    imgsz=640,
    conf=0.2,        # 信心門檻，建議 0.2~0.3
    iou=0.4,         # NMS IoU 門檻，建議 0.4~0.6
    augment=True,     # 開啟 TTA，輔助提升準確度
    save=True,        # 同時把可視化結果存到 runs/detect/predict*
    save_txt=True     # 同時把每個框的座標存成 txt
)

# 3. 解析結果：一次跑完所有偵測到的 box
for r in results:                      # results 裡通常只有一個 result 物件
    names = r.names                   # 類別名稱字典
    # r.boxes.xyxy 會是一個 [N,4] tensor，絕對像素座標 (x1,y1,x2,y2)
    for xyxy, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        x1, y1, x2, y2 = xyxy.cpu().numpy()
        score   = float(conf.cpu().numpy())
        cls_id  = int(cls.cpu().numpy())
        label   = names[cls_id]
        print(f"{label}: {score:.2f} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
