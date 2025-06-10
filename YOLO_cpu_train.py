# YOLO_cpu_train.py

from ultralytics import YOLO

def main():
    # 1. 載入模型架構（不含預訓練權重）
    model = YOLO('yolov12s.yaml')

    # 2. 開始訓練，全部在 CPU 上跑，同時關掉最耗時的增強，並啟用 cache 加速 IO
    results = model.train(
        data='D:/NTU CE/1132_Deep Learning/Project/data/traffic.yaml',  # 你的資料設定檔
        model='yolov12s.yaml',   # 小型架構
        device='cpu',            # 只用 CPU
        epochs=100,              # 你原本的訓練週期
        patience=10,             # 早停設為 10
        batch=16,                # CPU 可能吃不消太大 batch，視情況可調小
        imgsz=416,               # 小一點的尺寸可提速
        workers=12,              # DataLoader 多工數量，建議設成 CPU 核心數或略低
        cache=True,              # 把影像預先快取到記憶體／磁碟，加速 IO
        amp=False,               # CPU 上不做半精度
        rect=True,               # 啟用矩形訓練，省去 padding 時間
        # 關閉重擾動、mixup、copy_paste、auto_augment
        mosaic=0.5,
        mixup=0.5,
        copy_paste=0.5,
        auto_augment='randaugment',
        erasing=0.3,
        # 你可以保留最基本的隨機翻轉
        flipud=0.5,
        fliplr=0.5,
        # Cosine LR 可以保留
        cos_lr=True,
        lr0=0.01,                # 初始學習率
        lrf=0.0001,               # 最終學習率比例
    )

    print(f"訓練完成，結果保存在：{results.save_dir}")

if __name__ == "__main__":
    main()
