import cv2
import numpy as np
from detector import StampDetector # 從你修改後的 detector.py 檔案中匯入類別

# --- 步驟 1: 初始化檢測器 (這部分不變) ---
print("正在準備印章檢測器...")
detector = StampDetector()
print("檢測器準備就緒！")

# --- 步驟 2: 讀取圖片並呼叫檢測器 (這部分不變) ---
image_path = r"/home/chester/pytorch-YOLOv4/全部公司四大報表_灰階_JPG_add_noise_very_very_hard/AES-KY_2024_合併_現金流量表_2.jpg"
#my_image = cv2.imread(image)
image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

if image is not None:
    # 呼叫 predict 方法，它會返回一個包含「所有」高於信心分數結果的列表
    import time
    start_time = time.time()
    detected_stamps = detector.get_stamp_positions(image)
    print("detected_stamps:", detected_stamps)
    end_time = time.time()
    print(f"印章檢測時間: {end_time - start_time:.2f} 秒")
    # --- 步驟 3: 處理返回的結果列表 ---
    if not detected_stamps:
        print(f"在 '{image}' 中沒有找到任何印章。")
    else:
        
        # 使用 for 迴圈遍歷「所有」檢測到的印章
        for i, stamp_info in enumerate(detected_stamps):
            box = stamp_info['box']
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            obj_confidence = stamp_info['obj_confidence']
            class_conf = stamp_info['class_conf']
            class_name = stamp_info['class_name']
            
            print(f"\n--- 印章 #{i+1} ---")
            print(f"  - 類別: {class_name}")
            print(f"  - 分數: {obj_confidence:.4f}")
            print(f"  - 類別分數: {class_conf:.4f}")
            print(f"  - 座標: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # 在這裡，同事可以根據每個印章的座標 (x1, y1, x2, y2) 進行他們需要的操作
            # 例如：將所有印章的座標記錄到資料庫、將所有印章都裁切下來等

else:
    print(f"無法讀取圖片: {image}")