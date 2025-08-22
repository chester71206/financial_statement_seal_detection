import cv2
import numpy as np
import os
import time

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """
    取代 cv2.imread，使其支援包含非 ASCII 字元 (如中文) 的路徑。
    """
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, flags)
        return img
    except Exception as e:
        print(f"使用 imread_unicode 讀取檔案時發生錯誤: {e}")
        return None

def imwrite_unicode(path, img):
    """
    取代 cv2.imwrite，使其支援包含非 ASCII 字元 (如中文) 的路徑。
    """
    try:
        ext = os.path.splitext(path)[1]
        result, buffer = cv2.imencode(ext, img)
        if result:
            with open(path, mode='wb') as f:
                f.write(buffer)
            return True
        else:
            return False
    except Exception as e:
        print(f"使用 imwrite_unicode 寫入檔案時發生錯誤: {e}")
        return False



def detect_red_seals(image_path):
    """
    偵測圖片中的紅色印章並用矩形框標示出來。

    Args:
        image_path (str): 圖片檔案的路徑。

    Returns:
        tuple: 一個包含 (處理後的圖片, 偵測到的印章數量) 的元組。
    """
    # 讀取圖片
    image =imread_unicode(image_path)
    if image is None:
        print(f"錯誤: 無法讀取圖片 {image_path}")
        return None, 0

    output_image = image.copy()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義紅色的 HSV 範圍 (涵蓋暗紅色到亮紅色)
    lower_red1 = np.array([0, 20, 30])
    upper_red1 = np.array([10, 255, 255])
    
    # 範圍 2: 偏亮的紅色
    lower_red2 = np.array([160, 20, 30])
    upper_red2 = np.array([180, 255, 255])

    # 建立並合併遮罩
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    # # 形態學處理 - 填補印章內部空洞
    # kernel = np.ones((5, 5), np.uint8)
    # red_mask_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 尋找輪廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    initial_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # 面積閾值
            # 計算最小外接矩形，並存入列表
            # 格式為 [x, y, w, h]
            initial_boxes.append(list(cv2.boundingRect(contour)))

    print(f"階段一：找到了 {len(initial_boxes)} 個原始碎塊矩形。")

    # ========================================================================
    # 階段二：暴力雙重 for 迴圈，合併互相包含的矩形
    # ========================================================================

    # 這個合併過程會一直重複，直到沒有任何框可以再被合併為止
    while True:
        merged_in_this_pass = False
        
        # 用來存放這一輪合併後的結果
        merged_boxes = []
        
        # 當還有待處理的框時，繼續迴圈
        while len(initial_boxes) > 0:
            
            # 1. 從列表中取出第一個框作為基準
            base_box = initial_boxes.pop(0)
            
            # 2. 建立一個臨時列表，存放這一輪沒有與 base_box 合併的框
            remaining_boxes_this_round = []
            
            # 3. 遍歷剩下的所有框，檢查是否與 base_box 「相交」
            for other_box in initial_boxes:
                
                # --- 核心修改：將「包含」檢查替換為「相交」檢查 ---
                # 計算兩個矩形的座標
                base_x1, base_y1, base_w, base_h = base_box
                base_x2, base_y2 = base_x1 + base_w, base_y1 + base_h
                
                other_x1, other_y1, other_w, other_h = other_box
                other_x2, other_y2 = other_x1 + other_w, other_y1 + other_h
                
                # 判斷是否有重疊區域
                # (no overlap condition) is not true
                has_intersection = not (base_x2 < other_x1 or 
                                        base_x1 > other_x2 or 
                                        base_y2 < other_y1 or 
                                        base_y1 > other_y2)

                # 4. 如果有相交，就合併它們
                if has_intersection:
                    # 計算兩個框合併後的邊界
                    min_x = min(base_x1, other_x1)
                    min_y = min(base_y1, other_y1)
                    max_x = max(base_x2, other_x2)
                    max_y = max(base_y2, other_y2)
                    
                    # 更新 base_box 為合併後的大框
                    base_box = [min_x, min_y, max_x - min_x, max_y - min_y]
                    
                    # 標記這一輪發生了合併
                    merged_in_this_pass = True
                else:
                    # 如果沒有相交，就先留著，下一輪再處理
                    remaining_boxes_this_round.append(other_box)
            
            # 5. 將處理完的 base_box (可能是合併過的) 存入結果
            merged_boxes.append(base_box)
            
            # 6. 更新待處理列表，準備檢查下一個 base_box
            initial_boxes = remaining_boxes_this_round

        # 將這一輪的合併結果作為下一輪的輸入
        initial_boxes = merged_boxes
        
        # 如果在一整輪大迴圈中都沒有發生任何合併，表示合併已完成，跳出迴圈
        if not merged_in_this_pass:
            break

    final_boxes = initial_boxes
    print(f"階段二：合併後剩下 {len(final_boxes)} 個最終矩形。")

    # ========================================================================
    # 階段三：繪製最終合併後的矩形框
    # ========================================================================
    
    for (x, y, w, h) in final_boxes:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 20)

    return output_image, final_boxes

# --- 主程式執行 ---
if __name__ == "__main__":
    # --- 請在此設定您的輸入與輸出路徑 ---
    # 使用 r'' 來確保 Windows 路徑中的反斜線被正確解讀
    input_dir = r'C:\Users\chester\Desktop\commeet\全部公司四大報表彩色JPG'
    output_dir = './全部公司四大報表彩色JPG_yolo'
    # ------------------------------------
# --- 新增 3：定義 YOLO 標籤的輸出資料夾 ---
    # 讓標籤檔和圖片檔分開存放，更整潔
    label_dir = os.path.join(output_dir, "labels")

    # 檢查並建立輸出資料夾
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立圖片輸出資料夾: {output_dir}")
    # --- 新增 4：建立標籤資料夾 ---
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        print(f"已建立標籤輸出資料夾: {label_dir}")

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"在 {input_dir} 中找不到任何 JPG 圖片。")
        exit()

    print(f"準備開始處理 {len(image_files)} 張圖片...")
    start_time = time.time()

    for index, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # --- 修改 5：接收回傳的 final_boxes ---
        result_image, final_boxes = detect_red_seals(input_path)
        count = len(final_boxes)

        if result_image is not None:
            # 儲存畫了框的結果圖片
            imwrite_unicode(output_path, result_image)
            
            # --- 新增 6：產生 YOLOv8 格式的 .txt 標籤檔 ---
            if count > 0:
                # 獲取圖片的原始尺寸
                img_h, img_w, _ = result_image.shape
                
                # 設定標籤檔案的路徑 (與圖片檔同名，副檔名為 .txt)
                label_filename = os.path.splitext(filename)[0] + ".txt"
                label_path = os.path.join(label_dir, label_filename)
                
                # 寫入標籤檔
                with open(label_path, 'w') as f:
                    for (x, y, w, h) in final_boxes:
                        # 進行座標轉換與正規化
                        x_center_norm = (x + w / 2) / img_w
                        y_center_norm = (y + h / 2) / img_h
                        width_norm = w / img_w
                        height_norm = h / img_h
                        
                        # 我們的類別只有 "seal"，所以 class_index 固定為 0
                        class_index = 0
                        
                        # 寫入一行 YOLO 格式的標籤
                        f.write(f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

            print(f"({index + 1}/{len(image_files)}) 已處理: {filename} -> 偵測到 {count} 個印章。標籤檔已儲存。")
        else:
            print(f"({index + 1}/{len(image_files)}) 跳過: {filename} (讀取失敗)")
            
    end_time = time.time()
    total_time = end_time - start_time
    print("-" * 30)
    print("全部處理完成！")
    print(f"總共花費時間: {total_time:.2f} 秒。")
    print(f"圖片結果已儲存至: {output_dir}")
    print(f"YOLO 標籤已儲存至: {label_dir}")