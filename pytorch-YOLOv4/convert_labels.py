import os
import glob
from PIL import Image # 使用 Pillow 庫來讀取圖片尺寸

# --- 1. 設定你的路徑 ---
# 你的 .txt 標籤檔案所在的資料夾
label_dir = '/home/chester/pytorch-YOLOv4/datasets/stamp_data/labels'

# 你的圖片檔案所在的資料夾
# 腳本需要讀取圖片尺寸來進行反歸一化計算
image_dir = '/home/chester/pytorch-YOLOv4/datasets/stamp_data/images' 

# 備份原始標籤檔案的資料夾名稱
# 腳本會自動在 label_dir 下建立這個資料夾
backup_dir_name = 'backup_original_labels'

# --- 2. 建立備份資料夾 ---
backup_path = os.path.join(label_dir, backup_dir_name)
if not os.path.exists(backup_path):
    os.makedirs(backup_path)
    print(f"備份資料夾已建立於: {backup_path}")

# --- 3. 核心轉換函式 ---
def convert_yolo_to_voc(label_file, img_width, img_height):
    """
    將 YOLO 格式轉換為此專案期望的 [xmin, ymin, xmax, ymax, class_id] 格式。
    """
    new_lines = []
    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = parts[0]
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        pixel_cx = cx * img_width
        pixel_cy = cy * img_height
        pixel_w = w * img_width
        pixel_h = h * img_height

        x_min = int(pixel_cx - (pixel_w / 2))
        y_min = int(pixel_cy - (pixel_h / 2))
        x_max = int(pixel_cx + (pixel_w / 2))
        y_max = int(pixel_cy + (pixel_h / 2))
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width - 1, x_max)
        y_max = min(img_height - 1, y_max)
        
        # --- 關鍵修正 ---
        # 將 class_id 放在行的最後
        new_lines.append(f"{x_min} {y_min} {x_max} {y_max} {class_id}\n")

    with open(label_file, 'w') as f:
        f.writelines(new_lines)

# --- 4. 執行轉換 ---
def main():
    # 獲取所有 .txt 標籤檔案
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    if not label_files:
        print(f"錯誤: 在 '{label_dir}' 中找不到任何 .txt 標籤檔案。請檢查路徑。")
        return

    print(f"找到 {len(label_files)} 個標籤檔案。開始轉換...")
    
    converted_count = 0
    failed_count = 0

    for label_file in label_files:
        # 取得主檔名 (不含副檔名)
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        
        # 尋找對應的圖片檔案
        possible_img_paths = [
            os.path.join(image_dir, f"{base_name}.jpg"),
            os.path.join(image_dir, f"{base_name}.png"),
            os.path.join(image_dir, f"{base_name}.jpeg"),
        ]
        
        img_path = None
        for path in possible_img_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if not img_path:
            print(f"錯誤: 找不到標籤檔案 '{label_file}' 對應的圖片。已跳過。")
            failed_count += 1
            continue
            
        # 備份原始檔案
        backup_file_path = os.path.join(backup_path, os.path.basename(label_file))
        import shutil
        shutil.copyfile(label_file, backup_file_path)

        try:
            # 獲取圖片尺寸
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # 執行轉換
            convert_yolo_to_voc(label_file, img_width, img_height)
            converted_count += 1
            if converted_count % 50 == 0:
                 print(f"已處理 {converted_count}/{len(label_files)} 個檔案...")

        except Exception as e:
            print(f"處理檔案 '{label_file}' 時發生錯誤: {e}")
            failed_count += 1
            # 如果出錯，可以選擇從備份還原
            # shutil.copyfile(backup_file_path, label_file)

    print("\n--- 轉換完成 ---")
    print(f"成功轉換檔案數: {converted_count}")
    print(f"失敗或跳過檔案數: {failed_count}")
    if converted_count > 0:
        print(f"所有原始標籤檔案已備份至: {backup_path}")
        print("現在你的標籤檔案已是 PASCAL VOC 格式 (<class_id> <xmin> <ymin> <xmax> <ymax>)。")


if __name__ == '__main__':
    main()