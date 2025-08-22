import os
from PIL import Image

# --- 請修改這裡的路徑 ---
# 輸入資料夾：您存放原始圖片的地方
input_folder = '/home/chester/pytorch-YOLOv4/全部公司四大報表_灰階_JPG'

# 輸出資料夾：裁切後的圖片將儲存在這裡 (如果資料夾不存在，程式會自動建立)
output_folder = '/home/chester/pytorch-YOLOv4/全部公司四大報表_灰階_JPG_cut_down'
# --- 路徑修改結束 ---

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 獲取資料夾中所有檔案的列表
try:
    files = os.listdir(input_folder)
except FileNotFoundError:
    print(f"錯誤：找不到輸入資料夾 '{input_folder}'。請檢查路徑是否正確。")
    exit()

print(f"開始處理資料夾：{input_folder}")
print(f"裁切後的圖片將儲存至：{output_folder}\n")

# 遍歷所有檔案
for filename in files:
    # 檢查是否為支援的圖片格式
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            # 組合完整的檔案路徑
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 開啟圖片
            with Image.open(input_path) as img:
                width, height = img.size

                # 計算裁切區域
                # 我們要保留底部 1/3，所以裁切的起點 Y 座標是頂部的 2/3 處
                top = height * 2 // 3
                left = 0
                right = width
                bottom = height

                # 定義裁切框 (left, top, right, bottom)
                crop_box = (left, top, right, bottom)

                # 進行裁切
                cropped_img = img.crop(crop_box)

                # 儲存裁切後的圖片
                cropped_img.save(output_path)
                print(f"已處理並儲存: {filename}")

        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {e}")

print("\n所有圖片處理完成！")