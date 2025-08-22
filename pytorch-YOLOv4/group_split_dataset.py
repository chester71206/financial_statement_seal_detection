import os
import random
from collections import defaultdict

# --- 設定 ---
# 圖片和標籤檔所在的目錄
# 假設你的檔名格式是 "公司名_其他資訊.jpg"
IMAGES_DIR = 'datasets/stamp_data/images'
LABELS_DIR = 'datasets/stamp_data/labels'

# 輸出檔案的路徑
NEW_TRAIN_FILE = 'data/train.txt'
NEW_VAL_FILE = 'data/val.txt'

# 驗證集所佔的公司比例 (0.1 代表 10% 的公司會被分到驗證集)
VALIDATION_GROUP_RATIO = 0.1
# --- 設定結束 ---

def get_company_name_from_filename(filename):
    """
    從檔名中提取公司名稱。
    這裡假設公司名是檔名中第一個 '_' 前面的部分。
    例如：'LINEPAY_2024_合併_資產負債表.txt' -> 'LINEPAY'
    您可以根據您的實際命名規則修改這裡的邏輯。
    """
    return filename.split('_')[0]

def group_split_data():
    """
    按照公司名稱分組，切分訓練集和驗證集。
    """
    if not os.path.isdir(IMAGES_DIR):
        print(f"錯誤：找不到圖片目錄 '{IMAGES_DIR}'")
        return

    # 1. 建立一個從公司名到其所有圖片檔名的映射
    company_files = defaultdict(list)
    all_image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in all_image_files:
        company_name = get_company_name_from_filename(filename)
        company_files[company_name].append(filename)

    # 2. 獲取所有獨立的公司名稱並隨機打亂
    unique_companies = list(company_files.keys())
    random.shuffle(unique_companies)
    
    if not unique_companies:
        print("錯誤：在圖片目錄中找不到任何圖片檔案！")
        return

    # 3. 計算分割點，並分割公司列表
    split_index = int(len(unique_companies) * VALIDATION_GROUP_RATIO)
    
    # 確保至少有一個公司在驗證集（如果資料足夠）
    if split_index == 0 and len(unique_companies) > 1:
        split_index = 1
        
    val_companies = unique_companies[:split_index]
    train_companies = unique_companies[split_index:]

    # 4. 根據分割好的公司列表，生成訓練集和驗證集的圖片路徑列表
    train_image_paths = []
    for company in train_companies:
        for filename in company_files[company]:
            train_image_paths.append(os.path.abspath(os.path.join(IMAGES_DIR, filename)) + '\n')

    val_image_paths = []
    for company in val_companies:
        for filename in company_files[company]:
            val_image_paths.append(os.path.abspath(os.path.join(IMAGES_DIR, filename)) + '\n')
            
    # 5. 將路徑列表寫入檔案
    with open(NEW_TRAIN_FILE, 'w') as f:
        f.writelines(train_image_paths)

    with open(NEW_VAL_FILE, 'w') as f:
        f.writelines(val_image_paths)

    # --- 輸出報告 ---
    print("--- 按公司分組，資料分割完成 ---")
    print(f"總公司數: {len(unique_companies)}")
    print(f"訓練集公司數: {len(train_companies)} | 驗證集公司數: {len(val_companies)}")
    print("\n分到【驗證集】的公司:")
    for company in val_companies:
        print(f"  - {company} ({len(company_files[company])} 個檔案)")
        
    print("\n--- 檔案寫入統計 ---")
    print(f"總圖片數: {len(all_image_files)}")
    print(f"訓練集資料: {len(train_image_paths)} 筆已寫入 '{NEW_TRAIN_FILE}'")
    print(f"驗證集資料: {len(val_image_paths)} 筆已寫入 '{NEW_VAL_FILE}'")
    print("-------------------------")


if __name__ == '__main__':
    group_split_data()