# ======================================================================
# 1. 匯入必要的函式庫
# ======================================================================
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# ======================================================================
# 2. 中文路徑處理函式 (保持不變)
# ======================================================================
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """
    使用 NumPy 和 cv2.imdecode 來讀取包含非 ASCII 字元路徑的圖片。
    """
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, flags)

# ======================================================================
# 3. 全新簡化的 add_noise_only 函式
# ======================================================================
def add_noise_only(input_path, out_path, add_noise=True, seed=42):
    """
    在不改變圖片尺寸的情況下，為其添加類似傳真的雜訊。
    - 將圖片轉為 1-bit 黑白。
    - 添加垂直線、水平線和鹽胡椒噪點。
    - 不會對圖片進行縮放或旋轉。
    """
    rng = np.random.default_rng(seed)

    # -- 1. 讀取並轉灰階
    img = imread_unicode(str(input_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"警告：無法讀取圖片 {input_path}，已跳過。")
        return

    # -- 2. 轉 1-bit + Floyd-Steinberg 抖動
    pil = Image.fromarray(img)
    pil = pil.convert("1")

    # -- 3. 人工雜訊
    if add_noise:
        # 將 1-bit 圖像(0/1)轉為 8-bit 陣列(0/255)以便操作
        arr = np.array(pil, dtype=np.uint8) * 255

        # 垂直黑線 (可自行調整 range 內的數字來增加/減少線條數量)
        for _ in range(rng.integers(0, 2)): # 隨機產生 0 或 1 條黑線
            x = rng.integers(0, arr.shape[1])
            arr[:, x] = 0
            
        # 水平缺行 (白線)
        for _ in range(rng.integers(0, 2)): # 隨機產生 0 或 1 條白線
            y = rng.integers(0, arr.shape[0])
            arr[y, :] = 255
            
        # 細鹽胡椒噪
        noise_level = 0.0001 # 雜訊密度，0.001 表示 0.1% 的像素會被翻轉
        mask = rng.random(arr.shape) < noise_level
        arr[mask] = 255 - arr[mask]
        
        # 將處理過的 8-bit 陣列轉回 Pillow 圖片
        pil = Image.fromarray(arr, mode="L")
        
        # 再次轉換為 1-bit，確保雜訊是純黑白的，並獲得最佳抖動效果
        pil = pil.convert("1")

    # -- 4. 輸出
    # 根據副檔名決定儲存方式，對於 1-bit 圖像，TIFF 或 PNG 是更好的選擇
    suffix = Path(out_path).suffix.lower()
    if suffix == ".tiff" or suffix == ".tif":
        # Group 4 壓縮是專為 1-bit 圖像設計的，效率很高
        pil.save(out_path, compression="group4")
    else:
        # 對於 PNG、JPG 等格式，Pillow 會自動處理
        pil.save(out_path)

# ======================================================================
# 4. 主程式區塊 (處理整個資料夾)
# ======================================================================
if __name__ == "__main__":
    # --- 設定路徑 (使用 raw string r"..." 避免反斜線問題) ---
    input_dir = Path(r"C:\Users\chester\Desktop\commeet\全部公司四大報表_灰階_JPG")
    # 建議輸出資料夾名稱可以更明確
    output_dir = Path(r"C:\Users\chester\Desktop\全部公司四大報表_灰階_JPG_add_noise_very_hard")

    # --- 建立輸出資料夾 (如果不存在的話) ---
    print(f"輸入資料夾: {input_dir}")
    print(f"輸出資料夾: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 篩選要處理的圖片類型 ---
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    files_to_process = []
    for ext in image_extensions:
        files_to_process.extend(input_dir.glob(ext))
    
    if not files_to_process:
        print("\n警告：在輸入資料夾中找不到任何圖片檔案。請檢查路徑和檔案類型。")
    else:
        print(f"\n找到 {len(files_to_process)} 個圖片檔案，開始處理...")

    for input_path in files_to_process:
        output_path = output_dir / input_path.with_suffix('.jpg').name
        print(f"正在處理: {input_path.name}")
        try:
            # 呼叫新的函式
            add_noise_only(
                input_path=input_path,
                out_path=output_path,
                add_noise=True
            )
            print(f" -> 已儲存至: {output_path.name}")
        except Exception as e:
            print(f" !! 處理失敗: {input_path.name} -> 錯誤: {e}")

    print("\n所有檔案處理完成！")