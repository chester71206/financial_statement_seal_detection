import re
import matplotlib.pyplot as plt
import os

# --- 1. 設定檔案路徑 ---
# 請將這裡的路徑換成你自己的 log 檔案路徑
log_file_path = '/home/chester/pytorch-YOLOv4/log/log_2025-08-06_17-58-43.txt'
output_image_path = 'yolov4_training_results.jpg'

# --- 2. 解析日誌檔案 ---
# 初始化一個字典來儲存所有數據
data = {
    'steps': [],
    'total_loss': [],
    'loss_xy': [],
    'loss_wh': [],
    'loss_obj': [],
    'loss_cls': [],
    'loss_l2': [],
    'lr': []
}

# 正規表示式，用來抓取我們需要的數據
# 這個 pattern 會匹配 "Train step_XXX: loss : YYY, ..." 這種格式的行
# 它能處理一般的逗號 "," 和全形的逗號 "，"
pattern = re.compile(
    r"Train step_(\d+): loss : ([\d.eE+-]+),"
    r"loss xy : ([\d.eE+-]+),"
    r"loss wh : ([\d.eE+-]+),"
    r"loss obj : ([\d.eE+-]+)[，,]\s*"  # 處理兩種逗號
    r"loss cls : ([\d.eE+-]+),"
    r"loss l2 : ([\d.eE+-]+),"
    r"lr : ([\d.eE+-]+)"
)

print(f"正在讀取日誌檔案: {log_file_path}")

try:
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # 將抓取到的數據轉換成數字並存入字典
                data['steps'].append(int(match.group(1)))
                data['total_loss'].append(float(match.group(2)))
                data['loss_xy'].append(float(match.group(3)))
                data['loss_wh'].append(float(match.group(4)))
                data['loss_obj'].append(float(match.group(5)))
                data['loss_cls'].append(float(match.group(6)))
                data['loss_l2'].append(float(match.group(7)))
                data['lr'].append(float(match.group(8)))
except FileNotFoundError:
    print(f"錯誤: 找不到檔案 '{log_file_path}'。請檢查路徑是否正確。")
    exit()

# 檢查是否成功解析到數據
if not data['steps']:
    print("錯誤: 在日誌檔案中沒有找到任何匹配的訓練數據。請檢查日誌格式是否正確。")
    exit()

print(f"成功解析 {len(data['steps'])} 筆訓練數據。")

# --- 3. 繪製圖表 ---
print("正在繪製圖表...")

# 創建一個 3x2 的圖表網格
fig, axs = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('YOLOv4 Training Analysis', fontsize=20)

# 子圖 1: Total Loss vs. Steps (包含 Learning Rate)
ax1 = axs[0, 0]
color = 'tab:blue'
ax1.plot(data['steps'], data['total_loss'], color=color, label='Total Loss')
ax1.set_title('Total Loss & Learning Rate vs. Steps')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# 創建第二個 y 軸來顯示學習率
ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(data['steps'], data['lr'], color=color, linestyle='--', label='Learning Rate')
ax2.set_ylabel('Learning Rate', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log') # 學習率變化範圍大，使用對數座標

# 子圖 2: BBox Loss (xy, wh) vs. Steps
axs[0, 1].plot(data['steps'], data['loss_xy'], label='BBox Loss (xy)')
axs[0, 1].plot(data['steps'], data['loss_wh'], label='BBox Loss (wh)')
axs[0, 1].set_title('Bounding Box Loss Components vs. Steps')
axs[0, 1].set_xlabel('Steps')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle='--', alpha=0.6)

# 子圖 3: Objectness Loss vs. Steps
axs[1, 0].plot(data['steps'], data['loss_obj'], label='Objectness Loss', color='tab:green')
axs[1, 0].set_title('Objectness Loss vs. Steps')
axs[1, 0].set_xlabel('Steps')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle='--', alpha=0.6)

# 子圖 4: Classification Loss vs. Steps
axs[1, 1].plot(data['steps'], data['loss_cls'], label='Classification Loss', color='tab:purple')
axs[1, 1].set_title('Classification Loss vs. Steps')
axs[1, 1].set_xlabel('Steps')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle='--', alpha=0.6)

# 子圖 5: L2 Regularization Loss vs. Steps
axs[2, 0].plot(data['steps'], data['loss_l2'], label='L2 Loss', color='tab:orange')
axs[2, 0].set_title('L2 Regularization Loss vs. Steps')
axs[2, 0].set_xlabel('Steps')
axs[2, 0].set_ylabel('Loss')
axs[2, 0].legend()
axs[2, 0].grid(True, linestyle='--', alpha=0.6)

# 隱藏最後一個空的子圖
axs[2, 1].axis('off')

# 調整子圖之間的間距
plt.tight_layout(rect=[0, 0, 1, 0.96]) # 調整佈局以容納主標題

# --- 4. 儲存圖片 ---
try:
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"圖表已成功儲存至: {os.path.abspath(output_image_path)}")
except Exception as e:
    print(f"儲存圖片時發生錯誤: {e}")