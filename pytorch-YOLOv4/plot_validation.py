import re
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def plot_multiple_metrics_from_log(log_file_path, output_image_path):
    """
    從 YOLOv4 的 log 檔案中解析多個評估指標，並將圖表儲存為圖片。
    """
    # 1. 定義我們想要追蹤的指標
    # 使用簡化的名稱作為字典的 key，正則表達式的一部分作為 value
    target_metrics = {
        'mAP @ .50:.95': r" Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \]",
        'mAP @ .50':      r" Average Precision  \(AP\) @\[ IoU=0.50      \| area=   all \| maxDets=100 \]",
        'mAP @ .75':      r" Average Precision  \(AP\) @\[ IoU=0.75      \| area=   all \| maxDets=100 \]",
        'AR @ maxDets=100': r" Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \]"
    }
    
    # 使用 defaultdict 來輕鬆地儲存每個指標的數值列表
    results = defaultdict(list)
    epochs = []
    
    current_epoch = None
    processed_epochs = set() # 用來防止同一個 epoch 的指標被重複記錄

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 尋找 Epoch 行
                epoch_match = re.search(r'Epoch (\d+)/\d+:', line)
                if epoch_match:
                    epoch_num = int(epoch_match.group(1))
                    # 只有當這是一個新的、未處理過的 validation epoch 時才更新
                    if epoch_num not in processed_epochs:
                        current_epoch = epoch_num

                # 如果我們在一個 validation epoch 區塊內，就尋找目標指標
                if current_epoch is not None:
                    for name, pattern in target_metrics.items():
                        if re.search(pattern, line):
                            try:
                                value_str = line.split('=')[-1].strip()
                                value = float(value_str)
                                results[name].append(value)
                                
                                # 當第一個指標被找到時，記錄 epoch
                                if name == list(target_metrics.keys())[0]:
                                    epochs.append(current_epoch)

                                # 檢查是否所有指標都已為此 epoch 找到
                                all_found = all(len(results[key]) == len(epochs) for key in target_metrics)
                                if all_found:
                                    processed_epochs.add(current_epoch)
                                    current_epoch = None # 重置，等待下一個 validation epoch

                            except (ValueError, IndexError):
                                continue # 解析失敗則跳過

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{log_file_path}'。")
        return
    
    # 檢查是否有數據
    if not epochs or not results:
        print("在 log 檔案中找不到任何可供繪製的數據。")
        return

    # 2. 繪製圖表
    plt.figure(figsize=(14, 8))
    
    for name, values in results.items():
        # 確保每個指標的數據點數量與 epochs 數量一致
        if len(values) == len(epochs):
            plt.plot(epochs, values, marker='o', linestyle='-', label=name,markersize=1)

    plt.title('YOLOv4 Validation Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend() # 顯示圖例
    plt.grid(True)
    plt.tight_layout()

    # 3. 儲存圖表
    try:
        plt.savefig(output_image_path, dpi=300)
        print(f"圖表已成功儲存至: {os.path.abspath(output_image_path)}")
    except Exception as e:
        print(f"儲存圖片時發生錯誤: {e}")
    
    plt.close()

# --- 主程式 ---
if __name__ == "__main__":
    log_path = '/home/chester/pytorch-YOLOv4/yolo_v4_v2.log'
    output_path = './validation_metrics_comparison.jpg'
    
    plot_multiple_metrics_from_log(log_path, output_path)