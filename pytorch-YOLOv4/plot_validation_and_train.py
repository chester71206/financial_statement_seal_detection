import re
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def parse_train_log(log_file_path):
    """從訓練日誌檔案中解析 'step' 和 'loss'。"""
    print(f"--- 開始解析訓練日誌: {log_file_path} ---")
    steps = []
    losses = []
    train_loss_pattern = re.compile(r"Train step_(\d+): loss : ([\d\.]+)")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = train_loss_pattern.search(line)
                if match:
                    steps.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"錯誤：找不到訓練日誌檔案 '{log_file_path}'。")
        return [], []
    
    print(f"成功解析 {len(steps)} 筆訓練數據。")
    return steps, losses

def parse_validation_log(log_file_path):
    """從驗證日誌檔案中解析 mAP 和 AR 等指標。"""
    print(f"--- 開始解析驗證日誌: {log_file_path} ---")
    target_metrics = {
        'mAP @ .50:.95': r" Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \]",
        'mAP @ .50':      r" Average Precision  \(AP\) @\[ IoU=0.50      \| area=   all \| maxDets=100 \]",
        'mAP @ .75':      r" Average Precision  \(AP\) @\[ IoU=0.75      \| area=   all \| maxDets=100 \]",
        'AR @ maxDets=100': r" Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \]"
    }
    
    results = defaultdict(list)
    epochs = []
    current_epoch = None
    processed_epochs = set()
    epoch_pattern = re.compile(r'Epoch (\d+)/\d+:')

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    epoch_num = int(epoch_match.group(1))
                    if epoch_num not in processed_epochs:
                        current_epoch = epoch_num

                if current_epoch is not None:
                    for name, pattern in target_metrics.items():
                        if re.search(pattern, line):
                            try:
                                value = float(line.split('=')[-1].strip())
                                results[name].append(value)
                                
                                if name == list(target_metrics.keys())[0]:
                                    epochs.append(current_epoch)

                                all_found = all(len(results[key]) == len(epochs) for key in target_metrics)
                                if all_found:
                                    processed_epochs.add(current_epoch)
                                    current_epoch = None
                                    break
                            except (ValueError, IndexError):
                                continue
    except FileNotFoundError:
        print(f"錯誤：找不到驗證日誌檔案 '{log_file_path}'。")
        return [], defaultdict(list)
        
    print(f"成功解析 {len(epochs)} 個 epoch 的驗證數據。")
    return epochs, results

def plot_combined_results(train_log_path, validation_log_path, output_image_path):
    """
    協調解析和繪圖過程的主函式。
    """
    # 1. 分別解析兩個日誌檔案
    train_steps, train_losses = parse_train_log(train_log_path)
    val_epochs, val_results = parse_validation_log(validation_log_path)

    # 檢查是否有任何數據可供繪製
    if not train_steps and not val_epochs:
        print("兩個日誌檔案中都沒有找到可供繪製的數據。程式終止。")
        return

    # 2. 繪製圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
    fig.suptitle('Training & Validation Analysis from Separate Logs', fontsize=16)

    # 左圖：訓練損失 (Training Loss)
    if train_steps:
        ax1.plot(train_steps, train_losses, marker='.', linestyle='-', linewidth=1, markersize=0.5, color='tab:blue')
        ax1.set_title('Training Loss over Steps')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
    else:
        ax1.text(0.5, 0.5, 'No Training Loss Data Found', ha='center', va='center', fontsize=12)
        ax1.set_title('Training Loss')

    # 右圖：驗證指標 (Validation Metrics)
    if val_epochs:
        for name, values in val_results.items():
            if len(values) == len(val_epochs):
                ax2.plot(val_epochs, values, marker='o', linestyle='-', markersize=0.5, label=name)
        ax2.set_title('Validation Metrics over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'No Validation Metrics Data Found', ha='center', va='center', fontsize=12)
        ax2.set_title('Validation Metrics')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 3. 儲存圖表到檔案
    try:
        fig.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"\n圖表已成功儲存至: {os.path.abspath(output_image_path)}")
    except Exception as e:
        print(f"儲存圖片時發生錯誤: {e}")
    finally:
        # 關閉畫布，釋放記憶體，並確保不顯示
        plt.close(fig)

# --- 主程式執行區塊 ---
if __name__ == "__main__":
    # *** 請修改以下三個路徑 ***

    # 1. 您的訓練日誌檔案路徑
    train_log_file = "./log/log_2025-08-08_17-13-54.txt"
    # 範例: train_log_file = './log/log_2025-08-08_17-13-54.txt'
    
    # 2. 您的驗證日誌檔案路徑
    validation_log_file = './yolo_v4_v2.log'
    # 範例: validation_log_file = '/home/chester/pytorch-YOLOv4/yolo_v4_v2.log'

    # 3. 您希望儲存的圖片檔案名稱
    output_image_file = './training_validation_analysis.png'
    
    # 呼叫主函式執行所有操作
    plot_combined_results(train_log_file, validation_log_file, output_image_file)