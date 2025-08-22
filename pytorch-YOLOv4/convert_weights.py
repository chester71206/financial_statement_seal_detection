import torch
from tool.darknet2pytorch import Darknet

if __name__ == '__main__':
    # --- 設定 ---
    cfg_file = 'cfg/yolov4.cfg'
    weights_file = 'yolov4.weights'
    output_file = 'yolov4.pth'
    # --- 設定結束 ---

    print(f"正在使用設定檔 '{cfg_file}' 建立模型結構...")
    model = Darknet(cfg_file)

    print(f"正在從 '{weights_file}' 載入 Darknet 權重...")
    model.load_weights(weights_file)

    print(f"權重載入成功，正在將模型儲存為 PyTorch 格式到 '{output_file}'...")
    torch.save(model.state_dict(), output_file)

    print(f"--- 轉換成功！模型已儲存為 {output_file} ---")