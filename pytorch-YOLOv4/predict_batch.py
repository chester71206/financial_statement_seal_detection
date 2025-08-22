# predict_batch.py
import torch
import cv2
import numpy as np
import argparse
import os
import random
from tqdm import tqdm
from models import Yolov4
from cfg import Cfg

# --- <<< 新增導入 >>> ---
# 從 tool.utils 導入後處理函式，這是解決 NMS 問題的關鍵
from tool.utils import post_processing 

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    在圖片上繪製一個邊界框。
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main(args):
    # --- 1. 設定 & 環境 (迴圈外) ---
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != '-1' else 'cpu')
    print(f"使用裝置: {device}")

    # --- 2. 載入模型 (迴圈外) ---
    print("正在載入模型...")
    # 確保 inference=True，這樣模型內部才會調用 get_region_boxes
    model = Yolov4(yolov4conv137weight=None, n_classes=args.num_classes, inference=True) 
    
    try:
        state_dict = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("權重載入成功！")
    except Exception as e:
        print(f"錯誤：無法載入權重檔案。請檢查路徑和模型結構是否匹配。")
        print(e)
        return

    model.to(device)
    model.eval()

    # --- 3. 準備圖片列表和輸出資料夾 (迴圈外) ---
    with open(args.val_txt_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    print(f"從 {args.val_txt_path} 讀取到 {len(image_paths)} 張圖片。")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"結果將儲存至: {args.output_dir}")

    class_names = [line.strip() for line in open(args.class_names)]
    import time
    # --- 進入迴圈，處理每一張圖片 ---
    for image_path in tqdm(image_paths, desc="Processing images"):
        start=time.time()
        original_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if original_img is None:
            print(f"警告：無法讀取圖片 {image_path}，跳過。")
            continue
        
        h, w, _ = original_img.shape
        input_size = (Cfg.w, Cfg.h)
        
        # 預處理圖片
        img_resized = cv2.resize(original_img, input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # 轉為 RGB
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).to(device).float().div(255.0).unsqueeze(0)

        # --- 4. 執行預測 ---
        with torch.no_grad():
            # model(img_tensor) 在 inference=True 模式下會返回 get_region_boxes 的結果
            outputs = model(img_tensor)
        
        # --- 5. 後處理與繪製結果 (使用 post_processing 函式) ---
        # <<< 核心修改部分開始 >>>
        
        # 使用專案自帶的 post_processing 函式，它內部包含了 NMS 邏輯
        # 我們將從命令列傳入的 conf_thresh 和 nms_thresh 傳給它
        boxes = post_processing(img_tensor, args.conf_thresh, args.nms_thresh, outputs)
        print("boxes:",boxes)
        # post_processing 返回一個批次的結果，我們是單張圖預測，所以取第一個元素
        if len(boxes) > 0:
            final_boxes = boxes[0]
        else:
            final_boxes = []

        result_img = original_img.copy()

        if len(final_boxes) > 0:
            print(f"\n圖片: {os.path.basename(image_path)} - 檢測到 {len(final_boxes)} 個物件：")
            # `final_boxes` 已經是經過 NMS 篩選後的結果
            # 每個 box 的格式是: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
            for box in final_boxes:
                # 座標是 0-1 之間的比例，需要乘以原始圖片的寬高
                x1 = int(box[0] * w)
                y1 = int(box[1] * h)
                x2 = int(box[2] * w)
                y2 = int(box[3] * h)
                
                score = box[5]  # 類別的置信度
                class_id = int(box[6])
                
                label = f'{class_names[class_id]}: {score:.2f}'
                print(f"  - {label} at [{x1}, {y1}, {x2}, {y2}]")
                
                plot_one_box([x1, y1, x2, y2], result_img, label=label, color=(0, 255, 0))
        # <<< 核心修改部分結束 >>>

        # --- 6. 儲存結果 ---
        base_filename = os.path.basename(image_path)
        output_filename = os.path.join(args.output_dir, base_filename)
        cv2.imwrite(output_filename, result_img)

        end=time.time()
        print("time:",end-start)

    print("\n所有圖片處理完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--val_txt_path', type=str, default='/home/chester/pytorch-YOLOv4/印章辨識有問題.txt', help='包含圖片路徑的 txt 檔案')    
    # parser.add_argument('--output_dir', type=str, default='./output_val_原先cv不行的_add_mid_noise', help='儲存預測結果的資料夾')

    parser.add_argument('--val_txt_path', type=str, default='/home/chester/pytorch-YOLOv4/印章辨識有問題.txt', help='包含圖片路徑的 txt 檔案')    
    parser.add_argument('--output_dir', type=str, default='./output_test_best_原先CV不行', help='儲存預測結果的資料夾')


    # parser.add_argument('--val_txt_path', type=str, default='/home/chester/pytorch-YOLOv4/val_加躁.txt', help='包含圖片路徑的 txt 檔案')
    # parser.add_argument('--output_dir', type=str, default='./output_val_add_mid_noise', help='儲存預測結果的資料夾')
    
    parser.add_argument('--weights_path', type=str, default='/home/chester/pytorch-YOLOv4/yolov4_model/Yolov4_epoch1460.pth', help='你訓練好的權重路徑')
    parser.add_argument('--num_classes', type=int, default=1, help='類別數量，必須與訓練時一致')
    parser.add_argument('--class_names', type=str, default='data/custom.names', help='類別名稱檔案路徑')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='置信度閾值 (建議設高一點, e.g., 0.5)')
    
    # --- <<< 新增 NMS 參數 >>> ---
    parser.add_argument('--nms_thresh', type=float, default=0.4, help='NMS 的 IoU 閾值 (建議設 0.4 左右)')
    
    parser.add_argument('--gpu', type=str, default='0', help='指定 GPU，-1 代表使用 CPU')
    
    args = parser.parse_args()
    main(args)


