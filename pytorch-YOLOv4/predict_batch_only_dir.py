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

    # <<< --- 3. 準備圖片列表和輸出資料夾 (此處為主要修改點) --- >>>
    print(f"正在從資料夾 {args.input_dir} 讀取圖片...")
    
    # 定義支援的圖片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    # 使用 os.listdir() 掃描資料夾，並用 os.path.join() 組成完整路徑
    # 只將符合格式的檔案加入列表
    image_paths = [os.path.join(args.input_dir, fname) 
                   for fname in os.listdir(args.input_dir) 
                   if fname.lower().endswith(supported_formats)]

    if not image_paths:
        print(f"錯誤：在資料夾 '{args.input_dir}' 中找不到任何支援的圖片格式 {supported_formats}。")
        return

    print(f"找到 {len(image_paths)} 張圖片準備進行處理。")
    # <<< --- 修改結束 --- >>>

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"結果將儲存至: {args.output_dir}")

    class_names = [line.strip() for line in open(args.class_names)]

    # --- 進入迴圈，處理每一張圖片 ---
    for image_path in tqdm(image_paths, desc="Processing images"):
        original_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if original_img is None:
            print(f"警告：無法讀取圖片 {image_path}，跳過。")
            continue
        
        h, w, _ = original_img.shape
        input_size = (Cfg.w, Cfg.h)
        
        img_resized = cv2.resize(original_img, input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).to(device).float().div(255.0).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
        
        boxes = post_processing(img_tensor, args.conf_thresh, args.nms_thresh, outputs)
        
        if len(boxes) > 0:
            final_boxes = boxes[0]
        else:
            final_boxes = []

        result_img = original_img.copy()

        if len(final_boxes) > 0:
            # 為了避免在處理大量圖片時洗版，可以選擇性地關閉詳細輸出
            # print(f"\n圖片: {os.path.basename(image_path)} - 檢測到 {len(final_boxes)} 個物件：")
            for box in final_boxes:
                x1 = int(box[0] * w)
                y1 = int(box[1] * h)
                x2 = int(box[2] * w)
                y2 = int(box[3] * h)
                
                score = box[5]
                class_id = int(box[6])
                
                label = f'{class_names[class_id]}: {score:.2f}'
                # print(f"  - {label} at [{x1}, {y1}, {x2}, {y2}]")
                
                plot_one_box([x1, y1, x2, y2], result_img, label=label, color=(0, 255, 0))

        base_filename = os.path.basename(image_path)
        output_filename = os.path.join(args.output_dir, base_filename)
        cv2.imwrite(output_filename, result_img)

    print("\n所有圖片處理完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # <<< --- 修改命令列參數 --- >>>
    # 刪除 --val_txt_path，新增 --input_dir
    parser.add_argument('--input_dir', type=str, default='/home/chester/pytorch-YOLOv4/全部公司四大報表_灰階_JPG_add_noise_hard', help='包含待預測圖片的資料夾路徑')
    # <<< --- 修改結束 --- >>>
    
    parser.add_argument('--output_dir', type=str, default='./output_test_best_hard', help='儲存預測結果的資料夾')
    parser.add_argument('--weights_path', type=str, default='/home/chester/pytorch-YOLOv4/checkpoints/Yolov4_best.pth', help='你訓練好的權重路徑')
    parser.add_argument('--num_classes', type=int, default=1, help='類別數量，必須與訓練時一致')
    parser.add_argument('--class_names', type=str, default='data/custom.names', help='類別名稱檔案路徑')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='置信度閾值')
    parser.add_argument('--nms_thresh', type=float, default=0.4, help='NMS 的 IoU 閾值')
    parser.add_argument('--gpu', type=str, default='0', help='指定 GPU，-1 代表使用 CPU')
    
    args = parser.parse_args()
    main(args)