# predict_custom.py
import torch
import cv2
import numpy as np
import argparse
import os
from models import Yolov4  # 從我們訓練時用的 models.py 導入 Yolov4 類
from cfg import Cfg         # 導入設定，特別是圖片尺寸

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    在圖片上繪製一個邊界框。
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # 線條粗細
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # 字體粗細
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 填充
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main(args):
    # --- 1. 設定 & 環境 ---
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != '-1' else 'cpu')
    print(f"使用裝置: {device}")

    # --- 2. 載入模型 ---
    print("正在載入模型...")
    # 使用與訓練時完全相同的 Yolov4 類來建立模型
    # n_classes 必須與你訓練時的設定一致
    # inference=True 會啟用模型的內部後處理，直接輸出預測框
    model = Yolov4(yolov4conv137weight=None, n_classes=args.num_classes, inference=True)
    
    # 載入你訓練好的權重
    try:
        state_dict = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("權重載入成功！")
    except Exception as e:
        print(f"錯誤：無法載入權重檔案。請檢查路徑和模型結構是否匹配。")
        print(e)
        return

    model.to(device)
    model.eval()  # 設定為評估模式

    # --- 3. 讀取與預處理圖片 ---
    print(f"正在讀取圖片: {args.image_path}")
    original_img = cv2.imdecode(np.fromfile(args.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original_img is None:
        print(f"錯誤：無法讀取圖片 {args.image_path}")
        return
    
    # 獲取模型輸入尺寸 (從 cfg.py)
    input_size = (Cfg.w, Cfg.h)
    
    # 預處理圖片
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, input_size)
    img_tensor = torch.from_numpy(img_resized).to(device).float()
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # 增加一個 batch 維度

    # --- 4. 執行預測 ---
    print("正在進行預測...")
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # --- 5. 後處理與繪製結果 ---
    # outputs 是一個列表，包含了 [boxes, scores]
    # boxes 的 shape 是 [N, 1, 4], scores 的 shape 是 [N, num_classes]
    # N 是檢測到的物件數量
    
    if outputs[0] is None or len(outputs[0]) == 0:
        print("沒有檢測到任何物件。")
        result_img = original_img
    else:
        # # 將預測結果從 Tensor 轉換為 numpy 陣列
        # all_boxes = outputs[0][0].cpu().numpy() # [N, 4], 格式是 [x1, y1, x2, y2]
        # all_scores = outputs[1][0].cpu().numpy() # [N, num_classes]

        all_boxes = outputs[0][0].squeeze().cpu().numpy() # 現在 shape 會是 [N, 4]
        all_scores = outputs[1][0].squeeze().cpu().numpy() # 現在 shape 會是 [N, num_classes]
    
        # 如果只有一個檢測結果，squeeze 後可能會變成一維，需要再加一個維度回來
        if all_boxes.ndim == 1:
            all_boxes = np.expand_dims(all_boxes, axis=0)
            all_scores = np.expand_dims(all_scores, axis=0)
        
        # 載入類別名稱
        class_names = [line.strip() for line in open(args.class_names)]

        # 獲取原始圖片尺寸，用於將預測框縮放回去
        h, w, _ = original_img.shape
        
        result_img = original_img.copy()

        print(f"檢測到 {len(all_boxes)} 個物件：")
        for i, box in enumerate(all_boxes):
            # # 獲取分數和類別 ID
            # class_id = np.argmax(all_scores[i])
            # score = all_scores[i][class_id]
            if np.isscalar(all_scores[i]):
                # 如果是純量，class_id 固定為 0，score 就是這個純量本身
                class_id = 0
                score = all_scores[i]
            else:
                # 如果不是純量 (多類別情況)，則使用原始的 argmax 邏輯
                class_id = np.argmax(all_scores[i])
                score = all_scores[i][class_id]

            # 過濾掉低於閾值的預測
            if score < args.conf_thresh:
                continue

            # 將預測框座標從模型輸入尺寸 (e.g., 640x640) 縮放回原始圖片尺寸
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            
            label = f'{class_names[class_id]}: {score:.2f}'
            print(f"  - {label} at [{x1}, {y1}, {x2}, {y2}]")
            
            # 繪製邊界框
            plot_one_box([x1, y1, x2, y2], result_img, label=label, color=(0, 255, 0))

    # --- 6. 顯示與儲存結果 ---
    output_filename = os.path.join(os.path.dirname(args.image_path), f"result_{os.path.basename(args.image_path)}")
    #cv2.imwrite(output_filename, result_img)
    cv2.imwrite("./output.jpg", result_img)
    #print(f"\n預測結果已儲存至: {output_filename}")

    # 如果你想直接顯示結果，可以取消下面兩行的註解 (在伺服器上可能無法使用)
    # cv2.imshow('Prediction', result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default='/home/chester/pytorch-YOLOv4/checkpoints/Yolov4_epoch520.pth', help='你訓練好的權重路徑')
    parser.add_argument('--image_path', type=str, default='/home/chester/pytorch-YOLOv4/全部公司四大報表_灰階_JPG/上銀_2024_合併_綜合損益表_3.jpg', help='要預測的圖片路徑')
    parser.add_argument('--num_classes', type=int, default=1, help='類別數量，必須與訓練時一致')
    parser.add_argument('--class_names', type=str, default='data/custom.names', help='類別名稱檔案路徑')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='置信度閾值')
    parser.add_argument('--gpu', type=str, default='0', help='指定 GPU，-1 代表使用 CPU')
    
    args = parser.parse_args()
    main(args)