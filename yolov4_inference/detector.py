import torch
import cv2
import numpy as np
import os
from models import Yolov4
from cfg import Cfg
from tool.utils import post_processing

class StampDetector:
    """
    YOLOv4 印章檢測器。
    
    這個類別封裝了模型的載入和預測邏輯，使其易於在其他程式中重複使用。
    """
    def __init__(self, 
                 weights_path='./Yolov4_epoch1460.pth', 
                 num_classes=1, 
                 class_names_path='data/custom.names', 
                 conf_thresh=0.5, 
                 nms_thresh=0.4,  # NMS 閾值也設個預設值
                 gpu_id='0'):
        """
        初始化檢測器，載入模型和設定。這部分只會在建立物件時執行一次。
        
        Args:
            weights_path (str): 訓練好的權重檔案路徑。
            num_classes (int): 類別數量。
            class_names_path (str): 類別名稱檔案路徑。
            conf_thresh (float): 置信度閾值。
            nms_thresh (float): NMS 的 IoU 閾值。
            gpu_id (str): 使用的 GPU ID，'-1' 表示使用 CPU。
        """
        # 1. 設定裝置
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id != '-1' else 'cpu')
        print(f"StampDetector 使用裝置: {self.device}")

        # 2. 載入模型
        print("正在載入模型...")
        self.model = Yolov4(yolov4conv137weight=None, n_classes=num_classes, inference=True)
        
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("權重載入成功！")
        except Exception as e:
            print(f"錯誤：無法載入權重檔案 {weights_path}。")
            raise e

        self.model.to(self.device)
        self.model.eval()

        # 3. 儲存設定
        self.input_size = (Cfg.w, Cfg.h)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.class_names = [line.strip() for line in open(class_names_path)]
        print("檢測器初始化完成。")

    def get_stamp_positions(self, image_bgr):
        """
        對單張 BGR 格式的圖片 (來自 cv2.imread) 進行印章位置預測。
        
        Args:
            image_bgr (np.ndarray): 使用 cv2 讀取的 BGR 格式圖片。

        Returns:
            list: 一個包含所有檢測到的物件的列表。
                  每個物件是一個字典，格式為：
                  {'box': [x1, y1, x2, y2], 'score': float, 'class_name': str}
                  如果沒有檢測到任何物件，則返回空列表 []。
        """
        if image_bgr is None:
            print("警告：輸入的圖片為 None。")
            return []

        # 1. 圖片前處理
        original_h, original_w, _ = image_bgr.shape
        img_resized = cv2.resize(image_bgr, self.input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).to(self.device).float().div(255.0).unsqueeze(0)

        # 2. 模型推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # 3. 後處理
        boxes = post_processing(img_tensor, self.conf_thresh, self.nms_thresh, outputs)
        
        # 4. 整理並返回結果
        detected_objects = []
        if len(boxes) > 0 and boxes[0] is not None:
            # 每個 box 的格式是: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
            final_boxes = boxes[0]
            for box in final_boxes:
                x1 = int(box[0] * original_w)
                y1 = int(box[1] * original_h)
                x2 = int(box[2] * original_w)
                y2 = int(box[3] * original_h)
                
                obj_confidence = box[4]
                class_conf = box[5]
                class_id = int(box[6])
                class_name = self.class_names[class_id]

                detected_objects.append({
                    'box': [x1, y1, x2, y2],
                    'obj_confidence': obj_confidence.item(), # .item() 轉為純 python float
                    'class_conf': class_conf.item(), 
                    'class_name': class_name
                })
        
        return detected_objects