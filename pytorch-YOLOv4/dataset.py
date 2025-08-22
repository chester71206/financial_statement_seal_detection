# -*- coding: utf-8 -*-
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

# --- 輔助函式 ---
def rand_uniform_strong(min_val, max_val):
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    return random.random() * (max_val - min_val) + min_val

def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    return scale if random.randint(0, 1) % 2 else 1. / scale

def rect_intersection(a, b):
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])
    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]

def get_image_id(filename: str) -> int:
    """
    從檔名生成一個唯一的整數 ID。
    """
    try:
        # 優先嘗試將檔名（不含副檔名）直接轉為整數
        return int(os.path.splitext(os.path.basename(filename))[0])
    except ValueError:
        # 如果檔名不是純數字，則使用 hash 生成一個唯一的 ID
        import hashlib
        return int(hashlib.md5(filename.encode()).hexdigest(), 16) % (10**6)

# --- 資料增強與標籤處理函式 (加固版) ---
def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
    if bboxes.shape[0] == 0:
        return np.zeros((num_boxes, 5), dtype=np.float32), 10000

    np.random.shuffle(bboxes)
    bboxes[:, 0:3:2] -= dx
    bboxes[:, 1:4:2] -= dy
    
    bboxes[:, 0:3:2] = np.clip(bboxes[:, 0:3:2], 0, sx)
    bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, sy)

    bboxes_w = bboxes[:, 2] - bboxes[:, 0]
    bboxes_h = bboxes[:, 3] - bboxes[:, 1]
    bboxes = bboxes[(bboxes_w > 1) & (bboxes_h > 1)]

    if bboxes.shape[0] == 0:
        return np.zeros((num_boxes, 5), dtype=np.float32), 10000

    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]
    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes]

    min_w_h = np.min([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]) if bboxes.shape[0] > 0 else 10000

    bboxes[:, 0:3:2] *= (net_w / sx)
    bboxes[:, 1:4:2] *= (net_h / sy)

    if flip:
        temp = bboxes[:, 0].copy()
        bboxes[:, 0] = net_w - bboxes[:, 2]
        bboxes[:, 2] = net_w - temp
    
    truth = np.zeros((num_boxes, 5), dtype=np.float32)
    truth[:bboxes.shape[0], :] = bboxes
    
    return truth, min_w_h

def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur):
    try:
        img = mat
        oh, ow, _ = img.shape
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
        
        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]
        img_rect = [0, 0, ow, oh]
        new_src_rect = rect_intersection(src_rect, img_rect)

        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
        
        if new_src_rect[2] - new_src_rect[0] <= 0 or new_src_rect[3] - new_src_rect[1] <= 0:
            return cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        
        cropped = np.full((sheight, swidth, 3), 128, dtype=img.dtype)
        cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
            img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

        sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)

        if flip:
            sized = cv2.flip(sized, 1)

        if dsat != 1 or dexp != 1 or dhue != 0:
            hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)
            hsv = list(cv2.split(hsv_src))
            hsv[1] *= dsat
            hsv[2] *= dexp
            hsv[0] = (hsv[0] + 179 * dhue) % 180
            hsv_src = cv2.merge(hsv)
            sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255).astype(np.uint8)

        if blur:
            ksize_val = (int(blur) // 2) * 2 + 1
            if ksize_val > 1:
                sized = cv2.GaussianBlur(sized, (ksize_val, ksize_val), 0)

        if gaussian_noise:
            noise = np.random.normal(0, gaussian_noise, sized.shape).astype(np.int16)
            sized = np.clip(sized.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        return sized
    except Exception as e:
        print(f"Warning: Data augmentation failed: {e}. Resizing original image.")
        return cv2.resize(mat, (w, h), cv2.INTER_LINEAR)

# --- 核心 Dataset 類別 ---
class Yolo_dataset(Dataset):
    def __init__(self, label_path, cfg, train=True):
        super(Yolo_dataset, self).__init__()
        self.cfg = cfg
        self.train = train
        
        with open(label_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 為了相容驗證部分的 coco_utils.py，我們也創建 self.imgs
        self.imgs = self.lines
        
        # 創建一個 lazy loader 來獲取標籤，避免一次性將所有標籤讀入記憶體
        self.annotations = self._load_annotations()
        
        print(f"Dataset in {'Train' if train else 'Validation'} mode initialized. Found {len(self.lines)} images.")

    def __len__(self):
        return len(self.lines)

    def _load_annotations(self):
        """Lazy annotation loader."""
        cache = {}
        def get_annotation(img_path):
            if img_path in cache:
                return cache[img_path]
            
            label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                bboxes = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
                cache[img_path] = bboxes
                return bboxes
            else:
                return np.zeros((0, 5), dtype=np.float32)
        return get_annotation

    def __getitem__(self, index):
        if self.train:
            return self._get_train_item(index)
        else:
            return self._get_val_item(index)

    def _get_train_item(self, index):
        img_path = self.lines[index]
        bboxes = self.annotations(img_path).copy()

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Failed to load train image: {img_path}. Using a gray placeholder.")
            img = np.full((self.cfg.h, self.cfg.w, 3), 128, dtype=np.uint8)
            bboxes = np.zeros((0, 5), dtype=np.float32)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        oh, ow, _ = img.shape
        dh, dw, _ = np.array(np.array([oh, ow, 0]) * self.cfg.jitter, dtype=int)
        
        pleft, pright = random.randint(-dw, dw), random.randint(-dw, dw)
        ptop, pbot = random.randint(-dh, dh), random.randint(-dh, dh)
        
        swidth, sheight = ow - pleft - pright, oh - ptop - pbot
        
        flip = random.randint(0, 1) if self.cfg.flip else 0
        dhue, dsat, dexp = rand_uniform_strong(-self.cfg.hue, self.cfg.hue), rand_scale(self.cfg.saturation), rand_scale(self.cfg.exposure)
        
        truth, _ = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, flip, pleft, ptop, swidth, sheight, self.cfg.w, self.cfg.h)
        
        augmented_img = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, 0, 0)
        
        return augmented_img, truth

    def _get_val_item(self, index):
        """
        為驗證集生成正確的 COCO 格式 target。
        """
        img_path = self.lines[index]
        target = {}

        # 讀取圖片
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Failed to load validation image: {img_path}. Using empty placeholder.")
            img = np.zeros((self.cfg.h, self.cfg.w, 3), dtype=np.uint8)
            bboxes_xyxyc = np.zeros((0, 5), dtype=np.float32)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 讀取標籤 [x_min, y_min, x_max, y_max, class_id]
            bboxes_xyxyc = self.annotations(img_path).copy()

        num_objs = len(bboxes_xyxyc)
        
        if num_objs > 0:
            boxes_xyxy = bboxes_xyxyc[:, :4]
            labels = bboxes_xyxyc[:, 4]

            # 將 boxes 從 [x1, y1, x2, y2] 轉換為 COCO 格式 [x1, y1, w, h]
            boxes_xywh = boxes_xyxy.copy()
            boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # width
            boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # height

            target['boxes'] = torch.as_tensor(boxes_xywh, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['area'] = (target['boxes'][:, 3]) * (target['boxes'][:, 2])
        else:
            # 處理沒有物件的圖片
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
            target['area'] = torch.zeros(0, dtype=torch.float32)

        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        
        # 注意：驗證時，圖片的 resize 等操作在 train.py 的 evaluate 函式中完成
        # 這裡只需要回傳原始圖片和正確格式的 target
        return img, target