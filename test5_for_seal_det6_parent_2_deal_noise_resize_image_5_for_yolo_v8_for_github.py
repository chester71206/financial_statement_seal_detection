import cv2
import numpy as np
import os
import random
from collections import Counter

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """
    取代 cv2.imread，使其支援包含非 ASCII 字元 (如中文) 的路徑。
    """
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, flags)
        return img
    except Exception as e:
        print(f"使用 imread_unicode 讀取檔案時發生錯誤: {e}")
        return None

def imwrite_unicode(path, img):
    """
    取代 cv2.imwrite，使其支援包含非 ASCII 字元 (如中文) 的路徑。
    """
    try:
        ext = os.path.splitext(path)[1]
        result, buffer = cv2.imencode(ext, img)
        if result:
            with open(path, mode='wb') as f:
                f.write(buffer)
            return True
        else:
            return False
    except Exception as e:
        print(f"使用 imwrite_unicode 寫入檔案時發生錯誤: {e}")
        return False


def detect_lines(image_thresh, kernel_shape, hough_params):
    """
    一個通用的線條偵測函式，使用形態學和霍夫變換。

    :param image_thresh: 輸入的二值化圖片 (前景為白色)。
    :param kernel_shape: 用於形態學操作的 Kernel 形狀, e.g., (20, 1) for horizontal。
    :param hough_params: 霍夫變換的參數字典。
    :return: 偵測到的線條陣列 (or None if no lines found)。
    """
    # 1. 根據傳入的形狀創建 Kernel
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    
    # 2. 形態學開運算，保留特定方向的線條
    detected_lines_img = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    
    # 3. (可選) 輕微膨脹，連接斷裂的線段
    dilate_kernel = np.ones((3, 3), np.uint8)
    detected_lines_img = cv2.dilate(detected_lines_img, dilate_kernel, iterations=1)
    
    # 4. 霍夫變換偵測直線
    lines = cv2.HoughLinesP(
        detected_lines_img, 
        1, 
        np.pi / 180,  
        threshold=hough_params['threshold'], 
        minLineLength=hough_params['minLineLength'], 
        maxLineGap=hough_params['maxLineGap']
    )
    
    return lines



def find_and_draw_stamp_boxes(input_path, output_path):
    # --- 1. 讀取與預處理 ---
    image_gray = imread_unicode(input_path, cv2.IMREAD_GRAYSCALE)
    imwrite_unicode("0-original_image.jpg", image_gray)
    
    output_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    image_gray = cv2.medianBlur(image_gray, 3)
    imwrite_unicode("1-medianBlur.jpg", image_gray)

    _, thresh_image = cv2.threshold(image_gray, 245, 255, cv2.THRESH_BINARY_INV)

    # (可選) 儲存二值化後的結果，方便觀察效果
    cv2.imwrite('2-medianBlur_threshold.png', thresh_image)

    kernel = np.ones((2, 2), np.uint8)
    closed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

    # (可選) 儲存關閉操作後的結果，方便觀察效果
    cv2.imwrite('3-output_closed.png', closed_image)


    # 步驟 3: 將顏色反轉回來，變回我們習慣的「白底黑字」
    # cv2.bitwise_not 會將 0 變成 255，255 變成 0。
    image_gray = cv2.bitwise_not(closed_image)

    imwrite_unicode("4-image_gray.jpg", image_gray)

    _, thresh = cv2.threshold(image_gray, 245, 255, cv2.THRESH_BINARY_INV)
    imwrite_unicode("5-thresh.jpg", thresh)
    # 創建一個彩色的輸出圖片，以便在其上繪製彩色的框

    hough_lines_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    # --- 2. 偵測所有線條 --3
    hough_config = {'threshold': 40, 'minLineLength': 40, 'maxLineGap': 10}
    table_line_hough_config = {'threshold': 50, 'minLineLength': 700, 'maxLineGap': 20}
    long_horizontal_lines = detect_lines(thresh, (50, 1), table_line_hough_config)
    long_vertical_lines = detect_lines(thresh, (1, 50), table_line_hough_config)

        # --- 3. 執行減法：從原始二值圖中移除表格線 ---

    # 創建一個新的影像副本，我們將在這個副本上操作，保留原始的 thresh
    image_no_lines = thresh.copy()

    # 在 image_no_lines 上將偵測到的長線條用黑色(0)畫掉
    # 增加線條寬度(e.g., 5)，確保能完全覆蓋原始線條及其邊緣
    line_thickness = 3 
    if long_horizontal_lines is not None:
        for line in long_horizontal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_no_lines, (x1, y1), (x2, y2), (0, 0, 0), line_thickness)

    if long_vertical_lines is not None:
        for line in long_vertical_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_no_lines, (x1, y1), (x2, y2), (0, 0, 0), line_thickness)

    # (強烈建議) 儲存這個中間結果，確認表格線是否被成功移除
    imwrite_unicode("6-DEBUG_IMAGE_NO_LINES.jpg", image_no_lines)

    # 使用新的設定來偵測長的水平和垂直線
    horizontal_lines = detect_lines(image_no_lines, (25, 1), hough_config)
    vertical_lines = detect_lines(image_no_lines, (1, 40), hough_config)

    #----------------------------
    if horizontal_lines is not None:
        for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if vertical_lines is not None: 
        for line in vertical_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2) 

    imwrite_unicode("7-hough_lines_image.jpg", hough_lines_image)
    #----------------------------



    # --- 3. 結構生成 ---
    mask = np.zeros_like(image_no_lines)
    if horizontal_lines is not None:
        for line in horizontal_lines:
            if abs(line[0][2]-line[0][0])<700:
                #print("horizontal_lines:",line)
                cv2.line(mask, (line[0][0], line[0][1]), (line[0][2], line[0][3]), 255, 3)
    if vertical_lines is not None:
        for line in vertical_lines:
            if abs(line[0][3]-line[0][1])<700:
                #print("vertical_lines:",line)
                cv2.line(mask, (line[0][0], line[0][1]), (line[0][2], line[0][3]), 255, 3)
    
    #imwrite_unicode("DEBUG_MASK.jpg", mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    imwrite_unicode("8-DEBUG_MASK.jpg", dilated_mask)
    # --- 4. 尋找輪廓和層級結構 ---
    #contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #---------------------------------------------------
    contour_visualization_image = cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR) 

    # 使用 cv2.drawContours 畫輪廓
    # 參數: 目標圖片, 輪廓列表, -1 (表示畫所有輪廓), 顏色(藍色), 線寬
    cv2.drawContours(contour_visualization_image, contours, -1, (255, 0, 0), 2)

    # 儲存這張畫滿了輪廓的圖片，方便你查看
    imwrite_unicode("9-DEBUG_ALL_CONTOURS.jpg", contour_visualization_image)

    hierarchy_visualization_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    #---------------------------------------------------    
    img_h, img_w = image_gray.shape[:2]
    # --- 5. 層級篩選 ---
    final_stamp_contours = []
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # 初步過濾
            area = cv2.contourArea(contour)
            if area < 3000: continue
            
            x, y, w, h = cv2.boundingRect(contour)
            # 長寬比過濾稍微放寬，以容納可能被拉長的輪廓
            aspect_ratio = float(w) / h if h > 0 else 0
            if not (0.25 < aspect_ratio < 4.0): continue
            
            # 層級判斷
            parent_index = hierarchy[0][i][3]
            if parent_index == -1:
                cv2.drawContours(hierarchy_visualization_image, [contour], -1, (0, 255, 0), 3)
                cv2.putText(hierarchy_visualization_image, "Parent", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                final_stamp_contours.append(contour)
            else:
                cv2.drawContours(hierarchy_visualization_image, [contour], -1, (0, 0, 255), 2)
                cv2.putText(hierarchy_visualization_image, "Child", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    imwrite_unicode("10-DEBUG_HIERARCHY_VISUALIZATION.jpg", hierarchy_visualization_image)

    #cv2.drawContours(output_image, final_stamp_contours, -1, (0, 255, 0), 3)
    
        # 遍歷所有被認定為印章的父輪廓
    # 遍歷所有被認定為印章的父輪廓
    # ==============================================================================
# == 最終步驟：基於上下最短邊，繪製矩形框 ==
# ==============================================================================

    img_total_height = output_image.shape[0]

    stamp_candidates = []
    for stamp_contour in final_stamp_contours:
        segments_info = get_top_bottom_segments(stamp_contour)

        # 只有能成功計算出精確邊界的，才被視為有效候選者
        if segments_info['top_segment'] and segments_info['bottom_segment']:
            top_seg = segments_info['top_segment']
            bottom_seg = segments_info['bottom_segment']

            if top_seg['length'] < bottom_seg['length']:
                base_seg = top_seg
            else:
                base_seg = bottom_seg

            # 您的精確矩形參數
            rect_y = top_seg['start'][1]
            rect_h = bottom_seg['start'][1] - rect_y
            rect_x = base_seg['start'][0]
            rect_w = base_seg['length']
            
            # 將計算結果和原始輪廓一起儲存
            if rect_w > 0 and rect_h > 0:
                stamp_candidates.append({
                    'min_y': rect_y,
                    'max_y': rect_y + rect_h,
                    'rect': (rect_x, rect_y, rect_w, rect_h) # 儲存精確的矩形資訊
                })

    print(f"共提取到 {len(stamp_candidates)} 個有效的候選印章。")

    # --- 步驟 2: 根據位置分類為頂部和底部 ---
    print("\n--- 步驟 2: 將候選印章分為頂部和底部 ---")
    img_total_height = output_image.shape[0]
    upper_bound_y = img_total_height / 7
    lower_bound_y = img_total_height * 4 / 5

    top_group = []
    bottom_candidates = [] # 這裡先收集所有底部候選者，稍後再群組化

    for stamp in stamp_candidates:
        if stamp['min_y'] < upper_bound_y:
            top_group.append(stamp)
        elif stamp['max_y'] > lower_bound_y:
            bottom_candidates.append(stamp)
        else:
            print(f"  -> 過濾掉中間區域的候選框: Y 範圍 ({stamp['min_y']}, {stamp['max_y']})")

    print(f"分類結果：頂部 {len(top_group)} 個，底部 {len(bottom_candidates)} 個。")

    # --- 步驟 3: 執行您指定的底部印章群組化演算法 ---
    print("\n--- 步驟 3: 對底部印章執行迭代式群組化 ---")
    final_bottom_group = []
    if bottom_candidates:
        # a. 根據 y_max 座標降序排序，最下面的印章會在第一個
        bottom_candidates.sort(key=lambda s: s['max_y'], reverse=True)
        
        # b. 將最下面的印章作為群組的起點
        cluster = [bottom_candidates.pop(0)]
        cluster_min_y = cluster[0]['min_y']
        cluster_max_y = cluster[0]['max_y']
        print(f"以最低的印章 (Y:{cluster_min_y}-{cluster_max_y}) 開始建立群組。")
        
        # c. 迭代循環，直到沒有新的印章可以被加入群組
        while True:
            added_in_this_pass = False
            # 建立一個臨時列表，存放這一輪要被加入的成員
            members_to_add = []
            
            for candidate in bottom_candidates:
                # 檢查候選印章的 Y 範圍是否與當前群組的 Y 範圍重疊
                # (您的演算法核心)
                if candidate['min_y'] < cluster_max_y and candidate['max_y'] > cluster_min_y:
                    members_to_add.append(candidate)
                    added_in_this_pass = True
            
            # 如果這一輪沒有找到新成員，則群組建立完成
            if not added_in_this_pass:
                print("沒有更多印章可加入，群組建立完成。")
                break
            
            # 如果找到了，就把它們加入群組並更新範圍
            for member in members_to_add:
                print(f"  -> 印章 (Y:{member['min_y']}-{member['max_y']}) 加入群組。")
                cluster.append(member)
                bottom_candidates.remove(member) # 從待選列表中移除
                # 更新群組的整體 Y 範圍
                cluster_min_y = min(cluster_min_y, member['min_y'])
                cluster_max_y = max(cluster_max_y, member['max_y'])
            print(f"  ==> 群組範圍更新為: Y({cluster_min_y}-{cluster_max_y})")

        final_bottom_group = cluster

    # --- 步驟 4: 組合最終結果並繪製 ---
    print("\n--- 步驟 4: 繪製最終確認的印章框 ---")
    final_stamps_to_draw = top_group + final_bottom_group
    print(f"總計將繪製 {len(final_stamps_to_draw)} 個印章。")

    padding = 10
    for stamp_data in final_stamps_to_draw:
        # 從我們儲存的資料中直接取出已經計算好的精確矩形
        x, y, w, h = stamp_data['rect']
        
        # 應用 Padding
        padded_x = max(0, x - padding)
        padded_y = max(0, y - padding)
        padded_w = w + (padding * 2)
        padded_h = h + (padding * 2)
        
        # 繪製最終的矩形
        cv2.rectangle(output_image, 
                      (padded_x, padded_y), 
                      (padded_x + padded_w, padded_y + padded_h), 
                      (0, 0, 255), 3)

    # 儲存最終結果
    imwrite_unicode(output_path, output_image)
    print(f"處理完成，已將過濾後的印章框出並儲存至 '{output_path}'")


    #新增的程式碼
    global current_label_path

    # 2. 檢查 final_stamps_to_draw 是否為空，如果不為空才繼續
    if 'final_stamps_to_draw' in locals() and final_stamps_to_draw:
        
        # 3. 準備一個列表來收集所有標註行
        yolo_labels = []
        
        # 4. 獲取圖片的總寬度和高度，用於歸一化
        img_h, img_w = output_image.shape[:2]
        dw = 1.0 / img_w
        dh = 1.0 / img_h
        padding = 10 # 確保這個 padding 和你繪圖時用的一樣
        
        # 5. 遍歷每一個找到的印章框
        for stamp_data in final_stamps_to_draw:
            x, y, w, h = stamp_data['rect']
            
            # 應用 Padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_w, x + w + padding)
            y2 = min(img_h, y + h + padding)

            # 計算 YOLO 格式的歸一化座標
            cx_norm = ((x1 + x2) / 2.0) * dw
            cy_norm = ((y1 + y2) / 2.0) * dh
            w_norm = (x2 - x1) * dw
            h_norm = (y2 - y1) * dh
            
            # 印章的類別 ID，假設是 0
            class_id = 0
            
            # 將格式化的字串加入列表
            yolo_labels.append(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        # 6. 將所有標註行寫入檔案
        if yolo_labels:
            try:
                with open(current_label_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_labels))
                print(f"  -> 成功生成標註檔案: {current_label_path}")
            except Exception as e:
                print(f"  -> 寫入標註檔案時發生錯誤: {e}")
    else:
        print(f"  -> 未偵測到印章，跳過生成標OTZ檔案。")

def get_top_bottom_segments(contour):
    """
    分析一個輪廓，提取其頂部和底部的水平線段資訊。

    Args:
        contour: 一個 OpenCV 輪廓 (NumPy array of shape (N, 1, 2)).

    Returns:
        一個字典，包含 'top_segment' 和 'bottom_segment'。
        如果無法提取，對應的值會是 None。
        每個 segment 是一個字典: {'start':(x,y), 'end':(x,y), 'length':val}
    """
    if contour is None or len(contour) < 10:
        return {'top_segment': None, 'bottom_segment': None}

    x, y, w, h = cv2.boundingRect(contour)

    # 定義切片高度，使用總高度的 5%，但最小為 5 像素，最大為 20 像素
    slice_height = int(max(10, min(h * 0.2, 40)))

    points = contour.reshape(-1, 2)
    top_points = points[np.where(points[:, 1] < y + slice_height)]
    bottom_points = points[np.where(points[:, 1] > y + h - slice_height)]

    result = {'top_segment': None, 'bottom_segment': None}

    if len(top_points) > 1:
        top_x_coords = top_points[:, 0]
        top_y_coords = top_points[:, 1]
        
        top_x_min, top_x_max = np.min(top_x_coords), np.max(top_x_coords)
        top_y_avg = int(np.mean(top_y_coords))
        
        result['top_segment'] = {
            'start': (top_x_min, top_y_avg),
            'end': (top_x_max, top_y_avg),
            'length': top_x_max - top_x_min
        }

    if len(bottom_points) > 1:
        bottom_x_coords = bottom_points[:, 0]
        bottom_y_coords = bottom_points[:, 1]

        bottom_x_min, bottom_x_max = np.min(bottom_x_coords), np.max(bottom_x_coords)
        bottom_y_avg = int(np.mean(bottom_y_coords))

        result['bottom_segment'] = {
            'start': (bottom_x_min, bottom_y_avg),
            'end': (bottom_x_max, bottom_y_avg),
            'length': bottom_x_max - bottom_x_min
        }
        
    return result



# --- 主程式執行區塊 ---
if __name__ == "__main__":
    # ==============================================================================

    # input_directory = r"C:\Users\chester\Desktop\commeet\全部公司四大報表_灰階_JPG"
    # output_directory = r"C:\Users\chester\Desktop\commeet\全部公司四大報表_灰階_JPG_印章辨識_yolo"
    

    # label_directory = r"C:\Users\chester\Desktop\commeet\yolo_labels"
    # if not os.path.exists(label_directory):
    #     os.makedirs(label_directory)
    # # +++++++++++++++++++++++++++++++++++++++++++++++

    # # ... (您原有的 os.makedirs 和 os.listdir) ...

    # # +++ 新增步驟：在這裡宣告一個全域變數，用來傳遞路徑 +++
    # current_label_path = ""

    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    #     print(f"已建立輸出資料夾: {output_directory}")

    # try:
    #     filenames = os.listdir(input_directory)
    # except FileNotFoundError:
    #     print(f"錯誤: 輸入資料夾不存在: {input_directory}")
    #     filenames = []

    # # 遍歷所有檔案
    # for filename in filenames:
    #     # 檢查檔案是否為常見的圖片格式
    #     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            

            
    #         # 組合完整的輸入和輸出檔案路徑
    #         input_path = os.path.join(input_directory, filename)
    #         output_path = os.path.join(output_directory, filename)

    #         # +++ 新增步驟：在迴圈內更新全域變數的值 +++
    #         base_name = os.path.splitext(filename)[0]
    #         current_label_path = os.path.join(label_directory, f"{base_name}.txt")

    #         print(f"\n正在處理檔案: {filename}")
            
    #         find_and_draw_stamp_boxes(input_path, output_path)
            
    #     else:
    #         # 如果不是圖片檔，則跳過
    #         print(f"跳過非圖片檔: {filename}")

    # print("\n所有圖片處理完成！")

    
    # input_filename = r"C:\Users\chester\Desktop\commeet\全部公司四大報表_灰階_JPG\志強-KY_2024_合併_現金流量表_1.jpg"
    # input_filename = r"C:\Users\chester\Desktop\commeet\後137公司四大報表_灰階_JPG\美利達_2024_合併_資產負債表.jpg"
    # input_filename = r"C:\Users\chester\Desktop\commeet\全部公司掃描_JPG_add_noise_easy\世紀鋼_2024_合併_資產負債表_1.jpg"
    #全部公司四大報表_僅加雜訊
    #全部公司四大報表_灰階_JPG_add_noise_hard
    input_filename = r"C:\Users\chester\Desktop\commeet\全部公司掃描_JPG_add_noise_hard\億豐_2024_個體_綜合損益表.jpg"
    output_filename = "image_detection.jpg"

    find_and_draw_stamp_boxes(input_filename, output_filename)


#世紀鋼_2024_個體_綜合損益表 
#卜蜂_2024_合併_現金流量表_2
#大亞_2024_合併_權益變動表

#亞翔_2024_合併_權益變動表
#亞德客-K_2024_合併_現金流量表_1

#亞德客-K_2024_合併_資產負債表

#京元電子_2024_合併_資產負債表_2
#京城_2024_個體_綜合損益表
#力成_2024_合併_權益變動表
#中保科_2024_合併_權益變動表
#京城_2024_個體_現金流量表
#卜蜂_2024_個體_綜合損益表
#中鼎_2024_合併_綜合損益表

# 中保科_2024_合併_資產負債表_2.jpg
# 中保科_2024_合併_權益變動表.jpg
#一詮_2024_合併_現金流量表_1.jpg
#台船_2024_合併_資產負債表_2.jpg

#"C:\Users\chester\Desktop\commeet\全部公司四大報表_灰階_JPG_印章辨識_3\聯電_2024_合併_權益變動表.jpg"
#"C:\Users\chester\Desktop\commeet\全部公司四大報表_灰階_JPG_印章辨識_2\彰銀_2024_個體_資產負債表.jpg"
# 一詮_2024_個體_權益變動表.jpg
