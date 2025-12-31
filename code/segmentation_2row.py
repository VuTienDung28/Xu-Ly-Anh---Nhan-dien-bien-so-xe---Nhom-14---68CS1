# segmentation_2row.py
import cv2
import numpy as np

# ==========================================
# 1. TIỀN XỬ LÝ (CHUYÊN BIỆT 2 DÒNG)
# ==========================================

def preprocess_two_rows(img):
    if img is None: return None, None

    # Chuyển xám
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 1. Resize vuông vức (140x120) cho biển xe máy
    target_size = (140, 120)
    img_resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_CUBIC)
    
    # 2. Tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_resized)
    
    # 3. Adaptive Threshold
    img_blurred = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        img_blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        19, 5
    )
    
    # 4. CẮT VIỀN 
   
    margin = 4
    h, w = binary.shape
    binary[:margin, :] = 0       
    binary[h-margin:, :] = 0     
    binary[:, :margin] = 0       
    binary[:, w-margin:] = 0     
    
    # Lọc nhiễu
    kernel = np.ones((2,2), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary_clean, img_resized

# ==========================================
# 2. CÁC HÀM CẮT & TÁCH DÒNG
# ==========================================

def get_vertical_projection(image):
    return np.sum(image, axis=0) / 255 

def get_horizontal_projection(image):
    return np.sum(image, axis=1) / 255

def split_two_rows(binary_img):
    """Cắt đôi biển số thành dòng trên/dưới"""
    h, w = binary_img.shape
    proj = get_horizontal_projection(binary_img)
    
    start = int(h * 0.25)
    end = int(h * 0.75)
    center_proj = proj[start:end]
    
    if len(center_proj) == 0: return binary_img, None, 0
    
    split_idx = np.argmin(center_proj)
    split_y = start + split_idx
    
    row1 = binary_img[0:split_y, :]
    row2 = binary_img[split_y:, :]
    
    return row1, row2, split_y

def recursive_segmentation(image, projection, threshold=0, width_limit=35):
    segments = []
    start = None
    width = image.shape[1]
    
    for x, val in enumerate(projection):
        if val > threshold and start is None:
            start = x
        elif (val <= threshold or x == width - 1) and start is not None:
            end = x
            if (end - start) > width_limit:
                sub_img = image[:, start:end]
                sub_proj = get_vertical_projection(sub_img)
                sub_segs = recursive_segmentation(sub_img, sub_proj, threshold + 2, width_limit)
                if sub_segs:
                    for s, e in sub_segs: segments.append((start + s, start + e))
                else:
                    segments.append((start, end))
            else:
                segments.append((start, end))
            start = None 
    return segments

def filter_two_rows(binary_img, segments, offset_x=0, offset_y=0, h_plate=0):
    final_chars = []
    if h_plate == 0: h_plate = binary_img.shape[0]
    
    # THAM SỐ LỌC CHO BIỂN 2 DÒNG
    min_height_ratio = 0.35 
    safe_margin = 7
        
    for (x1, x2) in segments:
        col = binary_img[:, x1:x2]
        contours, _ = cv2.findContours(col, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        real_x = offset_x + x1 + x
        real_y = offset_y + y
        
        # 1. Lọc biên: Loại bỏ rác sát mép
        if real_x < safe_margin or (real_x + w) > (binary_img.shape[1] - safe_margin):
            continue

        # 2. Chiều cao
        if h < binary_img.shape[0] * min_height_ratio: continue
            
        # 3. Tỷ lệ
        aspect = w / float(h)
        if aspect < 0.1 or aspect > 1.0: 
             if not (h > binary_img.shape[0] * 0.6 and aspect > 0.08): 
                 continue

        final_chars.append((real_x, real_y, w, h))
        
    return final_chars

# ==========================================
# 3. PIPELINE
# ==========================================

def segment_plate_pipeline(plate_img):
    if plate_img is None or plate_img.size == 0: return [], [], None, None
    
    h, w = plate_img.shape[:2]
    
    # 1. Preprocess
    binary, gray_resized = preprocess_two_rows(plate_img)
    if binary is None: return [], [], None, None
    
    final_chars_box = []
    
    # 2. Tách dòng
    row1_img, row2_img, split_y = split_two_rows(binary)
    
    # 3. Xử lý từng dòng
    # Dòng 1
    proj1 = get_vertical_projection(row1_img)
    segs1 = recursive_segmentation(row1_img, proj1, threshold=2, width_limit=35)
    chars1 = filter_two_rows(row1_img, segs1, offset_y=0, h_plate=h)
    
    # Dòng 2
    if row2_img is not None:
        proj2 = get_vertical_projection(row2_img)
        segs2 = recursive_segmentation(row2_img, proj2, threshold=2, width_limit=35)
        chars2 = filter_two_rows(row2_img, segs2, offset_y=split_y, h_plate=h)
        final_chars_box = chars1 + chars2
    else:
        final_chars_box = chars1
        
    # 4. Sắp xếp
    final_chars_box = sorted(final_chars_box, key=lambda c: (c[1] // (binary.shape[0]//2), c[0]))

    # 5. Crop Output
    char_imgs = []
    final_rects = []
    
    target_h, target_w = binary.shape
    scale_x = w / float(target_w)
    scale_y = h / float(target_h)
    margin = 2
    
    for (x, y, cw, ch) in final_chars_box:
        real_x = int(x * scale_x)
        real_y = int(y * scale_y)
        real_w = int(cw * scale_x)
        real_h = int(ch * scale_y)
        
        p_x1 = max(0, real_x - margin)
        p_y1 = max(0, real_y - margin)
        p_x2 = min(w, real_x + real_w + margin)
        p_y2 = min(h, real_y + real_h + margin)
        
        char_crop = plate_img[p_y1:p_y2, p_x1:p_x2]
        char_imgs.append(char_crop)
        final_rects.append((p_x1, p_y1, p_x2-p_x1, p_y2-p_y1))
        
    return char_imgs, final_rects, binary, gray_resized