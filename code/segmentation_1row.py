import cv2
import numpy as np

# ==========================================
# 1. CÁC HÀM TIỀN XỬ LÝ 
# ==========================================

def remove_speckles(binary_img, min_size=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    clean_img = binary_img.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_size:
            clean_img[labels == i] = 0
    return clean_img

def remove_horizontal_lines(binary_img):
    h, w = binary_img.shape
    row_sums = np.sum(binary_img, axis=1) 
    threshold = w * 255 * 0.6 
    binary_clean = binary_img.copy()
    for r in range(h):
        if row_sums[r] > threshold:
            binary_clean[r, :] = 0
    return binary_clean

def preprocess_one_row(img):
    if img is None: return None, None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    target_size = (240, 70) 
    img_resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_resized)
    
    img_blurred = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        img_blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 5
    )
    
    # 1. Cắt nhẹ biên (Chỉ 2px để không phạm vào số)
    margin = 2
    h, w = binary.shape
    binary[:margin, :] = 0
    binary[h-margin:, :] = 0
    binary[:, :margin] = 0
    binary[:, w-margin:] = 0
    
    # 2. Xóa kẻ ngang
    binary = remove_horizontal_lines(binary)
    
    # 3. Xóa nhiễu đốm
    binary_clean = remove_speckles(binary, min_size=20)
    
    # KHÔNG dùng remove_vertical_borders theo cột nữa vì dễ xóa nhầm số 1 sát mép.
    # Ta sẽ xử lý việc đó ở bước Filter thông minh hơn.

    return binary_clean, img_resized

# ==========================================
# 2. THUẬT TOÁN CẮT 
# ==========================================

def get_vertical_projection(image):
    return np.sum(image, axis=0) / 255 

def recursive_segmentation(image, projection, threshold=0, width_limit=25):
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

# ==========================================
# 3. BỘ LỌC PHÂN VÙNG (DUAL-ZONE FILTER) 
# ==========================================

def filter_one_row(binary_img, segments):
    h_plate, w_plate = binary_img.shape
    final_chars = []

    # Định nghĩa "Vùng Biên Giới" (Danger Zone)
    # 8 pixel đầu và 8 pixel cuối
    BORDER_ZONE_WIDTH = 8 
    
    for (x1, x2) in segments:
        col = binary_img[:, x1:x2]
        contours, _ = cv2.findContours(col, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        
        cnt = max(contours, key=cv2.contourArea)
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        
        real_x = x1 + x_cnt
        real_y = y_cnt
        
        # Tính tỷ lệ khung hình (Aspect Ratio)
        aspect = w_cnt / float(h_cnt)
        
        # Kiểm tra xem đối tượng có nằm trong "Vùng Biên Giới" không?
        is_in_danger_zone = (real_x < BORDER_ZONE_WIDTH) or ((real_x + w_cnt) > (w_plate - BORDER_ZONE_WIDTH))
        
        # --- LOGIC PHÂN VÙNG ---
        
        if is_in_danger_zone:
            # === ZONE 1: SÁT MÉP (Khắt khe) ===
           
            if aspect < 0.25: 
                continue # Loại bỏ viền mảnh
            
        else:
            # === ZONE 2: Ở GIỮA  ===
           
            if aspect < 0.05: 
                continue # Chỉ loại nếu nó thực sự là một sợi chỉ nhiễu (< 5%)

        # --- CÁC LỌC CHUNG KHÁC ---
        # 1. Chiều cao:
        if h_cnt < h_plate * 0.35: continue
            
        # 2. Tỷ lệ tối đa (để không lấy phải mảng bám quá to)
        if aspect > 1.0: continue

        final_chars.append((real_x, real_y, w_cnt, h_cnt))
        
    return final_chars

# ==========================================
# 4. PIPELINE
# ==========================================

def segment_plate_pipeline(plate_img):
    if plate_img is None or plate_img.size == 0: return [], [], None, None
    
    # 1. Preprocess
    binary, gray_resized = preprocess_one_row(plate_img)
    if binary is None: return [], [], None, None
    
    h, w = binary.shape
    
    # 2. Cắt dọc
    proj = get_vertical_projection(binary)
    
    segs = recursive_segmentation(binary, proj, threshold=1, width_limit=25)
    
    # 3. Lọc phân vùng
    final_chars_box = filter_one_row(binary, segs)
    
    # 4. Crop Output
    char_imgs = []
    final_rects = []
    
    orig_h, orig_w = plate_img.shape[:2]
    scale_x = orig_w / float(w)
    scale_y = orig_h / float(h)
    margin = 2
    
    for (x, y, cw, ch) in final_chars_box:
        real_x = int(x * scale_x)
        real_y = int(y * scale_y)
        real_w = int(cw * scale_x)
        real_h = int(ch * scale_y)
        
        p_x1 = max(0, real_x - margin)
        p_y1 = max(0, real_y - margin)
        p_x2 = min(orig_w, real_x + real_w + margin)
        p_y2 = min(orig_h, real_y + real_h + margin)
        
        char_crop = plate_img[p_y1:p_y2, p_x1:p_x2]
        char_imgs.append(char_crop)
        final_rects.append((p_x1, p_y1, p_x2-p_x1, p_y2-p_y1))
        
    return char_imgs, final_rects, binary, gray_resized