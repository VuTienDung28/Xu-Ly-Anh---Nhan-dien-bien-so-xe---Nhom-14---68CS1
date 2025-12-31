# plate_localization.py
import cv2
import numpy as np


def step5_canny_edge(binary_image):
    return cv2.Canny(binary_image, 100, 200)

def step6_morphological_closing(edges_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, kernel)

def step7_remove_noise_opening(binary_img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    output = np.zeros_like(binary_img)
    min_size = 30 
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
    return output

def step8_dilation(binary_img):
    # Kernel 3x3 giúp liền nét chữ tốt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(binary_img, kernel, iterations=1)



def verify_plate_texture(binary_roi):
    """Kiểm tra mật độ chuyển đổi màu (Texture) để loại bỏ nhiễu khối đặc"""
    h, w = binary_roi.shape
    if h == 0 or w == 0: return 0
    # Chỉ xét vùng trung tâm để tránh viền
    roi_center = binary_roi[int(h*0.25):int(h*0.75), int(w*0.15):int(w*0.85)]
    if roi_center.size == 0: return 0
    
    transitions = 0
    for row in range(roi_center.shape[0]):
        line = roi_center[row, :]
        # Đếm số lần pixel đổi màu (Trắng->Đen hoặc ngược lại)
        diffs = np.sum(line[1:] != line[:-1])
        transitions += diffs
    
    avg_trans = transitions / roi_center.shape[0]
    return avg_trans

def step9_find_candidates_rotated(dilated_img, binary_clean_img):
    """
    Sử dụng MinAreaRect để tìm hình chữ nhật xoay tối thiểu (ôm sát biển số).
    """
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for cnt in contours:
        # Lấy Box xoay (center, size, angle)
        rect = cv2.minAreaRect(cnt) 
        (center, size, angle) = rect
        (w, h) = size
        
        # Chuẩn hóa w, h (OpenCV có thể trả về w < h kèm góc 90 độ)
        if w < h: w, h = h, w
            
        area = w * h
        if area < 500: continue # Quá nhỏ
        
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Tỷ lệ biển số Việt Nam
        is_moto = 1.1 <= aspect_ratio <= 1.8
        is_car = 1.9 <= aspect_ratio <= 6.5
        
        if is_moto or is_car:
            x_u, y_u, w_u, h_u = cv2.boundingRect(cnt)
            roi_bin = binary_clean_img[y_u:y_u+h_u, x_u:x_u+w_u]
            
            # Kiểm tra mật độ ký tự
            avg_trans = verify_plate_texture(roi_bin)
            
            # Ngưỡng texture: > 3.0 (phải có ký tự)
            if avg_trans > 3.0: 
                # Tính điểm ưu tiên (Score)
                box_area = w * h
                contour_area = cv2.contourArea(cnt)
                solidity = contour_area / box_area if box_area > 0 else 0
                
                # Score = Mật độ * Độ đặc (Ưu tiên hình chữ nhật rõ ràng)
                score = avg_trans * solidity
                candidates.append((rect, score))

    if not candidates: return None
    
    # Chọn ứng viên có điểm cao nhất
    best_candidate = sorted(candidates, key=lambda c: c[1], reverse=True)[0]
    return best_candidate[0] # Trả về rect object

def step10_smart_crop_rotate_v3_7(img, rect):
    """
    Cắt và Xoay thông minh (Smart Crop & Rotate).
    - Không cắt bớt viền (gây lẹm chữ).
    - Thêm biên an toàn (+4px).
    - Chỉ xoay nếu góc nghiêng thực sự đáng kể.
    """
    (center, (w, h), angle) = rect

    # 1. Chuẩn hóa góc xoay
    if w < h:
        w, h = h, w
        angle += 90
    
    # 2. Logic "Khóa góc" (Angle Clamping)
    # Nếu nghiêng > 20 độ -> Có thể là lỗi -> Ép về 0
    if angle > 20 or angle < -20: angle = 0
    
    # Nếu nghiêng quá ít (< 2 độ) -> Ép về 0 để giữ độ nét
    if abs(angle) < 2: angle = 0

    # 3. Thêm biên an toàn (Safety Padding)
    # Cộng thêm pixel thay vì trừ đi để tránh cắt vào chữ
    w += 6 
    h += 6

    # 4. Xoay ảnh (Affine Warp)
    box_center = (int(center[0]), int(center[1]))
    M = cv2.getRotationMatrix2D(box_center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    # 5. Cắt ảnh con (Sub-pixel accuracy)
    cropped = cv2.getRectSubPix(rotated_img, (int(w), int(h)), box_center)
    
    return cropped

# --- PIPELINE CHÍNH ---

def locate_plate_pipeline_full_steps(binary_image, original_image):
    """
    Pipeline định vị biển số (V3.7 Logic).
    Input: 
      - binary_image: Ảnh nhị phân từ Preprocessing (đã qua Otsu).
      - original_image: Ảnh màu gốc.
    """
    steps = {}
    
    # Xử lý hình thái học
    edges = step5_canny_edge(binary_image)
    steps["6_canny"] = edges
    
    closed = step6_morphological_closing(edges)
    steps["7_closing"] = closed
    
    no_noise = step7_remove_noise_opening(closed)
    steps["8_noise_removed"] = no_noise # Ảnh sạch dùng để check texture
    
    dilated = step8_dilation(no_noise)
    steps["9_dilated"] = dilated # Ảnh dùng để tìm contour
    
    # Bước 9: Tìm ứng viên 
    
    rect = step9_find_candidates_rotated(dilated, no_noise)
    
    if rect:
        # Vẽ box ứng viên lên ảnh gốc 
        box_points = cv2.boxPoints(rect)
        box_points = np.int64(box_points)
        candidate_viz = original_image.copy()
        cv2.drawContours(candidate_viz, [box_points], 0, (0, 255, 0), 2)
        steps["10_candidate_box"] = candidate_viz
        
        # : Cắt & Xoay V3.7 (Fix crop)
        final_plate = step10_smart_crop_rotate_v3_7(original_image, rect)
        steps["11_final_plate"] = final_plate
        
       
        steps["12_final_cut"] = final_plate 
    else:
        steps["10_candidate_box"] = np.zeros_like(original_image)
        print("Không tìm thấy biển số!")

    return steps