import cv2
import numpy as np
import os
import glob
import csv  # <--- Thêm thư viện này

# ==================================================================================
# 1. CẤU HÌNH
# ==================================================================================
IMG_DIR = "./datasets/imgs/2_row/"       # Thư mục ảnh
LABEL_DIR = "./datasets/labels/2_row/"   # Thư mục nhãn (.txt)
OUTPUT_CSV = "localization_results_2row.csv"  # Tên file kết quả đầu ra

IOU_THRESHOLD = 0.5

# Import module của bạn
try:
    import preprocessing
    import plate_localization
except ImportError:
    print("Lỗi: Không tìm thấy file 'preprocessing.py' hoặc 'plate_localization.py'")
    exit()

# ==================================================================================
# 2. CÁC HÀM TÍNH TOÁN (Giữ nguyên như cũ)
# ==================================================================================
def get_ground_truth_polygon(txt_path, img_w, img_h):
    if not os.path.exists(txt_path): return None
    with open(txt_path, 'r') as f:
        content = f.read().strip().split()
    if len(content) < 9: return None 
    coords = [float(x) for x in content[1:]]
    points = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * img_w)
        y = int(coords[i+1] * img_h)
        points.append([x, y])
    return np.array(points, dtype=np.float32)

def get_predicted_polygon(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, None
    h_img, w_img = img.shape[:2]

    # Preprocessing
    pre_steps = preprocessing.preprocess_pipeline_full_steps(image_path)
    binary_img = pre_steps.get("5_binary")
    if binary_img is None: return None, (h_img, w_img)

    # Localization logic
    edges = plate_localization.step5_canny_edge(binary_img)
    closed = plate_localization.step6_morphological_closing(edges)
    no_noise = plate_localization.step7_remove_noise_opening(closed)
    dilated = plate_localization.step8_dilation(no_noise)
    rect = plate_localization.step9_find_candidates_rotated(dilated, no_noise)
    
    if rect is None: return None, (h_img, w_img)
    
    box_points = cv2.boxPoints(rect)
    box_points = np.float32(box_points)
    return box_points, (h_img, w_img)

def calculate_iou(poly1, poly2):
    if poly1 is None or poly2 is None: return 0.0
    try:
        ret, intersection_poly = cv2.intersectConvexConvex(poly1, poly2)
        if not ret: return 0.0
        intersection_area = cv2.contourArea(intersection_poly)
    except:
        return 0.0
    area1 = cv2.contourArea(poly1)
    area2 = cv2.contourArea(poly2)
    union_area = area1 + area2 - intersection_area
    if union_area <= 0: return 0.0
    return intersection_area / union_area

# ==================================================================================
# 3. MAIN (Ghi ra CSV)
# ==================================================================================
def main():
    extensions = ['*.jpg', '*.png', '*.jpeg']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(IMG_DIR, ext)))
    
    if not img_files:
        print(f"Không tìm thấy ảnh trong: {IMG_DIR}")
        return

    print(f"--- ĐANG CHẠY TEST ({len(img_files)} ảnh) ---")
    print(f"Kết quả sẽ được lưu vào file: {OUTPUT_CSV}")

    count_correct = 0
    total_images = 0

    # Mở file CSV để ghi
    # newline='' là bắt buộc trên Windows để không bị dòng trống
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # 1. Ghi dòng tiêu đề (Header)
        writer.writerow(['filename', 'iou', 'value'])

        # 2. Duyệt qua từng ảnh
        for i, img_path in enumerate(img_files):
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]
            txt_path = os.path.join(LABEL_DIR, name_no_ext + ".txt")
            
            # Bỏ qua nếu không có file nhãn
            if not os.path.exists(txt_path):
                continue
                
            img_temp = cv2.imread(img_path)
            if img_temp is None: continue
            h, w = img_temp.shape[:2]

            # Lấy Ground Truth & Prediction
            gt_poly = get_ground_truth_polygon(txt_path, w, h)
            if gt_poly is None: continue

            total_images += 1
            pred_poly, _ = get_predicted_polygon(img_path)
            
            # Tính IoU
            iou = calculate_iou(gt_poly, pred_poly)
            
            # Xác định trạng thái
            is_correct = iou > IOU_THRESHOLD
            status = "TRUE" if is_correct else "FALSE"
            
            if is_correct: count_correct += 1

            # 3. Ghi dữ liệu vào CSV
            # Format IoU lấy 4 chữ số thập phân cho gọn
            writer.writerow([filename, f"{iou:.4f}", status])

            # In tiến trình ra console (cứ 10 ảnh in 1 lần cho đỡ rối)
            if (i+1) % 10 == 0:
                print(f"Đã xử lý {i+1}/{len(img_files)} ảnh...")

    # Tổng kết
    accuracy = (count_correct / total_images * 100) if total_images > 0 else 0
    
    print("\n" + "="*40)
    print("HOÀN TẤT!")
    print(f"File CSV đã được tạo tại: {os.path.abspath(OUTPUT_CSV)}")
    print(f"Tổng số ảnh có nhãn: {total_images}")
    print(f"Số lượng ĐÚNG (TRUE):  {count_correct}")
    print(f"Độ chính xác:          {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()