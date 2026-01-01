import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Import các module
try:
    import preprocessing
    import plate_localization
    import segmentation_2row
    import segmentation_1row
except ImportError as e:
    print(f"Thiếu module: {e}")
    sys.exit()

# ==================================================================================
# 1. CẤU HÌNH & LOAD MODEL
# ==================================================================================
# Kích thước train (khớp với Data)
SIZE_1ROW = (13, 19) 
SIZE_2ROW = (14, 30)

knn_1row = None
knn_2row = None

def load_models():
    global knn_1row, knn_2row
    # Load model
    if os.path.exists("knn_model_1row.xml"):
        knn_1row = cv2.ml.KNearest_load("knn_model_1row.xml")
    
    if os.path.exists("knn_model_2row.xml"):
        knn_2row = cv2.ml.KNearest_load("knn_model_2row.xml")

    print(f"Model Status: 1-Row: {'OK' if knn_1row else 'MISSING'} | 2-Row: {'OK' if knn_2row else 'MISSING'}")

# ==================================================================================
# 2. HÀM XỬ LÝ QUAN TRỌNG NHẤT
# ==================================================================================
def preprocess_and_predict(char_img_original, is_2row):
    """
    Hàm này nhận đầu vào là 1 ảnh ký tự cắt từ 'Phần 8',
    Thực hiện nhị phân lại cho phù hợp, sau đó nhận diện.
    """
    if char_img_original is None or char_img_original.size == 0: return "?"
    
    # BƯỚC A: CHUẨN HÓA ĐẦU VÀO (XÁM)
    if len(char_img_original.shape) == 3:
        img_gray = cv2.cvtColor(char_img_original, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = char_img_original

    # BƯỚC B: NHỊ PHÂN LẠI (Re-binarize)
    # Dùng Otsu để tự tìm ngưỡng, dùng BINARY_INV để đảo nền trắng thành đen
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # BƯỚC C: CHUẨN BỊ CHO KNN
    # Chọn model & size
    if is_2row:
        model = knn_2row
        target_size = SIZE_2ROW
    else:
        model = knn_1row
        target_size = SIZE_1ROW
    
    if model is None: return "?"

    # Resize ảnh NHỊ PHÂN (img_binary) chứ không phải ảnh gốc
    img_resized = cv2.resize(img_binary, target_size, interpolation=cv2.INTER_AREA)
    img_flat = img_resized.reshape((1, target_size[0] * target_size[1])).astype(np.float32)

    # BƯỚC D: NHẬN DIỆN
    try:
        ret, results, _, _ = model.findNearest(img_flat, k=1)
        char_detected = str(chr(int(results[0][0])))
        return char_detected, img_binary # Trả về cả ảnh đã xử lý để xem
    except:
        return "?", img_binary

# ==================================================================================
# 3. HÀM HIỂN THỊ DẢI ẢNH (MONTAGE) KÈM KẾT QUẢ & GHÉP CHUỖI
# ==================================================================================
def show_result_strip(char_imgs_original, is_2row):
    if not char_imgs_original: return

    processed_imgs = []
    results_text = []

    print("\n--- KẾT QUẢ NHẬN DIỆN CHI TIẾT ---")
    
    # Vòng lặp xử lý từng ký tự trong dải ảnh (Phần 8)
    for i, char_img in enumerate(char_imgs_original):
        # Gọi hàm xử lý & nhận diện
        char_text, img_bin_debug = preprocess_and_predict(char_img, is_2row)
        
        results_text.append(char_text)
        print(f"  Char {i+1}: '{char_text}'")
        
        # Chuẩn bị ảnh để hiển thị lên dải (Resize về cùng chiều cao cho đẹp)
        H_DISP = 60
        h, w = img_bin_debug.shape
        scale = H_DISP / h
        w_new = int(w * scale)
        
        # Resize ảnh nhị phân để hiển thị
        img_disp = cv2.resize(img_bin_debug, (w_new, H_DISP))
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
        
        # Vẽ viền xanh ngăn cách
        img_disp = cv2.copyMakeBorder(img_disp, 0, 0, 2, 2, cv2.BORDER_CONSTANT, value=(0, 200, 0))
        processed_imgs.append(img_disp)

    # ---  GHÉP CHUỖI ---
    final_string = "".join(results_text)
    print("\n" + "="*40)
    print(f" >>> BIỂN SỐ NHẬN DIỆN: {final_string}")
    print("="*40 + "\n")
    montage = np.hstack(processed_imgs)

    header_h = 40
    header = np.zeros((header_h, montage.shape[1], 3), dtype=np.uint8) + 255 # Nền trắng
    
    current_x = 0
    for i, img in enumerate(processed_imgs):
        w_img = img.shape[1]
        text = results_text[i]
        
        # Canh giữa chữ
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = current_x + (w_img - text_size[0]) // 2
        
        # Ghi chữ (Màu đỏ)
        cv2.putText(header, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        current_x += w_img

    # Ghép Header + Dải ảnh
    final_result_img = np.vstack([header, montage])
    
    # Hiển thị
    plt.figure(figsize=(10, 4))
    plt.title(f"KET QUA: {final_string} (Input: Anh Nhi Phan)", fontsize=14, color='blue', fontweight='bold')
    plt.imshow(cv2.cvtColor(final_result_img, cv2.COLOR_BGR2RGB)) # Matplotlib dùng RGB
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==================================================================================
# 4. MAIN PIPELINE
# ==================================================================================
def main_run(img_path):
    load_models()
    
    print(f"Đang xử lý: {img_path}")
    img_origin = cv2.imread(img_path)
    if img_origin is None: print("Lỗi đọc ảnh"); return

    # 1. Preprocess + Localize
    pre = preprocessing.preprocess_pipeline_full_steps(img_path)
    loc = plate_localization.locate_plate_pipeline_full_steps(pre["5_binary"], img_origin)
    final_plate = loc.get("11_final_plate")
    if final_plate is None:
        final_plate = loc.get("12_final_plate")
    
    if final_plate is None: print("Không tìm thấy biển"); return

    # 2. Segmentation (Cắt ra các ký tự - Đây là 'Phần 8' bạn nói)
    h, w = final_plate.shape[:2]
    is_2row = (w/h) < 2.2
    
    if is_2row:
        char_imgs, _, _, _ = segmentation_2row.segment_plate_pipeline(final_plate)
    else:
        char_imgs, _, _, _ = segmentation_1row.segment_plate_pipeline(final_plate)
        
    if not char_imgs: print("Không cắt được ký tự"); return

    # 3. GỌI HÀM HIỂN THỊ & NHẬN DIỆN TRÊN DẢI ẢNH (Theo yêu cầu)
    show_result_strip(char_imgs, is_2row)

if __name__ == "__main__":
    img_path = './datasets/imgs/2_row/greenpack_0001.png' # Xe máy
    # img_path = './datasets/imgs/1_row/carlong_0005.png' # Ô tô
    main_run(img_path)