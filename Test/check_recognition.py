import cv2
import numpy as np
import os
import csv
import glob
import re
import sys

# ==================================================================================
# 1. CẤU HÌNH HỆ THỐNG (SỬA ĐƯỜNG DẪN TẠI ĐÂY)
# ==================================================================================
IMG_DIR = "./datasets/imgs/1_row/"       # Thư mục chứa ảnh test
GT_CSV_FILE = "./datasets/1_row.csv"                # File Ground Truth
OUTPUT_CSV = "evaluation_result_1.csv"     # File kết quả đầu ra

# Kích thước Model (Phải khớp 100% với lúc train)
SIZE_1ROW = (13, 19)
SIZE_2ROW = (14, 30)

# Import module của bạn
try:
    import preprocessing
    import plate_localization
    import segmentation_1row
    import segmentation_2row
    print("✅ Đã import thành công các module xử lý ảnh.")
except ImportError as e:
    print(f"❌ LỖI NGHIÊM TRỌNG: Thiếu module code! ({e})")
    print("Hãy đảm bảo file này nằm cùng thư mục với preprocessing.py, plate_localization.py...")
    sys.exit()

knn_1row = None
knn_2row = None

# ==================================================================================
# 2. CÁC HÀM HỖ TRỢ (UTILITIES)
# ==================================================================================
def load_models():
    """Load model KNN, kiểm tra kỹ đường dẫn"""
    global knn_1row, knn_2row
    
    # Load Model 1 dòng
    if os.path.exists("knn_model_1row.xml"):
        knn_1row = cv2.ml.KNearest_load("knn_model_1row.xml")
        print("✅ Đã load Model 1 dòng (Ô tô).")
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy 'knn_model_1row.xml'. Máy sẽ không nhận diện được biển ô tô!")

    # Load Model 2 dòng
    if os.path.exists("knn_model_2row.xml"):
        knn_2row = cv2.ml.KNearest_load("knn_model_2row.xml")
        print("✅ Đã load Model 2 dòng (Xe máy).")
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy 'knn_model_2row.xml'.")

def read_image_safe(path):
    """
    Hàm đọc ảnh an toàn, chấp nhận đường dẫn tiếng Việt/Ký tự lạ
    Thay thế cho cv2.imread thường hay bị lỗi None
    """
    try:
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"Lỗi đọc file ảnh {path}: {e}")
        return None

def normalize_text(text):
    """Viết hoa, xóa ký tự đặc biệt"""
    if not text: return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def calculate_metrics_lcs(gt, pred):
    """Tính TP, FP, FN dựa trên thuật toán LCS (Chuỗi con chung)"""
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    tp = dp[m][n]           # Ký tự đúng
    fp = len(pred) - tp     # Ký tự thừa (rác)
    fn = len(gt) - tp       # Ký tự sót
    return tp, fp, fn

def calculate_p_r_f1(tp, fp, fn):
    """Tính Precision, Recall, F1 an toàn"""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

# ==================================================================================
# 3. PIPELINE NHẬN DIỆN CHÍNH (PREDICT) - ĐÃ SỬA LỖI NUMPY
# ==================================================================================
def predict_plate_string(img_path):
    """Chạy toàn bộ quy trình từ Ảnh -> Chuỗi ký tự"""
    # 1. Đọc ảnh
    img_origin = read_image_safe(img_path)
    if img_origin is None:
        print(f"❌ Không đọc được ảnh: {os.path.basename(img_path)}")
        return ""

    try:
        # 2. Preprocess & Localize
        pre = preprocessing.preprocess_pipeline_full_steps(img_path)
        
        # Nếu preprocessing fail
        if not pre or "5_binary" not in pre:
            return ""

        loc = plate_localization.locate_plate_pipeline_full_steps(pre["5_binary"], img_origin)
        
        # --- SỬA LỖI TẠI ĐÂY ---
        # Code cũ (Gây lỗi): final_plate = loc.get("11_final_plate") or loc.get("12_final_plate")
        # Code mới (An toàn):
        final_plate = loc.get("11_final_plate")
        if final_plate is None:
            final_plate = loc.get("12_final_plate")
        # -----------------------
        
        if final_plate is None:
            return ""

        # 3. Segmentation (Cắt ký tự)
        h, w = final_plate.shape[:2]
        is_2row = (w / h) < 2.2
        
        if is_2row:
            chars, _, _, _ = segmentation_2row.segment_plate_pipeline(final_plate)
            model = knn_2row
            target_size = SIZE_2ROW
        else:
            chars, _, _, _ = segmentation_1row.segment_plate_pipeline(final_plate)
            model = knn_1row
            target_size = SIZE_1ROW
        
        if not chars: return ""
        if model is None: 
            # print(f"❌ Lỗi: Chưa load model (is_2row={is_2row})")
            return ""

        # 4. Nhận diện từng ký tự (OCR)
        pred_text = ""
        for char_img in chars:
            # Chuyển xám nếu cần
            if len(char_img.shape) == 3:
                gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = char_img
            
            # Threshold lại
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Resize
            img_resized = cv2.resize(bin_img, target_size, interpolation=cv2.INTER_AREA)
            img_flat = img_resized.reshape((1, target_size[0] * target_size[1])).astype(np.float32)
            
            # Predict
            retval, results, _, _ = model.findNearest(img_flat, k=1)
            pred_text += str(chr(int(results[0][0])))
            
        return pred_text

    except Exception as e:
        # In lỗi ra để biết tại sao cột Predict bị rỗng
        print(f"❌ Lỗi Runtime tại {os.path.basename(img_path)}: {e}")
        return ""
# ==================================================================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN LOOP)
# ==================================================================================
def main():
    print("\n" + "="*50)
    print("   CHƯƠNG TRÌNH ĐÁNH GIÁ NHẬN DIỆN BIỂN SỐ")
    print("="*50)
    
    # 1. Load Model
    load_models()
    if knn_1row is None and knn_2row is None:
        print("❌ CẢNH BÁO: Không có model nào được load. Dừng chương trình.")
        return

    # 2. Đọc file CSV Ground Truth
    gt_dict = {}
    if os.path.exists(GT_CSV_FILE):
        with open(GT_CSV_FILE, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None) # Bỏ qua header
            for row in reader:
                if len(row) >= 2:
                    gt_dict[row[0].strip()] = normalize_text(row[1])
        print(f"✅ Đã đọc {len(gt_dict)} nhãn từ file {GT_CSV_FILE}")
    else:
        print(f"❌ Lỗi: Không tìm thấy file {GT_CSV_FILE}")
        return

    # 3. Lấy danh sách ảnh
    all_images = glob.glob(os.path.join(IMG_DIR, "*.*"))
    print(f"✅ Tìm thấy {len(all_images)} ảnh trong thư mục test.")

    # Biến tổng hợp
    g_tp = 0
    g_fp = 0
    g_fn = 0
    count_perfect = 0
    total_processed = 0

    print("\n--- ĐANG CHẠY... (Vui lòng chờ) ---")

    # Mở file ghi kết quả
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['Filename', 'GT', 'Predict', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'Perfect'])
        
        for i, img_path in enumerate(all_images):
            filename = os.path.basename(img_path)
            
            # Bỏ qua nếu không có nhãn
            if filename not in gt_dict:
                continue
            
            total_processed += 1
            gt_text = gt_dict[filename]
            
            # --- PREDICT ---
            pred_raw = predict_plate_string(img_path)
            pred_text = normalize_text(pred_raw)
            
            # --- CALC METRICS ---
            tp, fp, fn = calculate_metrics_lcs(gt_text, pred_text)
            p, r, f1 = calculate_p_r_f1(tp, fp, fn)
            
            is_perfect = (gt_text == pred_text) and (len(gt_text) > 0)
            
            # Cộng dồn
            g_tp += tp
            g_fp += fp
            g_fn += fn
            if is_perfect: count_perfect += 1
            
            # Ghi file
            writer.writerow([
                filename, gt_text, pred_text,
                tp, fp, fn,
                f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}",
                "TRUE" if is_perfect else "FALSE"
            ])
            
            # Print progress bar
            if (i+1) % 10 == 0:
                print(f"-> Đã xử lý {i+1}/{len(all_images)} ảnh...")

    # ==================================================================================
    # 5. BÁO CÁO TỔNG KẾT (CONSOLE REPORT)
    # ==================================================================================
    # Tính chỉ số toàn cục (Global Metrics)
    gp, gr, gf1 = calculate_p_r_f1(g_tp, g_fp, g_fn)
    
    acc_plate = (count_perfect / total_processed * 100) if total_processed > 0 else 0
    
    print("\n" + "="*60)
    print(f"{'BÁO CÁO KẾT QUẢ CUỐI CÙNG (FINAL REPORT)':^60}")
    print("="*60)
    print(f"1. Tổng số ảnh đã test:     {total_processed}")
    print(f"2. Số biển đúng tuyệt đối:  {count_perfect} ({acc_plate:.2f}%)")
    print("-" * 60)
    print("3. THỐNG KÊ CHI TIẾT KÝ TỰ (Character Level):")
    print(f"   - Tổng ký tự ĐÚNG (TP):  {g_tp}")
    print(f"   - Tổng ký tự THỪA (FP):  {g_fp} (Nhiễu, sai)")
    print(f"   - Tổng ký tự SÓT (FN):   {g_fn}")
    print("-" * 60)
    print("4. CHỈ SỐ HIỆU SUẤT TOÀN CỤC (Global Metrics):")
    print(f"   - Precision:             {gp*100:.2f}%  (Độ tin cậy)")
    print(f"   - Recall:                {gr*100:.2f}%  (Độ phủ)")
    print(f"   >>> GLOBAL F1-SCORE:     {gf1*100:.2f}% <<< (Con số chốt)")
    print("="*60)
    print(f"Chi tiết từng ảnh đã lưu tại: {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":
    main()