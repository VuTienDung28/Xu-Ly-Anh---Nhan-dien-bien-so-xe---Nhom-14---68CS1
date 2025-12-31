# test_visualize.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Import các module
try:
    import preprocessing
    import plate_localization
    import segmentation_1row 
    import segmentation_2row
    
except ImportError as e:
    print(f"Lỗi Import: {e}")
    exit()

def create_char_montage(char_imgs, target_h=50):
    if not char_imgs: return None
    resized_chars = []
    for char in char_imgs:
        h, w = char.shape[:2]
        if h == 0: continue
        aspect = w / float(h)
        target_w = int(target_h * aspect)
        char_resized = cv2.resize(char, (target_w, target_h))
        if len(char_resized.shape) == 2:
            char_resized = cv2.cvtColor(char_resized, cv2.COLOR_GRAY2BGR)
        resized_chars.append(char_resized)
        spacer = np.ones((target_h, 5, 3), dtype=np.uint8) * 255 
        resized_chars.append(spacer)
    if not resized_chars: return None

    return np.hstack(resized_chars[:-1])

def visualize_result_clean(steps_dict):
    
    ordered_keys = [
        ("1_original", "1. Ảnh Gốc"),
        ("5_binary", "2. Nhị phân Otsu"),
        ("9_dilated", "3. Giãn nở"),
        ("10_candidate_box", "4. Vùng Ứng Viên"),
        ("11_final_plate", "5. Cắt Biển (Final)"), 
        ("12_seg_binary", "6. Nhị phân (Seg)"),
        ("12_viz_boxes", "7. VỊ TRÍ CẮT"),
        ("13_char_montage", "8. KẾT QUẢ")
    ]
    
    images_to_show = []
    seen_titles = set()
    for key, title in ordered_keys:
        val = steps_dict.get(key)
        if val is None and key == "11_final_plate": val = steps_dict.get("12_final_plate")
        if val is not None and title not in seen_titles:
            images_to_show.append((title, val))
            seen_titles.add(title)
    
    if not images_to_show: return

    n = len(images_to_show)
    cols = 3 
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(f"QUY TRÌNH FULL (AUTO DETECT 1-ROW / 2-ROWS)", fontsize=16, fontweight='bold')
    
    if rows == 1 and cols == 1: axes = [axes]
    elif rows == 1 or cols == 1: axes = axes.flatten()
    else: axes = axes.flatten()

    for i, (title, img) in enumerate(images_to_show):
        ax = axes[i]
        if len(img.shape) == 2: ax.imshow(img, cmap='gray')
        else: ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11, pad=10)
        ax.axis('off')
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_color('#ddd')
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def visualize_full_process(image_path):
    print(f"\n--- ĐANG XỬ LÝ: {image_path} ---")
    if not os.path.exists(image_path): print("Không thấy file!"); return

    try:
        # Giai đoạn 1 & 2
        pre_steps = preprocessing.preprocess_pipeline_full_steps(image_path)
        loc_steps = plate_localization.locate_plate_pipeline_full_steps(
            pre_steps["5_binary"], pre_steps["1_original"]
        )
        
        seg_steps = {}
        final_plate = loc_steps.get("11_final_plate")
        if final_plate is None: final_plate = loc_steps.get("12_final_plate")
            
        if final_plate is not None:
            
            h, w = final_plate.shape[:2]
            aspect = w / float(h)
            
            # Ngưỡng 2.2 để phân loại
            if aspect < 2.2:
                print(f"-> Phát hiện BIỂN 2 DÒNG (Aspect={aspect:.2f}). Dùng segmentation_2row.")
                chars, rects, bin_debug, gray_debug = segmentation_2row.segment_plate_pipeline(final_plate)
            else:
                print(f"-> Phát hiện BIỂN 1 DÒNG (Aspect={aspect:.2f}). Dùng segmentation_1row.")
                chars, rects, bin_debug, gray_debug = segmentation_1row.segment_plate_pipeline(final_plate)
            # =========================================
            # DEBUG: IN KÍCH THƯỚC KÝ TỰ
            # =========================================
            print("\n===== DEBUG CHAR SHAPES =====")
            if chars:
                for i, c in enumerate(chars):
                    print(f"Char {i} size: {c.shape}")
            else:
                print("KHÔNG TÁCH ĐƯỢC KÝ TỰ")
            print("================================\n")
            
            seg_steps["12_seg_binary"] = bin_debug
            viz_plate = final_plate.copy()
            if rects:
                for (x, y, w, h) in rects:
                    cv2.rectangle(viz_plate, (x, y), (x+w, y+h), (0, 255, 0), 1)
            seg_steps["12_viz_boxes"] = viz_plate 
            
            if chars:
                montage = create_char_montage(chars)
                seg_steps["13_char_montage"] = montage
                print(f"-> Tách được: {len(chars)} ký tự.")
            else:
                print("-> Không tách được ký tự nào.")
        else:
            print("-> Không tìm thấy biển số.")
        
        full_data = {**pre_steps, **loc_steps, **seg_steps}
        visualize_result_clean(full_data)
        
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":

    img_path = './images/train/carlong_0280.png ' 
    visualize_full_process(img_path)
    