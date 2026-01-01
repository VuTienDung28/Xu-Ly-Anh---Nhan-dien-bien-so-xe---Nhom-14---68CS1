# train_knn.py
import cv2
import numpy as np
import os

def train_and_save(label_file, data_file, output_filename):
    if not os.path.exists(label_file) or not os.path.exists(data_file):
        print(f"[BỎ QUA] Không thấy file dữ liệu: {label_file}")
        return

    try:
        # 1. Đọc dữ liệu
        classifications = np.loadtxt(label_file, np.float32)
        flattened_images = np.loadtxt(data_file, np.float32)
        
        # 2. Reshape label cho đúng chuẩn OpenCV
        classifications = classifications.reshape((classifications.size, 1))

        # 3. Train
        print(f"Đang train model: {output_filename}...")
        kNearest = cv2.ml.KNearest_create()
        kNearest.train(flattened_images, cv2.ml.ROW_SAMPLE, classifications)

        # 4. Lưu model
        kNearest.save(output_filename)
        print(f"-> Đã lưu thành công: {output_filename}")

    except Exception as e:
        print(f"Lỗi khi train {output_filename}: {e}")

if __name__ == "__main__":
    # Train Model 1 (Cho biển 1 dòng - Ô tô)
    train_and_save("classifications_1row_AUG.txt", "flattened_images_1row_AUG.txt", "knn_model_1row.xml")

    # Train Model 2 (Cho biển 2 dòng - Xe máy)
    train_and_save("classifications_2row_AUG.txt", "flattened_images_2row_AUG.txt", "knn_model_2row.xml")