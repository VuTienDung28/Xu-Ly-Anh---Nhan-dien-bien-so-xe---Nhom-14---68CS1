# preprocessing.py
import cv2
import numpy as np


def step1_rgb_to_grayscale(image_rgb):
    B = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    R = image_rgb[:, :, 2]
    gray_image = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return gray_image.astype(np.uint8)

def step2_morphological_opening(gray_image, kernel_size=25):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    return opened_image

def step3_image_subtraction(gray_image, opened_image):
    subtracted_image = cv2.subtract(gray_image, opened_image)
    return subtracted_image

def step4_otsu_binarization(subtracted_image):
    T, binary_image = cv2.threshold(subtracted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def preprocess_pipeline_full_steps(image_path):
    """Trả về dictionary chứa tất cả các bước tiền xử lý"""
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError("Không tìm thấy ảnh.")
    
    gray = step1_rgb_to_grayscale(original_img)
    opened = step2_morphological_opening(gray, kernel_size=25)
    subtracted = step3_image_subtraction(gray, opened)
    binary = step4_otsu_binarization(subtracted)
    
    steps = {
        "1_original": original_img,
        "2_gray": gray,
        "3_opened": opened,
        "4_subtracted": subtracted,
        "5_binary": binary
    }
    return steps