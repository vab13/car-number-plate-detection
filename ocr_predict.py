from ultralytics import YOLO
import cv2
import pytesseract
import os
import re
import numpy as np

# ---------------------------
# 🔧 CONFIG
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"

model = YOLO("models/best.pt")

source_folder = "dataset/car_number_plate/test/images"
output_folder = "results/ocr"
debug_folder = "results/debug"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# ---------------------------
# 🚀 PROCESS IMAGES
# ---------------------------
for img_name in os.listdir(source_folder):
    img_path = os.path.join(source_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)

    for r in results:
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):

            # ✅ Filter weak detections
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box)

            # ---------------------------
            # ✂️ Crop plate
            # ---------------------------
            crop = img[y1:y2, x1:x2]

            # ✅ Save debug crop
            debug_name = f"{img_name}_{x1}_{y1}.jpg"
            cv2.imwrite(os.path.join(debug_folder, debug_name), crop)

            # ---------------------------
            # 🔍 Resize (VERY IMPORTANT)
            # ---------------------------
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            # ---------------------------
            # 🧠 Preprocessing
            # ---------------------------
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            # ✅ Adaptive threshold (better than normal threshold)
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

            # ✅ Morphology (clean noise)
            kernel = np.ones((3, 3), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            # ✅ Sharpen
            sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            gray = cv2.filter2D(gray, -1, sharp_kernel)

            # ---------------------------
            # 🔤 OCR
            # ---------------------------
            text = pytesseract.image_to_string(
                gray,
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            # ---------------------------
            # 🧹 Clean text
            # ---------------------------
            text = re.sub('[^A-Z0-9]', '', text)

            # Fix common OCR mistakes
            text = text.replace("O", "0")
            text = text.replace("I", "1")
            text = text.replace("Z", "2")
            text = text.replace("S", "5")
            text = text.replace("B", "8")

            # ---------------------------
            # 🇮🇳 Validate Indian Plate
            # ---------------------------
            pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}'
            match = re.search(pattern, text)

            if match:
                final_text = match.group()
            else:
                final_text = text  # keep useful OCR

            # ---------------------------
            # 🖨️ Output
            # ---------------------------
            print(f"{img_name} → OCR: {text} | FINAL: {final_text}")

            # ---------------------------
            # 💾 Save result
            # ---------------------------
            save_name = f"{img_name}_{x1}_{y1}.jpg"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, crop)

print("✅ OCR completed")