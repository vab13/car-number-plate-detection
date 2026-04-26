from ultralytics import YOLO
import cv2
import easyocr
import os
import re
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
model = YOLO("models/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

source_folder = "dataset/car_number_plate/test/images"
output_folder = "results/ocr"
debug_folder = "results/debug"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# ---------------------------
# PROCESS IMAGES
# ---------------------------
for img_name in os.listdir(source_folder):
    img_path = os.path.join(source_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)

    for r in results:
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):

            # 1. Confidence filter
            if conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box)

            # ✅ 2. Size filter (remove tiny detections)
            w = x2 - x1
            h = y2 - y1
            if w < 50 or h < 20:
                continue

            # ---------------------------
            # Crop with padding
            # ---------------------------
            pad = 15
            crop = img[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]

            # Save debug crop
            debug_name = f"{img_name}_{x1}_{y1}.jpg"
            cv2.imwrite(os.path.join(debug_folder, debug_name), crop)

            # ---------------------------
            # Resize
            # ---------------------------
            crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

            # ---------------------------
            # Preprocessing
            # ---------------------------
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # ---------------------------
            # EasyOCR
            # ---------------------------
            result = reader.readtext(gray)

            text = ""
            if len(result) > 0:
                best = max(result, key=lambda x: x[2])
                text = best[1]

            # ---------------------------
            # Clean text
            # ---------------------------
            text = re.sub('[^A-Z0-9]', '', text.upper())

            # Fix common OCR mistakes
            text = text.replace("O", "0")
            text = text.replace("I", "1")
            text = text.replace("Z", "2")
            text = text.replace("S", "5")
            text = text.replace("B", "8")

            # 3. Fix starting issues (0L → DL)
            if len(text) >= 2:
                if text[0] == '0':
                    text = 'D' + text[1:]

            # 4. Remove short garbage
            if len(text) < 6:
                continue

            # 5. Limit length (avoid long garbage)
            text = text[:10]

            # ---------------------------
            # 🇮🇳 Validate plate
            # ---------------------------
            pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}'
            match = re.search(pattern, text)

            if match:
                final_text = match.group()
            else:
                final_text = text 

            # ---------------------------
            # Output
            # ---------------------------
            print(f"{img_name} → OCR: {text} | FINAL: {final_text}")

            # ---------------------------
            # Save result
            # ---------------------------
            save_name = f"{img_name}_{x1}_{y1}.jpg"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, crop)

print("EasyOCR completed")