from ultralytics import YOLO
import cv2
import easyocr
import re
import numpy as np

# ---------------------------
# 🔧 CONFIG
# ---------------------------
model = YOLO("models/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Press 'q' to quit")

# ---------------------------
# 🎥 LIVE DETECTION
# ---------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame)

    for r in results:
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):

            if conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box)

            # Skip tiny detections
            if (x2 - x1) < 50 or (y2 - y1) < 20:
                continue

            # Crop with padding
            pad = 15
            crop = frame[max(0,y1-pad):y2+pad, max(0,x1-pad):x2+pad]

            # Resize
            crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

            # Preprocess
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # OCR
            result = reader.readtext(gray)

            text = ""
            if len(result) > 0:
                best = max(result, key=lambda x: x[2])
                text = best[1]

            # Clean text
            text = re.sub('[^A-Z0-9]', '', text.upper())

            # Fix common errors
            text = text.replace("O", "0")
            text = text.replace("I", "1")
            text = text.replace("Z", "2")
            text = text.replace("S", "5")
            text = text.replace("B", "8")

            # Remove garbage
            if len(text) < 6:
                continue

            text = text[:10]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Show text
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # Show frame
    cv2.imshow("🚗 ANPR Live Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()