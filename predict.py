from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Run prediction
results = model.predict(
    source="dataset/car_number_plate/test/images",  # adjust if needed
    save=True,
    project="results",
    name="output",
    conf=0.25
)

print("✅ Prediction done")