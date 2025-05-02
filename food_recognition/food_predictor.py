import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

# === Load model and settings ===
MODEL_PATH = "food_recognition/food_model.keras"
LABELS = ["bread", "meat", "pasta", "rice", "salad"]
SALT_CONTENT = {
    "pasta": 0.3,
    "salad": 0.2,
    "meat": 0.7,
    "rice": 0.1,
    "bread": 0.5
}

model = load_model(MODEL_PATH)

def predict_food(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    class_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))
    predicted_label = LABELS[class_idx]
    salt = SALT_CONTENT.get(predicted_label, 0.0)

    top3 = sorted(zip(LABELS, predictions), key=lambda x: x[1], reverse=True)[:3]
    return predicted_label, confidence, salt, top3

# === Batch Test All Images in test_samples/ ===
TEST_DIR = "food_recognition/test_samples"

if os.path.exists(TEST_DIR):
    for file in os.listdir(TEST_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png','.webp')):
            path = os.path.join(TEST_DIR, file)
            label, confidence, salt, top3 = predict_food(path)
            print(f"\n🖼️ {file} ➜ {label.capitalize()} ({confidence*100:.2f}%) | Salt: {salt:.2f}g")
            print("🔍 Top 3 Predictions:")
            for food, prob in top3:
                print(f"   - {food.capitalize()}: {prob*100:.2f}%")
else:
    print("❌ Test directory not found.")
