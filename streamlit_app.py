import streamlit as st
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# === Setup ===
nltk.download('vader_lexicon')

# === Load model and constants ===
MODEL_PATH = "food_recognition/food_model.keras"
LABELS = ["bread", "meat", "pasta", "rice", "salad"]
SALT_CONTENT = {"pasta": 0.3, "salad": 0.2, "meat": 0.7, "rice": 0.1, "bread": 0.5}
model = load_model(MODEL_PATH)

# === File paths ===
LOG_CSV = "salt_logs.csv"
LOG_XLSX = "salt_logs.xlsx"
FEEDBACK_CSV = "data/user_feedback.csv"

# === Create CSV logs if they don't exist ===
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "food", "confidence", "salt_content"])

if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "feedback", "sentiment"])

# === Prediction Function ===
def predict_image(image):
    img = image.resize((224, 224)).convert("RGB")
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

# === Streamlit UI ===
st.set_page_config(page_title="Smart Spoon AI", layout="centered")
st.title("🍽️ Smart Spoon Food Recognition")

uploaded_image = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, confidence, salt, top3 = predict_image(image)
    st.success(f"🍱 Predicted Food: **{label.capitalize()}**")
    st.info(f"📊 Confidence: `{confidence * 100:.2f}%`")
    st.warning(f"🧂 Estimated Salt Content: `{salt:.2f}g`")

    st.markdown("#### 🔍 Top 3 Predictions")
    for food, prob in top3:
        st.write(f"• {food.capitalize()}: `{prob * 100:.2f}%`")

    # Log prediction
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            f"{confidence:.2f}",
            f"{salt:.2f}"
        ])
    pd.read_csv(LOG_CSV).to_excel(LOG_XLSX, index=False)

# === Salt Intake Log & Chart ===
st.markdown("### 📈 Salt Intake Log & Trend")
if os.path.exists(LOG_CSV):
    df = pd.read_csv(LOG_CSV)
    if not df.empty:
        st.dataframe(df)
        chart = df.groupby("food")["salt_content"].sum().reset_index()
        chart["salt_content"] = chart["salt_content"].astype(float)
        st.bar_chart(chart.set_index("food"))

# === Feedback Section ===
st.markdown("### 💬 User Feedback Sentiment")

feedback = st.text_area("What did you think of the Smart Spoon experience?")
if st.button("Submit Feedback"):
    if feedback:
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(feedback)["compound"]
        sentiment = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
        with open(FEEDBACK_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                feedback,
                sentiment
            ])
        st.success(f"Thank you! Your feedback was marked as: **{sentiment}**")

# === Feedback Sentiment Chart ===
if os.path.exists(FEEDBACK_CSV):
    fb_df = pd.read_csv(FEEDBACK_CSV)
    if not fb_df.empty:
        st.markdown("#### 📊 Sentiment Summary")

        # Sort sentiment display
        sentiment_order = ["Positive", "Neutral", "Negative"]
        sentiment_counts = fb_df["sentiment"].value_counts().reindex(sentiment_order).fillna(0)
        st.bar_chart(sentiment_counts)
st.markdown("---")
st.markdown("🔬 *Powered by a custom-trained MobileNetV2 model*")
