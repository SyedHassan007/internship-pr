import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# === Setup ===
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# === Load Feedback CSV ===
feedback_path = "user_feedback.csv"

try:
    df = pd.read_csv(feedback_path)
except FileNotFoundError:
    print("❌ Feedback CSV not found.")
    exit()

# === Analyze or Re-analyze Sentiment Scores ===
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

if "sentiment_score" not in df.columns:
    df["sentiment_score"] = df["feedback"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
if "sentiment" not in df.columns:
    df["sentiment"] = df["sentiment_score"].apply(classify_sentiment)

# === Display Result ===
print("\n--- Feedback Summary ---\n")
print(df[["feedback", "sentiment_score", "sentiment"]])

# === Sentiment Bar Plot ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="sentiment", palette="pastel")
plt.title("User Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()
