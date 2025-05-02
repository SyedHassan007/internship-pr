import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

# Download VADER lexicon
nltk.download('vader_lexicon')

# ✅ Load feedback data
fb_df = pd.read_csv("data/user_feedback.csv", encoding='utf-8')

# ✅ Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# ✅ Analyze sentiment
fb_df['sentiment_score'] = fb_df['feedback'].apply(lambda x: sia.polarity_scores(x)['compound'])

# ✅ Classify feedback as Positive/Negative/Neutral
def classify(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

fb_df['sentiment'] = fb_df['sentiment_score'].apply(classify)

# ✅ Show sentiment results
print(fb_df[['feedback', 'sentiment_score', 'sentiment']])

# ✅ Seaborn bar chart for sentiment distribution
sns.countplot(data=fb_df, x='sentiment', palette='Set2')
plt.title('User Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
