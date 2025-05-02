# sentiment_analysis/behavior_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Simulated dataset: each row is a user session
df = pd.read_csv('data/usage_logs.csv')

# Assume columns: 'stimulation_level', 'food_type', 'duration_sec', 'user_age', 'satisfaction_label'
df['food_type'] = df['food_type'].astype('category').cat.codes  # Encode food type as numeric

# Features and label
X = df[['stimulation_level', 'food_type', 'duration_sec', 'user_age']]
y = df['satisfaction_score']  # 0 = Not Satisfied, 1 = Satisfied

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
import matplotlib.pyplot as plt

feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', title='Feature Importance in Behavior Prediction')
plt.tight_layout()
plt.show()