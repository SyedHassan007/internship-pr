import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load survey data
df = pd.read_csv("data/survey_responses.csv")

# Display basic insights
print("\n--- Survey Summary ---")
print(df.describe(include='all'))

# Visualize user preferences by age
graph = sns.barplot(data=df, x="age_group", y="preference_score", hue="preferred_taste")
graph.set_title("User Taste Preferences by Age Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("market_analysis/survey_insights.png")
plt.show()
