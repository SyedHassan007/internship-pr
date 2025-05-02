import pandas as pd
import matplotlib.pyplot as plt

# Load usage logs
data = pd.read_csv("data/usage_logs.csv")

print("\n--- Usage Logs Head ---")
print(data.head())

# Plot stimulation level vs user satisfaction
plt.scatter(data['stimulation_level'], data['satisfaction_score'], c='blue')
plt.title("Stimulation Level vs User Satisfaction")
plt.xlabel("Stimulation Level")
plt.ylabel("Satisfaction Score")
plt.grid(True)
plt.savefig("market_analysis/stimulation_vs_satisfaction.png")
plt.show()

# Correlation
correlation = data[['stimulation_level', 'satisfaction_score']].corr()
print("\nCorrelation:\n", correlation)
