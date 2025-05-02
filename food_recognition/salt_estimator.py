# salt_estimator.py

# Accurate salt values for known food labels
FOOD_SALT_DATA = {
    "rice": 0.1,
    "pasta": 0.3,
    "salad": 0.2,
    "meat": 0.7,
    "bread": 0.5
}

def estimate_salt(food_label):
    return FOOD_SALT_DATA.get(food_label.lower(), 0.0)

if __name__ == "__main__":
    test_label = "meat"
    salt = estimate_salt(test_label)
    print(f"Estimated salt content for {test_label}: {salt:.2f} grams")

