import pandas as pd

# Load Kaggle dataset files
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = "fake"
real["label"] = "real"

# Combine datasets
data = pd.concat([fake, real])

# Keep only required columns
data = data[["text", "label"]]

# Save final dataset
data.to_csv("final_dataset.csv", index=False)

print("Dataset prepared: final_dataset.csv")
