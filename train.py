import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load CSV
df = pd.read_csv("iris.csv")

# Split features and target
X = df.drop(columns=["target"])
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")

# Optional: save metrics (if you want to track them with DVC)
accuracy = model.score(X_test, y_test)
with open("metrics.txt", "w") as f:
    f.write(f"accuracy: {accuracy:.4f}\n")

print(f"Model saved to models/model.joblib with accuracy: {accuracy:.4f}")
