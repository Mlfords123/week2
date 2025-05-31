import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

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

# Save model directly to root
joblib.dump(model, "model.joblib")

# Save accuracy to metrics file
accuracy = model.score(X_test, y_test)
with open("metrics.txt", "w") as f:
    f.write(f"accuracy: {accuracy:.4f}\n")

print(f"Model saved to model.joblib with accuracy: {accuracy:.4f}")
