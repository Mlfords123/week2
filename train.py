import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("iris.csv")
X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Accuracy: {acc:.4f}")

# Save model & metrics
joblib.dump(model, "model.weights.h5")
with open("metrics.csv", "w") as f:
    f.write("accuracy\n")
    f.write(f"{acc:.4f}\n")
