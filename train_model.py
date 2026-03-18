import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data_dir = "data"
X, y = [], []

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        gesture = file.replace(".csv", "")
        samples = pd.read_csv(os.path.join(data_dir, file)).values
        X.extend(samples)
        y.extend([gesture] * len(samples))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

os.makedirs("models", exist_ok=True)
with open("models/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to models/gesture_model.pkl")
