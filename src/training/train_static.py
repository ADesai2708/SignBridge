import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Add src folder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.static_classifier import StaticSignClassifier

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

DATASET = "../preprocessing/data/processed/landmarks.csv"
print("Loading dataset...")

df = pd.read_csv(DATASET)

print(df.head())

print("\nDataset Shape:", df.shape)

# ---------------------------------------------------
# Split Features & Labels
# ---------------------------------------------------

X = df.drop("label", axis=1)

y = df["label"]

print("\nFeature Shape:", X.shape)
print("Label Shape:", y.shape)

# ---------------------------------------------------
# Train/Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

print("\nTraining MLP...")

classifier = StaticSignClassifier()

classifier.fit(X_train, y_train)

print("Training Complete!")

# ---------------------------------------------------
# Evaluate
# ---------------------------------------------------

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy")

print(f"{accuracy*100:.2f}%")

print("\nClassification Report")

print(classification_report(y_test, predictions))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, predictions))

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

MODEL_PATH = Path("models/asl_static_mlp.pkl")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

classifier.save(MODEL_PATH)

print("\nModel Saved")

print(MODEL_PATH)