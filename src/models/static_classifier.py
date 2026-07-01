import joblib
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


class StaticSignClassifier:

    def __init__(self):

        self.encoder = LabelEncoder()

        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )

    def fit(self, X, y):

        y_encoded = self.encoder.fit_transform(y)

        self.model.fit(X, y_encoded)

    def predict(self, X):

        pred = self.model.predict(X)

        return self.encoder.inverse_transform(pred)

    def predict_proba(self, X):

        probs = self.model.predict_proba(X)

        top3 = np.argsort(probs[0])[::-1][:3]

        results = []

        for idx in top3:

            results.append({
                "label": self.encoder.classes_[idx],
                "confidence": float(probs[0][idx])
            })

        return results

    def save(self, path):

        joblib.dump(
            {
                "model": self.model,
                "encoder": self.encoder
            },
            path
        )

    def load(self, path):

        data = joblib.load(path)

        self.model = data["model"]
        self.encoder = data["encoder"]

    def predict_single(self, features):
        features = np.array(features).reshape(1, -1)

        prediction = self.predict(features)[0]
        confidence = max(self.model.predict_proba(features)[0])

        return prediction, confidence