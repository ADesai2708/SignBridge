import numpy as np
def predict_single(self, features):

    probabilities = self.model.predict_proba([features])[0]

    best_index = probabilities.argmax()

    label = self.encoder.inverse_transform([best_index])[0]

    confidence = probabilities[best_index]

    return label, confidence
def predict_top3(self, features):

    probabilities = self.model.predict_proba([features])[0]

    indices = np.argsort(probabilities)[::-1][:3]

    predictions = []

    for idx in indices:

        predictions.append({

            "label": self.encoder.inverse_transform([idx])[0],

            "confidence": float(probabilities[idx])

        })

    return predictions