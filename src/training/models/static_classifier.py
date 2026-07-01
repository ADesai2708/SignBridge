def predict_single(self, features):

    probabilities = self.model.predict_proba([features])[0]

    best_index = probabilities.argmax()

    label = self.encoder.inverse_transform([best_index])[0]

    confidence = probabilities[best_index]

    return label, confidence