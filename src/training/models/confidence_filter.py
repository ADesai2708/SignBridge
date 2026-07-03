class ConfidenceFilter:

    def __init__(self, threshold=0.75):

        self.threshold = threshold

    def is_confident(self, prediction):

        return prediction["confidence"] >= self.threshold