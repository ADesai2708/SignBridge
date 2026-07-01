import cv2
import mediapipe as mp
import numpy as np


class MediaPipeLandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    def normalize_landmarks(self, hand):
        """
        Normalize landmarks relative to the wrist.
        """

        wrist = hand.landmark[0]

        features = []

        for landmark in hand.landmark:
            x = landmark.x - wrist.x
            y = landmark.y - wrist.y
            z = landmark.z - wrist.z

            features.extend([x, y, z])

        return np.array(features, dtype=np.float32)

    def extract(self, image):

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hand
        results = self.hands.process(rgb)

        # If no hand found
        if not results.multi_hand_landmarks:
            return None

        # First detected hand
        hand = results.multi_hand_landmarks[0]

        features = []

        # Extract x, y, z of all 21 landmarks
        for landmark in hand.landmark:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        return np.array(features, dtype=np.float32)