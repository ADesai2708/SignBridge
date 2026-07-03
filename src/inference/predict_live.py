import cv2
import sys
from pathlib import Path
import mediapipe as mp
# ------------------------------------
# Project Path
# ------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

# ------------------------------------
# Imports
# ------------------------------------
from extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from models.static_classifier import StaticSignClassifier

# ------------------------------------
# Load Extractor
# ------------------------------------
extractor = MediaPipeLandmarkExtractor()

# ------------------------------------
# Load Trained Model
# ------------------------------------
classifier = StaticSignClassifier()

MODEL_PATH = ROOT / "models" / "asl_static_mlp.pkl"

print("Loading model...")
classifier.load(str(MODEL_PATH))
print("Model Loaded Successfully!")

# ------------------------------------
# Webcam
# ------------------------------------
cap = cv2.VideoCapture(0)

print("Press Q to Quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    # ------------------------------------
    # Extract Features
    # ------------------------------------
    features = extractor.extract(frame)

    if features is not None:

        # Get Top 3 Predictions
        top3 = classifier.predict_top3(features)

        # Best Prediction
        best = top3[0]

        prediction = best["label"]
        confidence = best["confidence"]

        # -----------------------------
        # Prediction
        # -----------------------------
        cv2.putText(
            frame,
            f"Prediction : {prediction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # -----------------------------
        # Confidence
        # -----------------------------
        cv2.putText(
            frame,
            f"Confidence : {confidence*100:.2f}%",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # -----------------------------
        # Top 3 Heading
        # -----------------------------
        cv2.putText(
            frame,
            "Top 3 Predictions",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        y = 155

        # -----------------------------
        # Display Top 3
        # -----------------------------
        for item in top3:

            text = f"{item['label']} : {item['confidence']*100:.2f}%"

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            y += 30

    else:

        cv2.putText(
            frame,
            "No Hand Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # ------------------------------------
    # Show Window
    # ------------------------------------
    cv2.imshow("SignBridge", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()