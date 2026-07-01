import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from models.static_classifier import StaticSignClassifier

extractor = MediaPipeLandmarkExtractor()
classifier = StaticSignClassifier()

MODEL_PATH = ROOT / "models" / "asl_static_mlp.pkl"
print("Loading model from:", MODEL_PATH)

classifier.load(str(MODEL_PATH))

cap = cv2.VideoCapture(0)

print("Press Q to Quit")

while True:
    success, frame = cap.read()

    if not success:
        break

    features = extractor.extract(frame)

    if features is not None:
        prediction, confidence = classifier.predict_single(features)

        cv2.putText(
            frame,
            f"{prediction} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("SignBridge", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()