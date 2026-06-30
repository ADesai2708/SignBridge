import cv2
import sys
from pathlib import Path

# Add src folder to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from config import ASL_DATASET_PATH

# Pick one sample image from class A
image_path = next((ASL_DATASET_PATH / "A").glob("*"))

print(f"Reading: {image_path.name}")

image = cv2.imread(str(image_path))

extractor = MediaPipeLandmarkExtractor()

features = extractor.extract(image)

if features is None:
    print("No hand detected!")
else:
    print("Success!")
    print("Shape:", features.shape)
    print(features[:10])   # Show first 10 values