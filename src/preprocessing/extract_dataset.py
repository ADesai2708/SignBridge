import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from config import ASL_DATASET_PATH

# ---------------------------------------
# Configuration
# ---------------------------------------

OUTPUT_FILE = Path("data/processed/landmarks.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

extractor = MediaPipeLandmarkExtractor()

dataset = []

# Skip these classes for now
SKIP_CLASSES = {"del", "nothing", "space"}

classes = [
    ASL_DATASET_PATH / "A",
    ASL_DATASET_PATH / "B",
    ASL_DATASET_PATH / "C"
]

print(f"\nFound {len(classes)} classes")

# ---------------------------------------
# Process dataset
# ---------------------------------------

processed = 0
skipped = 0

for class_folder in classes:

    label = class_folder.name

    print(f"\nProcessing Class: {label}")

    images = list(class_folder.glob("*"))

    for image_path in tqdm(images):

        image = cv2.imread(str(image_path))

        if image is None:
            skipped += 1
            continue

        features = extractor.extract(image)

        if features is None:
            skipped += 1
            continue

        row = features.tolist()
        row.append(label)

        dataset.append(row)

        processed += 1

# ---------------------------------------
# Save CSV
# ---------------------------------------

columns = []

for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

columns.append("label")

df = pd.DataFrame(dataset, columns=columns)

df.to_csv(OUTPUT_FILE, index=False)

print("\n==========================")
print("Dataset Extraction Complete")
print("==========================")
print(f"Processed : {processed}")
print(f"Skipped   : {skipped}")
print(f"Saved To  : {OUTPUT_FILE}")
print(f"CSV Shape : {df.shape}")