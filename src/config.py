from pathlib import Path

# Path to the extracted ASL dataset folder
ASL_DATASET_PATH = Path(
    r"C:\Users\anjali desai\Downloads\ASL_Alphabet_Dataset\asl_alphabet_train"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LANDMARK_DATASET = PROJECT_ROOT / "data" / "landmarks"
MODEL_DIR = PROJECT_ROOT / "models"

LANDMARK_DATASET.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)