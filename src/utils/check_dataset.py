from pathlib import Path
import sys

try:
    from config import ASL_DATASET_PATH
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from config import ASL_DATASET_PATH

print("=" * 50)
print("Dataset Path:")
print(ASL_DATASET_PATH)
print("=" * 50)

classes = []

for folder in ASL_DATASET_PATH.iterdir():

    if folder.is_dir():
        classes.append(folder.name)

classes.sort()

print("\nFound Classes:\n")

for c in classes:
    print(c)

print("\nTotal Classes:", len(classes))