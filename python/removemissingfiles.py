import pandas as pd
from pathlib import Path

def verify_csv(csv_path, test_dir):
    df = pd.read_csv(csv_path)
    test_dir = Path(test_dir)
    for _, row in df.iterrows():
        img_path = test_dir / str(row['Class Label']) / row['ID']
        if not img_path.exists():
            print(f"Missing image: {img_path}")
        else:
            print(f"Found image: {img_path}")

verify_csv(
    csv_path=r'D:\projects ai\tamil-handwritten-recognition\data\test.csv',
    test_dir=r'D:\projects ai\tamil-handwritten-recognition\data\test'
)