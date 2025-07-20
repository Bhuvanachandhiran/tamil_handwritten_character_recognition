import pandas as pd
from pathlib import Path

def generate_test_csv(test_dir, output_csv):
    test_dir = Path(test_dir)
    data = []
    
    # Iterate through subfolders (0 to 155, optionally 156)
    for class_folder in test_dir.iterdir():
        if class_folder.is_dir():
            try:
                class_label = int(class_folder.name)  # Folder name is the class label
                # Skip class 156 if not needed (based on tamilchar.txt)
                if class_label == 156:
                    continue
                for img_path in class_folder.glob('*.bmp'):
                    data.append({
                        'ID': img_path.name,
                        'Class Label': class_label
                    })
            except ValueError:
                continue  # Skip non-numeric folder names

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated test.csv with {len(df)} images at {output_csv}")

if __name__ == "__main__":
    generate_test_csv(
        test_dir=r'D:\projects ai\tamil-handwritten-recognition\data\test',
        output_csv=r'D:\projects ai\tamil-handwritten-recognition\data\test.csv'
    )