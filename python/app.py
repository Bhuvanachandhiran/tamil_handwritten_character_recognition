import sys
import os
from ocr_engine import TamilOCR

def main(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    try:
        ocr = TamilOCR(
            model_path=r'D:\projects ai\tamil-handwritten-recognition\data\tamilnet_model.pt',
            char_map_path=r'D:\projects ai\tamil-handwritten-recognition\data\tamilchar.txt'
        )
        predicted_char = ocr.predict(image_path)
        print(f"Predicted character: {predicted_char}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <image_path>")
        print("Example: python app.py D:\\projects ai\\tamil-handwritten-recognition\\data\\test\\0\\12290.bmp")
        sys.exit(1)
    main(sys.argv[1])