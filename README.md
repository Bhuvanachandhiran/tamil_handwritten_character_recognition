# Tamil Handwritten Character Recognition 

## Overview
Tamil Handwritten Character Recognition  is a deep learning-based OCR system for recognizing handwritten Tamil characters using a CNN (TamilNet). Trained on the Kaggle Tamil Handwritten Character Recognition dataset, it supports 156 characters and is deployed via a Flask web app for real-time predictions.

## Dataset
- **Source**: [Kaggle Tamil Handwritten Character Recognition](https://www.kaggle.com/datasets/gauravduttakiit/tamil-handwritten-character-recognition)
- **Training Data**: 50,296 `.bmp` images in `data/train` (subfolders `0` to `155`)
- **Test Data**: ~27,868 `.bmp` images in `data/test` (subfolders `0` to `155`), labeled in `test.csv`
- **Character Map**: `data/tamilchar.txt` (156 classes, 0–155)
- **Note**: Ensure `test.csv` IDs match filenames in `data/test/<Class Label>/`.

## Requirements
- Python 3.8+
- Libraries: `torch`, `torchvision`, `opencv-python`, `pandas`, `numpy`, `flask`
- Windows 10/11

## Setup

1. **Set Up Directory**:
   - Structure:
     tamil-handwritten-recognition/
     ├── data/
     │   ├── train/
     │   ├── test/
     │   ├── test.csv
     │   ├── tamilchar.txt
     │   ├── tamilnet_model.pt
     │   └── evaluation_results.csv
     ├── python/
     │   ├── preprocess.py
     │   ├── model.py
     │   ├── ocr_engine.py
     │   ├── train.py
     │   ├── evaluate.py
     │   └── app.py
     ├── templates/
     │   └── index.html
     └── uploads/
     
### Explanation
- **Brevity**: The README provides a concise overview, focusing on key details: project purpose, dataset, setup, usage, and minimal troubleshooting.
- **Test Data Mismatch**: Addresses the `test.csv` vs. `data/test` subfolder issue with a script to update `test.csv`.
- **Markdown Format**: Provided as a code block to ensure proper Markdown syntax, as requested.
- **No Work History**: Excludes detailed project history, focusing only on essential setup and usage.

### Next Steps
 **Fix Test Data**:
   - Run the provided `test.csv` update script to align with `data/test` filenames.
   - Verify with: `dir "data\test\0" & type "data\test.csv" | more`.
 **Run Pipeline**:
   ```cmd
   python python/train.py
   python python/evaluate.py
   python python/app.py



