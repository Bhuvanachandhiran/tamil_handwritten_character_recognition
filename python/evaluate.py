import torch
import pandas as pd
from ocr_engine import TamilOCR
from preprocess import TamilDataset
from torch.utils.data import DataLoader

def evaluate_model():
    ocr = TamilOCR(
        model_path=r'D:\projects ai\tamil-handwritten-recognition\data\tamilnet_model.pt',
        char_map_path=r'D:\projects ai\tamil-handwritten-recognition\data\tamilchar.txt'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = TamilDataset(
        csv_file=r'D:\projects ai\tamil-handwritten-recognition\data\test.csv',
        img_dir=r'D:\projects ai\tamil-handwritten-recognition\data\test',
        is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predictions = []
    ground_truth = []
    model = ocr.model
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())

    # Map indices to characters
    char_map = ocr.char_map
    pred_chars = [char_map.get(pred, 'Unknown') for pred in predictions]
    true_chars = [char_map.get(true, 'Unknown') for true in ground_truth]

    # Save results
    results = pd.DataFrame({
        'Image': test_dataset.data['ID'],
        'True Label': ground_truth,
        'True Char': true_chars,
        'Predicted Label': predictions,
        'Predicted Char': pred_chars
    })
    results.to_csv(r'D:\projects ai\tamil-handwritten-recognition\data\evaluation_results.csv', index=False)
    print("Evaluation results saved to data/evaluation_results.csv")

    # Calculate accuracy
    accuracy = sum(p == t for p, t in zip(predictions, ground_truth)) / len(predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()